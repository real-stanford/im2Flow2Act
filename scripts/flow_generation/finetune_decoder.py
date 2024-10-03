import math
import os

os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_P2P_DISABLE"] = "1"
import hydra
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from diffusers import AutoencoderKL
from diffusers.optimization import get_scheduler
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from im2flow2act.flow_generation.ae_inference import ae_flow_inference


@hydra.main(
    version_base=None,
    config_path="../../config/flow_generation",
    config_name="finetune_decoder",
)
def train(cfg: DictConfig):
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps
    )
    output_dir = HydraConfig.get().runtime.output_dir
    if accelerator.is_local_main_process:
        wandb.init(project=cfg.project_name)
        wandb.config.update(OmegaConf.to_container(cfg))
        accelerator.print("Logging dir", output_dir)
        ckpt_save_dir = os.path.join(output_dir, "checkpoints")
        eval_save_dir = os.path.join(output_dir, "evaluations")
        # state_save_dir = os.path.join(output_dir, "state")
        os.makedirs(ckpt_save_dir, exist_ok=True)

    vae = AutoencoderKL.from_pretrained(cfg.pretrained_model_path, subfolder="vae")
    # vae = AutoencoderKL.from_pretrained(cfg.pretrained_model_path)
    # freeze encoder
    for param in vae.parameters():
        param.requires_grad_(False)
    # enable decoder gradient
    for param in vae.decoder.parameters():
        param.requires_grad_(True)
    for param in vae.post_quant_conv.parameters():
        param.requires_grad_(True)
    trainable_parameters = filter(lambda p: p.requires_grad, vae.parameters())

    optimizer = torch.optim.AdamW(
        vae.parameters(),
        lr=cfg.optimizer.learning_rate,
        betas=(cfg.optimizer.adam_beta1, cfg.optimizer.adam_beta2),
        weight_decay=cfg.optimizer.adam_weight_decay,
        eps=cfg.optimizer.adam_epsilon,
    )
    dataset = hydra.utils.instantiate(cfg.dataset)
    if accelerator.is_local_main_process:
        accelerator.print("datatset len:", len(dataset))
    evalulation_datasets = [
        hydra.utils.instantiate(cfg.evaluation.evalulation_dataset, data_pathes=[d])
        for d in cfg.evaluation.evalulation_dataset.data_pathes
    ]
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        shuffle=cfg.training.shuffle,
        pin_memory=cfg.training.pin_memory,
        persistent_workers=cfg.training.persistent_workers,
        drop_last=cfg.training.drop_last,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.training.gradient_accumulation_steps
    )
    if cfg.training.max_train_steps is None:
        max_train_steps = cfg.training.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        cfg.training.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.training.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )
    # Prepare everything with our `accelerator`.
    vae, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        vae, optimizer, train_dataloader, lr_scheduler
    )
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.training.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        max_train_steps = cfg.training.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    cfg.training.num_train_epochs = math.ceil(
        max_train_steps / num_update_steps_per_epoch
    )
    # Train!
    total_batch_size = (
        cfg.training.batch_size
        * accelerator.num_processes
        * cfg.training.gradient_accumulation_steps
    )
    if accelerator.is_local_main_process:
        accelerator.print("***** Running training *****")
        accelerator.print(f"  Num examples = {len(dataset)}")
        accelerator.print(f"  Num Epochs = {cfg.training.num_train_epochs}")
        accelerator.print(
            f"  Instantaneous batch size per device = {cfg.training.batch_size}"
        )
        accelerator.print(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        accelerator.print(
            f"  Gradient Accumulation steps = {cfg.training.gradient_accumulation_steps}"
        )
        accelerator.print(f"  Total optimization steps = {max_train_steps}")
        print("Start training...")

    for epoch in range(cfg.training.num_train_epochs):
        vae.train()
        epoch_loss = []
        for step, batch in enumerate(train_dataloader):
            point_tracking_sequence = batch["point_tracking_sequence"]  # (B,C,N1,N2)
            recover_sequence = vae.forward(
                point_tracking_sequence,
                sample_posterior=True,
            ).sample  # (B,C,N1,N2)
            loss = F.mse_loss(
                recover_sequence, point_tracking_sequence, reduction="mean"
            )
            # Backpropagate
            accelerator.backward(loss)
            # if accelerator.sync_gradients:
            #     params_to_clip = trainable_parameters
            #     accelerator.clip_grad_norm_(params_to_clip, cfg.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            # logging
            epoch_loss.append(loss.item())

        if accelerator.is_local_main_process and not cfg.debug:
            wandb.log(
                {
                    "epoch loss": np.mean(epoch_loss),
                    "epoch": epoch,
                }
            )
        if epoch % cfg.training.ckpt_frequency == 0 and epoch > 0:
            accelerator.wait_for_everyone()
            if accelerator.is_local_main_process:
                ckpt_save_path = os.path.join(ckpt_save_dir, f"epoch_{epoch}")
                os.makedirs(ckpt_save_path, exist_ok=True)
                ckpt_vae = accelerator.unwrap_model(vae)
                ckpt_vae.save_pretrained(ckpt_save_path)
                accelerator.print(f"Saved checkpoint at epoch {epoch}.")

        if epoch % cfg.evaluation.eval_frequency == 0 and epoch > 0:
            accelerator.wait_for_everyone()
            accelerator.print(f"Evaluate at epoch {epoch}.")
            if accelerator.is_local_main_process:
                eval_vae = accelerator.unwrap_model(vae)
                for dth, eval_dataset in enumerate(evalulation_datasets):
                    cfg.evaluation.evaluation_save_path = os.path.join(
                        eval_save_dir, f"epoch_{epoch}", f"dataset_{dth}"
                    )
                    os.makedirs(cfg.evaluation.evaluation_save_path, exist_ok=True)
                    ae_flow_inference(
                        cfg.evaluation,
                        eval_vae,
                        eval_dataset,
                        wandb_log=True,
                        prefix=f"dataset_{dth}",
                    )


if __name__ == "__main__":
    train()
