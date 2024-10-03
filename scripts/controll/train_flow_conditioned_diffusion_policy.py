import copy
import datetime
import os

# os.environ["NCCL_DEBUG"] = "INFO"
# os.environ["NCCL_P2P_DISABLE"] = "1"
import pickle
import subprocess

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from accelerate import Accelerator
from diffusers.optimization import get_scheduler
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from im2flow2act.common.utility.model import load_config

dist.init_process_group(
    backend="nccl", init_method="env://", timeout=datetime.timedelta(seconds=5400)
)


@hydra.main(
    version_base=None,
    config_path="../../config/diffusion_policy",
    config_name="train_flow_conditioned_diffusion_policy",
)
def train_diffusion_bc(cfg: DictConfig):
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps
    )
    # # create save dir
    output_dir = HydraConfig.get().runtime.output_dir
    resume = cfg.training.resume
    if resume:
        # overwrite the config with the checkpoint config
        accelerator.print("resume from checkpoint")
        model_path = cfg.training.model_path
        model_ckpt = cfg.training.model_ckpt
        cfg = load_config(model_path)
        # save the checkpoint config to the output dir
        OmegaConf.save(cfg, os.path.join(output_dir, ".hydra", "config.yaml"))
    if accelerator.is_local_main_process:
        wandb.init(project=cfg.project_name)
        wandb.config.update(OmegaConf.to_container(cfg))
        accelerator.print("Logging dir", output_dir)
        ckpt_save_dir = os.path.join(output_dir, "checkpoints")
        state_save_dir = os.path.join(output_dir, "state")
        os.makedirs(ckpt_save_dir, exist_ok=True)
        os.makedirs(state_save_dir, exist_ok=True)

    # max_episode = 5 if cfg.debug else cfg.dataset.max_episode
    dataset = hydra.utils.instantiate(
        cfg.dataset, max_episode=5 if cfg.debug else cfg.dataset.max_episode
    )
    print("Total training samples:", len(dataset))
    # save training data statistics (min, max) for each dim
    stats = dataset.stats
    # open a file for writing in binary mode
    with open(os.path.join(output_dir, "stats.pickle"), "wb") as f:
        # write the dictionary to the file
        pickle.dump(stats, f)
    # create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        shuffle=cfg.training.shuffle,
        pin_memory=cfg.training.pin_memory,
        persistent_workers=cfg.training.persistent_workers,
        drop_last=cfg.training.drop_last,
    )
    print("====================================")
    print("len of dataset", len(dataset))
    print("====================================")

    sample_batch = next(iter(dataloader))
    for k, v in sample_batch.items():
        accelerator.print(k, v.shape)

    model = hydra.utils.instantiate(cfg.model)
    noise_scheduler = hydra.utils.instantiate(cfg.noise_scheduler)
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    num_update_steps_per_epoch = len(dataloader)
    max_train_steps = cfg.training.epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.num_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )
    # Prepare everything with our `accelerator`.
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )
    if resume:
        model_state = os.path.join(model_path, "state", f"epoch_{model_ckpt}")
        accelerator.load_state(model_state)
        print("successfully loaded model from checkpoint!")

    if cfg.training.use_ema:
        ema_model = copy.deepcopy(accelerator.unwrap_model(model))
        ema = hydra.utils.instantiate(cfg.ema, model=ema_model)

    for epoch in range(cfg.training.epochs):
        epoch_loss = []
        epoch_action_loss = []
        epoch_alignment_loss = []
        epoch_proprioception_loss = []

        # batch loop
        for batch in dataloader:
            with accelerator.accumulate(model):
                (
                    initial_frames,
                    initial_flows,
                    current_flows,
                    frames_0,
                    # frames_1,
                    proprioceptions,
                    flows,
                    actions,
                    target_flows,
                    point_clouds,
                    # target_proprioceptions,
                ) = (
                    batch["initial_frame"],
                    batch["initial_flow"],
                    batch["current_flow"],
                    batch["camera_0"],
                    # batch["camera_6"],
                    batch["proprioception"],
                    batch["episode_flow_plan"],
                    batch["action"],
                    batch["target_flow"],
                    batch["point_cloud"],
                    # batch["target_proprioception"],
                )
                initial_frames = initial_frames.to(accelerator.device)
                initial_flows = initial_flows.to(accelerator.device)
                current_flows = current_flows.to(accelerator.device)
                frames_0 = frames_0.to(accelerator.device)
                # frames_1 = frames_1.to(accelerator.device)
                proprioceptions = proprioceptions.to(accelerator.device)
                flows = flows.to(accelerator.device)
                actions = actions.to(accelerator.device)
                target_flows = target_flows.to(accelerator.device)
                point_clouds = point_clouds.to(accelerator.device)
                noise = torch.randn(actions.shape, device=accelerator.device)
                bsz = flows.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=accelerator.device,
                ).long()
                noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)
                noise_pred, predict_plan, target_plan, proprioception_prediction = (
                    model(
                        noisy_actions,
                        timesteps,
                        initial_frames,
                        frames_0,
                        # frames_1,
                        None,
                        proprioceptions,
                        flows,
                        initial_flows,
                        current_flows,
                        target_flows,
                        point_clouds,
                    )
                )
                if proprioception_prediction is not None:
                    proprioception_loss = nn.functional.mse_loss(
                        proprioception_prediction, proprioceptions[:, -1, :]
                    )
                else:
                    proprioception_loss = 0

                if cfg.apply_alignment_loss:
                    alignment_loss = nn.MSELoss()(predict_plan, target_plan)
                else:
                    alignment_loss = 0
                # diffusion loss
                action_loss = nn.functional.mse_loss(noise_pred, noise)

                # total loss
                loss = (
                    action_loss
                    + cfg.alignment_loss_coef * alignment_loss
                    + cfg.proprioception_loss_coef * proprioception_loss
                )

                # optimize
                optimizer.zero_grad()
                accelerator.backward(loss)
                lr_scheduler.step()
                optimizer.step()
                # update ema
                if cfg.training.use_ema:
                    ema.step(accelerator.unwrap_model(model))
                # logging
                epoch_loss.append(loss.item())
                epoch_action_loss.append(action_loss.item())
                epoch_alignment_loss.append(
                    alignment_loss.item() if cfg.apply_alignment_loss else 0
                )
                if proprioception_prediction is not None:
                    epoch_proprioception_loss.append(proprioception_loss.item())
                else:
                    epoch_proprioception_loss.append(0)
        if accelerator.is_local_main_process and not cfg.debug:
            wandb.log(
                {
                    "epoch loss": np.mean(epoch_loss),
                    "epoch action loss": np.mean(epoch_action_loss),
                    "epoch_alignment_loss": np.mean(epoch_alignment_loss),
                    "epoch_proprioception_loss": np.mean(epoch_proprioception_loss),
                }
            )
        if epoch % cfg.training.ckpt_frequency == 0 and epoch > 0:
            accelerator.wait_for_everyone()
            if accelerator.is_local_main_process:
                ckpt_model = accelerator.unwrap_model(model)
                accelerator.save(
                    ckpt_model.state_dict(),
                    os.path.join(ckpt_save_dir, f"epoch_{epoch}.ckpt"),
                )
                accelerator.print(f"Saved checkpoint at epoch {epoch}.")
                if cfg.training.use_ema:
                    accelerator.save(
                        ema_model.state_dict(),
                        os.path.join(ckpt_save_dir, f"ema_epoch_{epoch}.ckpt"),
                    )
                    accelerator.print(f"Saved ema checkpoint at epoch {epoch}.")
                # also save the state
                accelerator.save_state(
                    output_dir=os.path.join(state_save_dir, f"epoch_{epoch}")
                )
                accelerator.print(f"Saved state checkpoint at epoch {epoch}.")

        if (
            accelerator.is_local_main_process
            and epoch % cfg.evaluation.eval_frequency == 0
            and epoch > 0
        ):
            print("Enter evaluation loop.")
            torch.cuda.empty_cache()
            processes = []
            process = subprocess.Popen(
                [
                    "python",
                    "evaluate_flow_diffusion_policy.py",
                    f"model_path={output_dir}",
                    f"ckpt={epoch}",
                ],
            )
            processes.append(process)
            for process in processes:
                process.wait()


if __name__ == "__main__":
    train_diffusion_bc()
