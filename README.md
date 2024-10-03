# Im2Flow2Act: 
**Conference on Robot Learning 2024**

<sup>1,</sup><sup>2,</sup><sup>3</sup>[Mengda Xu](https://mengdaxu.github.io/),  <sup>1,</sup><sup>2,</sup>[Zhenjia Xu](https://www.zhenjiaxu.com/), <sup>1</sup>[Yinghao Xu](https://justimyhxu.github.io/),  <sup>1,</sup><sup>2</sup>[Cheng Chi](https://cheng-chi.github.io/), <sup>1</sup>[Gordon Wetzstein](https://stanford.edu/~gordonwz/), <sup>3,</sup><sup>4</sup>[Manuela Veloso](https://www.cs.cmu.edu/~mmv/),  <sup>1,</sup><sup>2</sup>[Shuran Song](https://shurans.github.io/)


<sup>1</sup>Stanford University, <sup>2</sup>Columbia University,  <sup>3</sup>JP Morgan AI Research,<sup>4</sup>CMU  


[Project Page](https://im-flow-act.github.io/)|[arxiv](https://www.arxiv.org/abs/2407.15208)

This repository contains code for training and evaluating Im2Flow2Act in both simulation and real-world settings.

## 🚀 Installation

Follow these steps to install `Im2Flow2Act`:

1. Create and activate the conda environment:
   ```bash
   cd im2flow2act
   conda env create -f environment.yml
   conda activate im2flow2act
   ```
2. Set the DEV_PATH and export python path in your bashrc or zshrc. 
    ```bash
   export DEV_PATH="/parent/directory/of/im2flow2act"
   export PYTHONPATH="$PYTHONPATH:$DEV_PATH/im2flow2act"
    ```
3. Download pretrain weights for StableDiffusion 1.5. You may download the weight from the official repo or you can  download it from our [website](https://real.stanford.edu/im2flow2act/pretrain_weights/). Put the StableDiffusion pretrain weight under im2flow2act/pretrain_weights.

### 📦 Dataset
The dataset can be downloaded by

```bash
mkdir data
cd data 
wget https://real.stanford.edu/im2flow2act/data/simulation_evaluation.zip # evaluation dataset 
wget https://real.stanford.edu/im2flow2act/data/simulated_play/articulated.zip # policy articulated object training data
wget https://real.stanford.edu/im2flow2act/data/simulated_play/deformable.zip # policy deformable object training data
wget https://real.stanford.edu/im2flow2act/data/simulated_play/rigid.zip # policy rigid object training data
wget https://real.stanford.edu/im2flow2act/data/simulation_sphere_demonstration.zip # simulated sphere demonstration 
wget https://real.stanford.edu/im2flow2act/data/realworld_human_demonstration.zip # real-world human demonstration
```
The dataset contains several components. The `simulated_play` dataset contains the play data for rigid, articulated, and deformable objects. The `simulation_sphere_demonstration` contains the sphere agent’s demonstration on specific tasks, i.e., pick&place, pouring, drawer opening. The `realworld_human_demonstration` contains the human demonstration for the same tasks but in the real world. You can find more information at [Dataset Details](#Dataset-Details). The downloaded dataset already contains bounding box, SAM mask and tracked flows. `simulation_evaluation` is used to evalute both manipulation policy and flow generation model. 
```bash
.
├── realworld_human_demonstration
├── simulated_play
├── simulation_evaluation
└── simulation_sphere_demonstration
```

### Download Checkpoints
To reproduce our simulation experimental results in the paper, you may downlaod the checkpoints for both flow generation and manipulation policy. 
```bash
wget https://real.stanford.edu/im2flow2act/checkpoints.zip # include checkpoints for both policy and flow generation
```
Once downloaded, please refer to [Evaluation](#Evaluation) for running the model.

The folder structure should be as followed once you complete the above steps:
```bash
.
├── checkpoints
├── config
├── data
├── data_local
├── environment.yml
├── im2flow2act
├── LICENSE
├── pretrain_weights
├── README.md
├── scripts
└── tapnet
```

## Visualization
You can visualize the flows by nevigating to `scripts/data_script`:
```bash
python viz_all.py
```
You might need to change dataset path in the `viz_pathes`. To visualize the simulatated play dataset, use `--viz_sam`. You can change the minimum distance a keypoint travels on the image space by specify `viz_thresholds`. Only the keypoints moves more than the threshold will be visualized. To visualize the real-world task demonstration, use `'--viz_bbox'` and set the `viz_thresholds` to 0. Please check [here](https://github.com/mengdaxu/im2Flow2Act/blob/main/scripts/data_script/visualization.md) for details.

## 🚴‍♂️ Training
The training for flow generation and flow-conditioned policy is independent. You can train and evaluate each component separately. However, to evaluate the complete im2flow2act system, please refer to [Evaluation](#Evaluation). We use [Accelerate](https://huggingface.co/docs/accelerate/en/index) for multi-gpu training and set `mixed_precision='fp16'`. 

### Flow Generation
The scripts for training flow generation are located at scripts/flow_generation. You can either use Simulated task demonstration or Real-world task demonstration to train the model. However, to evaluate the complete system in simulation, you need to train with the simulated task demonstration dataset.

#### Step 1
Finetune the decoder from StableDiffusion:
```bash
accelerate launch finetune_decoder.py
```
#### Step 2
Train the flow generation model based on [Animatediff](https://animatediff.github.io/):
```bash
accelerate launch train_flow_generation.py
```
The model will be evaluated every 200 epochs and the results will be logged by Weights&Biases. Additionally, we log the generated flow and ground truth flow under `experiment/flow_generation/yyyy-mm-dd-ss/evaluations/epoch_x`:
```bash
dataset_0
├── generated_flow_0.gif
├── gt_flow_0.gif
```
### Flow Conditioned Policy
The scripts for training flow-conditioned policy is located at `scripts/controll`.
To train the policy:
```bash
accelerate launch train_flow_conditioned_diffusion_policy.py
```
During the training, the policy will be evaluated every 100 epochs with ground truth flow. You can change the frequency by modifying `training.ckpt_frequency` in the config file. You will need a gpu with at least 24GB memory to run the online point tracking and policy inference at the same time. The evaluation results will be saved at the policy folder:
```bash
.
├── episode_0
│   ├── action
│   ├── camera_0
│   ├── camera_0.mp4
│   ├── info
│   ├── proprioception
│   ├── qpos
│   └── qvel
├── episode_0_debug_pts.ply
├── episode_0_online_point_tracking_sequence.npy
├── episode_0_online_tracking.gif
├── episode_0_vis_pts.ply
```
- `episode_x_vis_pts.ply`: It contains the mesh for the initial scene. You can visualize it by software like [meshlab](https://www.meshlab.net/)
- `episode_x_vis_pts.ply`: It contains the mesh for the selected object keypoints. 
- `episode_x_online_tracking`: Online tracking for the selected object keypoints during the inference time.
- `episode_x_online_point_tracking_sequence.npy`: The numeric value for online tracking during the inference time.



<a name="Evaluation"></a>
## 🏂 Evaluation
### Evaluate Manipulation Policy
You can directly evaluate the trained policy by 
```bash
python evalute_flow_diffusion_policy.py
```
The quantitative results are stored in `success_count.json`. Notice, for cloth folding, you need to manully inspect the results.  
### Evaluate Complete System
To evaluate the complete system of im2flow2act, we begin by generating task flow from an initial image and a task description. You need the object bounding box to start, which we have already provided in the downloaded dataset. You can generate it yourself by going to scripts/data and running
```bash
python get_bbox.py
```
You might need to change the prompt and buffer path for [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) in config/data/get_bbox.yaml. For a drawer, you can use “red drawer”. For PickNplace and Pouring tasks, you can use “red mug”.

Once that is done, replace the model_path and model_ckpt at config/flow_generation/inference.yaml with the trained flow generation model path. Change the realworld_dataset_path to one of the tasks provided in the generation_model_eval_dataset, e.g., pickNplace. Change the directory to scripts/flow_generation and run

```bash
python inference.py
```

After finishing, the generated flow will be stored under the evaluation dataset folder. The numeric results are stored under the generated_flows folder for each episode. You can also find the gif for generated flows inside the dataset folder.

With the generated flow stored, you can evaluate the policy with the generated flow by navigating to scripts/control and running

```bash
python evaluate_from_flow_generation.py
```

You might need to modify the config/diffusion_policy/evaluate_from_flow_generation.yaml by replacing the `model_path` and `ckpt` with your trained flow condition policy. You also need to specify the evaluation dataset folder. Make sure you have already generated the flow for the dataset you passed in. You can find evaluation results under the experiment folder. The `generated_flow.gif` contains the processed generated flow animation.

```bash
.
├── episode_0
│   ├── action
│   ├── camera_0
│   ├── camera_0.mp4
│   ├── info
│   ├── proprioception
│   ├── qpos
│   └── qvel
├── episode_0_debug_pts.ply
├── episode_0_generated_flow.gif
├── episode_0_online_tracking.gif
├── episode_0_vis_pts.ply
```

<a name="Dataset-Details"></a>
## 📖 Dataset Details
All datasets are stored under the zarr format. The downloaded dataset already contains the processed flows. If you would like to process your own dataset, please refer to [real-world data](https://github.com/mengdaxu/im2Flow2Act/blob/main/scripts/data_script/realworld_data.md) and [simulation data](https://github.com/mengdaxu/im2Flow2Act/blob/main/scripts/data_script/simulation_data.md) for details. An episode from simulated data contains the following items:
```bash
.
├── action
├── camera_0
├── camera_0.mp4
├── info
├── moving_mask
├── point_tracking_sequence
├── proprioception
├── qpos
├── qvel
├── rgb_arr
├── robot_mask
├── sam_mask
├── sam_moving_mask
├── sample_indices
├── sam_point_tracking_sequence
└── task_description
```
- `point_tracking_sequence`: contains the flows by tracking uniform grid sampling keypoints using [TAPIR](https://github.com/google-deepmind/tapnet)
- `sam_point_tracking_sequence`: contains the object-centric flows iteratively generated by applying Segment Anything and point tracking.
- `moving_mask`: the binary mask over the `point_tracking_sequence`, which is created by setting whether a keypoint has moved a certain distance on the image.
- `robot_mask`: the binary mask over the "sam_point_tracking_sequence" to indicate whether a keypoint is located on the robot.
- `sam_moving_mask`: similar to `moving_mask` but over the `sam_point_tracking_sequence`
- `sam_mask`: the segment mask by running [Segment Anything](https://github.com/facebookresearch/segment-anything) on the initial scene
- `qpos` and `qvel`: the vectors used to restore the initial state for simulated data. You can also use them to re-render the data if you change the camera view.
- `rgb_arr`: contains the resized visual observations from `camera_0` with a downsampled factor of 2. It is passed to the point tracking algorithm. The main reason behind this is that a single 24GB GPU can run the point tracking algorithm on long-horizon play data.
- `task_description`: a text description for the episode.

Notice, during the training, we downsample the robot action and proprioception with a factor of 2 to align with the tracked flows. You can train on the original dataset by modifying the `dataset.downsample_rate=1` in the `config/train_flow_conditioned_diffusion_policy`. In this case, you also need to re-generate the flows for manipulation policy training data yourself using `scripts/data/data_pipeline` and modifying the `downsample_ratio=1`. You might need to use `scripts/clean` to remove the existing `rgb_arr` and corresponding flows first.

An episode in real-world dataset contains similar items but the point tracking sequence used to train flow generation model is stored at `bbox_point_tracking_sequence`.
- `bbox`: The bounding box of the intersted object, which is obtained by Grounding DINO.
- `bbox_point_tracking_sequence`: The flow generated by tracking keypoints inside the bounding box. 
To re-generate the dataset, you can use `scripts/data/flow_generation_data_pipeline`


### BibTeX
   ```bash
      @inproceedings{
      xu2024flow,
      title={Flow as the Cross-domain Manipulation Interface},
      author={Mengda Xu and Zhenjia Xu and Yinghao Xu and Cheng Chi and Gordon Wetzstein and Manuela Veloso and Shuran Song},
      booktitle={8th Annual Conference on Robot Learning},
      year={2024},
      url={https://openreview.net/forum?id=cNI0ZkK1yC}
      }
   ``` 
### License
This repository is released under the MIT license. 

### Acknowledgement
We would like to thank Yifan Hou, Zeyi Liu, Huy Ha, Mandi Zhao, Chuer Pan, Xiaomeng Xu, Yihuai Gao, Austin Patel, Haochen shi, John So, Yuwei Guo, Haoyu Xiong, Litian Liang, Dominik Bauer, Samir Yitzhak Gadre for their helpful feedback and fruitful discussions. 
#### Code
* We use [TAPIR](https://github.com/google-deepmind/tapnet) to track flow for both dataset generation and online point tracking. The flow generation code is based on [Animatediff](https://animatediff.github.io/). The tapnet code and the Animatediff code is directly copied from the official repo to make the code base more self-containing. 
* Diffusion Policy is adapted from [Diffusion Policy](https://github.com/real-stanford/diffusion_policy). 
* Simulation environment are adapted from [Scaling Up and Distilling Down](https://github.com/real-stanford/scalingup).
* Simulation Asset are obtained from [Mujoco Menagerie](https://github.com/google-deepmind/mujoco_menagerie) and [Kevin Zakka](https://kzakka.com/)'s [mujoco_scanned_objects](https://github.com/kevinzakka/mujoco_scanned_objects).