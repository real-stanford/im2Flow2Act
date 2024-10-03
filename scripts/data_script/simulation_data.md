# Simulation data preprocessing 

The downloaded dataset has already been processed. Use the following guide if you want to process the custom dataset. 

### Manipulation policy
To generate training flows for simulation dataset, run 
```bash
python data_pipeline.py
```
The configuration file is located at `config/data_pipeline.yaml`. Please note that you need to obtain the object bounding boxes for the drawer and folding tasks first, as the downloaded dataset already includes bounding boxes. For all four tasks, we use a `downsample_ratio=2` to downsample the raw videos, actions, and states. The n_sample_frame is set as follows for each task:
```bash
articulated: 60
deformable: 106
rigid: 105
```
The point_move_thresholds is set as following for each task:
```bash
articulated: 5
deformable: 2.5
rigid: 20
```
Set `simulation_heuristic_filter` to "box" for deformable and articulated tasks, and to `["left", "right"]` for rigid tasks. The `"box"` filter removes any points outside the object bounding box, while `"left"` and `"right"` are heuristic filters that remove background noise in the flows. Note that you can also use `"box"` for rigid tasks, but you will need to generate the bounding boxes for the dataset first. As there are multiple objects inside the rigid training datset, we simply use some herustic to filter out the object flows. Set `simulation_heuristic` to True to create a robot mask for the rigid dataset. Notice, some herustic filter depends on camera. You might need to modify them for custum dataset. 

### Flow Generation Model 

Please refer to the guide for real-world preprocessing. 
For all four tasks, we use a `downsample_ratio=2` to downsample the raw videos. The n_sample_frame is set as follows for each task.
```bash
drawer_open: 60
pickNplace: 72
pouring: 67
```

