# Real-World data preprocessing 
The downloaded dataset has already been processed. Use the following guide if you want to process the custom dataset.

To generate training flows for real-world dataset, run 
```bash
python flow_generation_data_pipeline.py
```
The config file is located at `config/flow_generation_data_pipeline.yaml`. Please note that you need to obtain the object bounding boxes first; the downloaded dataset already includes all bounding boxes. For all four tasks, we use a `downsample_ratio=2` to downsample the raw videos and run the point tracing algorithm on an L40S GPU with 46GB of memory. The n_sample_frame is set as follows for each task.
```bash
cloth: 110
drawer_open: 85
pickNplace: 105
pouring: 150
```
You can adjust the `downsample_ratio` and `n_sample_frame` settings based on the available GPU memory. The training of the flow generation model is not particularly sensitive to these parameters, as it uniformly samples 32 frames from the tracked flows. However, ensure that `n_sample_frame` remains greater than the number of frames after downsampling.

