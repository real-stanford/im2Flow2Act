### Visualization
You can visualize the flows by nevigating to `scripts/data`:
```bash
python viz_all.py
```
You may need to modify the dataset path in `viz_pathes`. To visualize the simulated play dataset, use `--viz_sam`. You can adjust the minimum distance a keypoint travels in the image space by specifying `viz_thresholds`; only keypoints that move more than the specified threshold will be visualized.

For visualizing real-world task demonstrations, use `--viz_bbox` and set `viz_thresholds` to 0. To visualize the simulation dataset, use `--viz_sam` and set `viz_thresholds` to -1, which utilizes the existing moving mask in the dataset. You can also set `viz_thresholds` to any value greater than 0 for visualization.

To display the trajectory, uncomment the `--draw_line` option.

Please note that when visualizing rigid objects, you may observe some noisy flows and robot flows, as the current visualization code does not filter them out. However, these are filtered out during policy training.