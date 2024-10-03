import os
import shutil

# Path to the main folder containing episode_x folders
main_folder_path = ""
items_to_delete = [
    "point_tracking_sequence",
    "sam_point_tracking_sequence",
    "bbox_point_tracking_sequence",
    "robot_mask",
    "sam_mask",
    "sam_moving_mask",
    "moving_mask",
    "rgb_arr",
    "sample_indices",
    "task_description",
    "bbox",
]
# Iterate through each item in the main folder
for episode_folder in os.listdir(main_folder_path):
    episode_folder_path = os.path.join(main_folder_path, episode_folder)
    # Check if the current item is a directory and follows the episode_x naming pattern
    if os.path.isdir(episode_folder_path) and episode_folder.startswith("episode_"):
        for item in items_to_delete:
            item_path = os.path.join(episode_folder_path, item)
            # Check if the item_path exists
            if os.path.exists(item_path):
                if os.path.isdir(item_path):
                    # Use shutil.rmtree() to remove directories
                    shutil.rmtree(item_path)
                    print(f"Deleted directory: {item_path}")
                elif os.path.isfile(item_path):
                    # Use os.remove() to remove files
                    os.remove(item_path)
                    print(f"Deleted file: {item_path}")
                print("Detect")
