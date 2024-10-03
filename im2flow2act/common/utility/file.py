import json
import pickle

import cv2
import numpy as np


def save_json(data, filename):
    """
    Save a Python dictionary to a JSON file.

    :param data: The data to be saved (usually a dictionary).
    :param filename: The name of the file where the data will be saved.
    """
    with open(filename, "w") as file:
        json.dump(data, file)


def load_json(filename):
    """
    Load JSON data from a file and return it as a Python dictionary.

    :param filename: The name of the file to load the data from.
    :return: The data loaded from the JSON file.
    """
    with open(filename, "r") as file:
        return json.load(file)


def read_pickle(file_name):
    """
    Read data from a pickle file.

    :param file_name: Name of the file to read from.
    :return: The data unpickled from the file.
    """
    with open(file_name, "rb") as file:
        return pickle.load(file)


def write_pickle(data, file_name):
    """
    Write data to a pickle file.

    :param data: The data to be pickled.
    :param file_name: Name of the file where data will be stored.
    """
    with open(file_name, "wb") as file:
        pickle.dump(data, file)


# Function to read a video file and extract frames at a specific interval
def extract_frames(video_path, frame_interval=1):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None

    frames = []
    frame_count = 0

    while True:
        # Read a frame
        ret, frame = cap.read()

        # If no frame is read, we have reached the end of the video
        if not ret:
            break

        # Check if the current frame is at the specified interval
        if frame_count % frame_interval == 0:
            # Convert the frame from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert the frame to a NumPy array
            frame_array = np.array(frame_rgb)
            frames.append(frame_array)

        frame_count += 1

    # Release the video capture object
    cap.release()

    return frames


def change_white_balance(image, red_scale=1.0, blue_scale=1.0):
    # Convert the image from BGR to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Convert to float32 for precise manipulation
    lab_image = np.float32(lab_image)

    # Scale the A (green-red) and B (blue-yellow) channels
    lab_image[:, :, 1] = lab_image[:, :, 1] * red_scale
    lab_image[:, :, 2] = lab_image[:, :, 2] * blue_scale

    # Ensure the values stay within valid range
    lab_image[:, :, 1] = np.clip(lab_image[:, :, 1], 0, 255)
    lab_image[:, :, 2] = np.clip(lab_image[:, :, 2], 0, 255)

    # Convert back to uint8
    lab_image = np.uint8(lab_image)

    # Convert the image back from LAB to BGR
    result_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

    return result_image


def change_saturation_brightness(image, saturation_scale=1.0, brightness_scale=1.0):
    # Convert the image from BGR to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Convert to float32 for precise manipulation
    hsv_image = np.float32(hsv_image)

    # Scale the saturation and brightness (value) channels
    hsv_image[:, :, 1] = hsv_image[:, :, 1] * saturation_scale
    hsv_image[:, :, 2] = hsv_image[:, :, 2] * brightness_scale

    # Ensure the values stay within valid range
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1], 0, 255)
    hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2], 0, 255)

    # Convert back to uint8
    hsv_image = np.uint8(hsv_image)

    # Convert the image back from HSV to BGR
    result_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return result_image
