import torch
import numpy as np


def collect_tensors_from_dataset(dataset, start_index, end_index):
    """
    Collects and stacks tensors from a PyTorch dataset within a specified range.

    Parameters:
    - dataset: A PyTorch dataset where each item is a dictionary.
    - start_index: The starting index of the range (inclusive).
    - end_index: The end index of the range (exclusive).

    Returns:
    - A dictionary where each key corresponds to stacked tensors for that key across the specified range.
    """
    # Initialize a dictionary to collect features.
    features_accumulator = {}

    # Check the first item to determine the keys and initialize lists for each key.
    if start_index < len(dataset):
        sample_item = dataset[start_index]
    else:
        raise IndexError("Start index is out of the dataset range.")

    for key in sample_item.keys():
        features_accumulator[key] = []

    # Collect features for each item in the specified range
    for i in range(start_index, end_index):
        if i >= len(dataset):
            break
        item = dataset[i]
        for key, value in item.items():
            # Convert numpy arrays to tensors if necessary
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            elif not isinstance(value, torch.Tensor):
                pass

            # Append the tensor to the list for the key
            features_accumulator[key].append(value)

    # Stack the lists of tensors for each key
    for key in features_accumulator.keys():
        if all(isinstance(x, torch.Tensor) for x in features_accumulator[key]):
            features_accumulator[key] = torch.stack(features_accumulator[key], dim=0)

    return features_accumulator
