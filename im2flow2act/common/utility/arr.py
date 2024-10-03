import numpy as np


def uniform_sampling(array, num_samples, return_indices=False):
    arr_len = len(array)
    sample_indices = np.linspace(0, arr_len - 1, num_samples).astype(int)
    if return_indices:
        return array[sample_indices], sample_indices
    return array[sample_indices]


def stratified_random_sampling(
    array, num_samples, randomness=0.4, return_indices=False
):
    arr_len = len(array)
    # Calculate step size for uniform distribution
    step = (arr_len - 1) / (num_samples - 1)
    # Generate uniform indices
    uniform_indices = np.linspace(0, arr_len - 1, num_samples)
    # Calculate maximum shift based on the step size and specified randomness factor
    max_shift = step * randomness
    # Generate random shifts for each index, ensuring the first and last indices remain unchanged
    random_shifts = np.random.uniform(-max_shift, max_shift, num_samples)
    random_shifts[0] = random_shifts[-1] = 0  # Keep the first and last indices anchored
    # Apply shifts and ensure indices are within valid bounds
    sample_indices = np.clip(uniform_indices + random_shifts, 0, arr_len - 1).astype(
        int
    )
    sample_indices = np.sort(sample_indices)
    if return_indices:
        return array[sample_indices], sample_indices
    return array[sample_indices]


def random_sampling(
    array, num_samples, zero_include=True, return_indices=False, replace=True
):
    arr_len = len(array)
    if zero_include:
        sample_indices = np.sort(
            np.random.choice(arr_len, size=num_samples - 1, replace=replace)
        )
        sample_indices = np.insert(sample_indices, 0, 0)
    else:
        sample_indices = np.sort(
            np.random.choice(arr_len, size=num_samples, replace=replace)
        )
    if return_indices:
        return array[sample_indices], sample_indices
    return array[sample_indices]


def complete_random_sampling(array, num_samples, return_indices=False):
    arr_len = len(array)
    if num_samples <= arr_len:
        sample_indices = np.sort(
            np.random.choice(arr_len, size=num_samples, replace=False)
        )
    else:
        sample_indices = np.arange(arr_len)
        slack = num_samples - arr_len
        slack_indices = np.random.choice(arr_len, size=slack, replace=True)
        sample_indices = np.concatenate([sample_indices, slack_indices])
        sample_indices = np.sort(sample_indices)

    if return_indices:
        return array[sample_indices], sample_indices
    return array[sample_indices]


def padding(array, padding_start, padding_size):
    padding = np.tile(array[-1:], (padding_size,) + (1,) * (array.ndim - 1))
    return np.concatenate([array[padding_start:], padding], axis=0)
