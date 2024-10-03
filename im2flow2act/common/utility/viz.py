import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch


def save_to_gif(frames, output_path, duration=5):
    imageio.mimsave(output_path, frames, duration=duration)


def convert_tensor_to_image(tensor):
    """
    Reverses the normalization on a tensor.

    Args:
    tensor (Tensor): The normalized tensor.
    mean (list): The mean used for normalization (for each channel).
    std (list): The standard deviation used for normalization (for each channel).

    Returns:
    Tensor: The unnormalized tensor.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # The mean and std have to be broadcastable to the tensor's shape
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)

    # We need to reshape mean and std to [C, 1, 1] to broadcast them correctly
    mean = mean[:, None, None]
    std = std[:, None, None]

    tensor = tensor * std + mean
    tensor = (tensor * 255).cpu().numpy()
    image = np.transpose(tensor, [1, 2, 0]).astype(np.uint8)
    image = np.ascontiguousarray(image)
    return image


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )
