import numpy as np
import torch


def process_image(image):
    """ Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    """
    # Resize to 256x256
    image.thumbnail((256, 256))

    # Centre crop to 224x224
    width, height = image.size
    new_width, new_height = 224, 224

    left = width // 2 - new_width // 2
    upper = height // 2 - new_height // 2
    right = width // 2 + new_width // 2
    lower = height // 2 + new_height // 2

    image = image.crop((left, upper, right, lower))

    # Convert to float
    np_image = np.array(image)
    np_image = np_image / 255

    # Normalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np.transpose(np_image, axes=[2, 0, 1])

    return torch.from_numpy(np_image)
