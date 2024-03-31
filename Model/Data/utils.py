"""
"""
import os
import logging

from PIL import Image
import numpy as np
from tqdm import tqdm


def load_images(folder_path) -> list:
    """ Load images from the specified folder.

    Args:
        folder_path: Path to folder containing images.

    Returns:
        images (list): List of images.
    """
    images = []
    for file in tqdm(os.listdir(folder_path), "Loading images"):
        img_path = os.path.join(folder_path, file)
        img = Image.open(img_path)
        img = np.array(img).flatten()
        images.extend(img)
    logging.info("Images loaded.")
    return images

def get_folder_name(folder_path) -> str:
    """ Get the name of the folder.

    Args:
        folder_path: Path to folder containing images.

    Returns:
        folder_name (str): Name of the folder.
    """
    return folder_path.split("/")[-1]
