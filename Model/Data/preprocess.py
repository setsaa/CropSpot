""" Preprocess data for the model.

This script is used to preprocess the dataset before training the model. It
normalizes the images and splits the dataset into training and validation sets.

Usage:
    python preprocess.py path

Arguments:
    path: Path to image folder to preprocess

Author:
    Sigurd Johnsen Setså (@setsaa)
"""
import logging
import os

from tqdm import tqdm

# from Model.Data.clean import validate_image_dimensions
# from Model.Data.utils import load_images


def resize_images(folder_path, dimensions) -> list:
    """ Check if images are not the same dimension and resize them.
    Return the resized images.

    Args:
        folder_path: Path to folder containing images.
        dimensions: Dimensions to resize images to.
    
    Returns:
        resized_images (list): List of resized images.
    """
    logging.info("Resizing images…")
    # Load images
    images = load_images(folder_path)

    if validate_image_dimensions(folder_path):
        return images

    # Add images to list
    resized_images = []
    for i, img in tqdm(images, "Resizing images"):
        if img.size != dimensions:
            img = img.resize(dimensions)
            # Remove original image
            os.remove(f"{folder_path}/{i}.jpg")
            # Add resized image
            resized_images.append(img)

    logging.info("Resized %d images.", len(images))
    return resized_images


def normalize_images(folder_path):
    """ Normalize the images in the folder and return the normalized images.

    Args:
        folder_path: Path to folder containing images.
    """
    logging.info("Normalizing images…")
    # Load images
    images = load_images(folder_path)

    # Normalize images
    normalized_images = []
    for img in tqdm(images, "Normalizing images"):
        normalized_images.append(img / 255.0)

    logging.info("Normalized %d images.", len(images))
    return normalized_images


def split_dataset(images, split_ratio=0.8):
    """ Split the dataset into training and validation sets.

    Args:
        images: List of images.
        split_ratio: Ratio to split the dataset.
    
    Returns:
        training_set (list): List of training images.
        validation_set (list): List of validation images.
    """
    logging.info("Splitting dataset…")
    split_index = int(len(images) * split_ratio)
    training_set = images[:split_index]
    validation_set = images[split_index:]

    logging.info("Dataset split.")
    return training_set, validation_set


if __name__ == "__main__":
    # Set logging level to INFO
    logging.basicConfig(level=logging.INFO)
