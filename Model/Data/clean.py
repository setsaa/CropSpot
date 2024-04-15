""" Clean data for model training.

This script is used to clean the dataset before training the model. It removes
non-jpg files, corrupt images, and detects outliers.

Usage:
    python clean.py path [--no-outliers]

Arguments:
    path: Path to image folder to clean
    --no-outliers: Avoid detecting outliers

Author:
    Sigurd Johnsen Setså (@setsaa)
"""
import argparse

import logging
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

# import Model from "" 

from Model.Data.utils import load_images


def detect_outliers(folder_path):
    """ Use statistical methods to detect outlier images in the folder.
    
    Args:
        folder_path: Path to folder containing images.

    Example path: 'Dataset/Tomato/healthy'
    """
    logging.info("Detecting outliers…")
    # Load images
    images = load_images(folder_path)

    # Calculate mean and standard deviation
    mean = 0
    for img in tqdm(images, "Calculating mean"):
        mean += img
    mean /= len(images)

    std = 0
    for img in tqdm(images, "Calculating standard deviation"):
        std += (img - mean) ** 2
    std = np.sqrt(std / len(images))

    # Detect outliers
    outliers = []
    for i, img in tqdm(enumerate(images)):
        if abs(img - mean) > 2 * std:
            outliers.append(i)

    # Print results
    if len(outliers) == 0:
        print('No outliers detected.')
    else:
        print(f'Outliers detected: {outliers}')


def remove_non_jpg(folder_path):
    """ Remove non-jpg files from folder.

    Args:
        folder_path: Path to folder containing images.
    """
    logging.info("Removing non-jpg images…")
    count = 0
    for file in tqdm(os.listdir(folder_path), "Removing non-jpg files"):
        if not file.endswith('.jpg'):
            os.remove(f'{folder_path}/{file}')
            count += 1
    print(f'Removed {count} non-jpg files.')


def validate_image_dimensions(folder_path) -> bool:
    """ Check the dimensions of the images in the folder.

    Args:
        folder_path: Path to folder containing images.
    """
    logging.info("Checking image dimensions…")
    dimensions = []
    for file in tqdm(os.listdir(folder_path), "Validating image dimensions"):
        img = Image.open(f'{folder_path}/{file}')
        dimensions.append(img.size)

    if len(set(dimensions)) > 1:
        print('Images have different dimensions.')
        return False
    
    print('All images have the dimension ' + str(dimensions[0]))
    return True


def remove_corrupt_images(folder_path):
    """ Remove potentially corrupt images from the specified folder.

    Args:
        folder_path (str): Path to folder containing images.

    Returns:
        removed_files (list): List of filenames of removed images.
    """
    logging.info("Removing corrupt images…")
    removed_files = []
    for file in tqdm(os.listdir(folder_path), "Check for corrupt images"):
        img_path = os.path.join(folder_path, file)
        try:
            img = Image.open(img_path)
            np.array(img).flatten() # Try to convert to numpy array
        except (IOError, SyntaxError):
            removed_files.append(file)
            os.remove(img_path)
            logging.info("Removed corrupt image: %s", file)
    return removed_files


def get_len_dataset(folder_path) -> int:
    """Get the number of images in the dataset. """
    len_dataset = len(os.listdir(folder_path))
    return len_dataset


if __name__ == '__main__':
    # Set logging level to INFO
    logging.basicConfig(level=logging.INFO)

    # Setup arg parse
    parser = argparse.ArgumentParser(description='Clean data for model training.')
    parser.add_argument('path', type=str, help='Path to image folder to clean')
    parser.add_argument('--no-outliers', action='store_true', help='Avoid detecting outliers')
    args = parser.parse_args()

    # Get number of images in dataset before cleaning
    num_images = get_len_dataset(args.path)
    logging.info('Found %d images in the folder. Cleaning…', num_images)

    # Run cleaning functions
    remove_non_jpg(args.path)
    remove_corrupt_images(args.path)
    validate_image_dimensions(args.path)
    if not args.no_outliers:
        detect_outliers(args.path)

    # Get number of images in dataset after cleaning
    remaining_images = get_len_dataset(args.path)
    removed_images = num_images - remaining_images
    logging.info('Removed %d images. %d images remaining.', removed_images, remaining_images)
