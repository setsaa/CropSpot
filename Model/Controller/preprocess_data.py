import argparse
import logging
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from Model.Data.utils import load_images
from Model.Data.clean import validate_image_dimensions


def detect_outliers(folder_path):
    """Use statistical methods to detect outlier images in the folder.

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
        print("No outliers detected.")
    else:
        print(f"Outliers detected: {outliers}")


def remove_non_jpg(folder_path):
    """Remove non-jpg files from folder.

    Args:
        folder_path: Path to folder containing images.
    """
    logging.info("Removing non-jpg images…")
    count = 0
    for file in tqdm(os.listdir(folder_path), "Removing non-jpg files"):
        if not file.endswith(".jpg"):
            os.remove(f"{folder_path}/{file}")
            count += 1
    print(f"Removed {count} non-jpg files.")


def validate_image_dimensions(folder_path) -> bool:
    """Check the dimensions of the images in the folder.

    Args:
        folder_path: Path to folder containing images.
    """

    logging.info("Checking image dimensions…")
    dimensions = []
    for file in tqdm(os.listdir(folder_path), "Validating image dimensions"):
        img = Image.open(f"{folder_path}/{file}")
        dimensions.append(img.size)

    if len(set(dimensions)) > 1:
        print("Images have different dimensions.")
        return False

    print("All images have the dimension " + str(dimensions[0]))
    return True


def remove_corrupt_images(folder_path):
    """Remove potentially corrupt images from the specified folder.

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
            np.array(img).flatten()  # Try to convert to numpy array
        except (IOError, SyntaxError):
            removed_files.append(file)
            os.remove(img_path)
            logging.info("Removed corrupt image: %s", file)
    return removed_files


def get_len_dataset(folder_path) -> int:
    """Get the number of images in the dataset."""
    len_dataset = len(os.listdir(folder_path))
    return len_dataset


if __name__ == "__main__":
    # Set logging level to INFO
    logging.basicConfig(level=logging.INFO)

    # Setup arg parse
    parser = argparse.ArgumentParser(description="Clean data for model training.")
    parser.add_argument("path", type=str, help="Path to image folder to clean")
    parser.add_argument("--no-outliers", action="store_true", help="Avoid detecting outliers")
    args = parser.parse_args()

    # Get number of images in dataset before cleaning
    num_images = get_len_dataset(args.path)
    logging.info("Found %d images in the folder. Cleaning…", num_images)

    # Run cleaning functions
    remove_non_jpg(args.path)
    remove_corrupt_images(args.path)
    validate_image_dimensions(args.path)
    if not args.no_outliers:
        detect_outliers(args.path)

    # Get number of images in dataset after cleaning
    remaining_images = get_len_dataset(args.path)
    removed_images = num_images - remaining_images
    logging.info("Removed %d images. %d images remaining.", removed_images, remaining_images)


def resize_images(folder_path, dimensions) -> list:
    """Check if images are not the same dimension and resize them.
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
    """Normalize the images in the folder and return the normalized images.

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
    """Split the dataset into training and validation sets.

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


def load_images(folder_path) -> list:
    """Load images from the specified folder.

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
    """Get the name of the folder.

    Args:
        folder_path: Path to folder containing images.

    Returns:
        folder_name (str): Name of the folder.
    """
    return folder_path.split("/")[-1]


if __name__ == "__main__":
    # Set logging level to INFO
    logging.basicConfig(level=logging.INFO)

    # Setup arg parse
    parser = argparse.ArgumentParser(description="Clean and preprocess data for model training.")
    parser.add_argument("path", type=str, help="Path to image folder to clean")
    parser.add_argument("--no-outliers", action="store_true", help="Avoid detecting outliers")
    parser.add_argument("--resize-dimensions", type=tuple, default=(256, 256), help="Dimensions to resize images to")
    parser.add_argument("--split-ratio", type=float, default=0.8, help="Ratio to split the dataset")
    args = parser.parse_args()

    # Get number of images in dataset before cleaning
    num_images = get_len_dataset(args.path)
    logging.info("Found %d images in the folder. Cleaning…", num_images)

    # Run cleaning functions
    remove_non_jpg(args.path)
    remove_corrupt_images(args.path)
    validate_image_dimensions(args.path)
    if not args.no_outliers:
        detect_outliers(args.path)

    # Get number of images in dataset after cleaning
    remaining_images = get_len_dataset(args.path)
    removed_images = num_images - remaining_images
    logging.info("Removed %d images. %d images remaining.", removed_images, remaining_images)

    # Preprocessing
    logging.info("Preprocessing images…")
    resize_images(args.path, args.resize_dimensions)
    normalize_images(args.path)

    # Load images
    images = load_images(args.path)

    # Split dataset
    training_set, validation_set = split_dataset(images, args.split_ratio)
    logging.info("Split dataset into %d training images and %d validation images.", len(training_set), len(validation_set))
