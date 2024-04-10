import argparse
import logging
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from clearml import Dataset, Task


def clean_images(folder_path, remove_outliers=False):
    """Clean images in the folder.

    Args:
        folder_path: Path to folder containing images.
        remove_outliers: Whether to remove outlier images.
    """
    logging.info("Cleaning images…")

    # Remove non-jpg files
    count = 0
    for file in tqdm(os.listdir(folder_path), "Removing non-jpg files"):
        if not file.endswith(".jpg"):
            os.remove(f"{folder_path}/{file}")
            count += 1
    print(f"Removed {count} non-jpg files.")

    # Remove corrupt images
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

    # Validate image dimensions
    dimensions = []
    for file in tqdm(os.listdir(folder_path), "Validating image dimensions"):
        img = Image.open(f"{folder_path}/{file}")
        dimensions.append(img.size)

    if len(set(dimensions)) > 1:
        print("Images have different dimensions.")
    else:
        print("All images have the dimension " + str(dimensions[0]))

    # Detect and remove outliers
    if remove_outliers:
        logging.info("Detecting outliers…")
        # Load images
        images = [np.array(Image.open(os.path.join(folder_path, file)).flatten()) for file in os.listdir(folder_path)]

        # Calculate mean and standard deviation
        mean = np.mean(images)
        std = np.std(images)

        # Detect outliers
        outliers = [i for i, img in enumerate(images) if abs(img - mean) > 2 * std]

        # Print results
        if len(outliers) == 0:
            print("No outliers detected.")
        else:
            print(f"Outliers detected: {outliers}")


def preprocess_images(folder_path, resize_dimensions=(224, 224)):
    """Preprocess images in the folder.

    Args:
        folder_path: Path to folder containing images.
        resize_dimensions: Dimensions to resize images to.
    """
    logging.info("Preprocessing images…")

    # Resize images
    resized_images = []
    for file in tqdm(os.listdir(folder_path), "Resizing images"):
        img_path = os.path.join(folder_path, file)
        img = Image.open(img_path)
        if img.size != resize_dimensions:
            img = img.resize(resize_dimensions)
            img.save(img_path)
            resized_images.append(img)

    logging.info("Resized %d images.", len(resized_images))

    # Normalize images
    normalized_images = []
    for file in tqdm(os.listdir(folder_path), "Normalizing images"):
        img_path = os.path.join(folder_path, file)
        img = Image.open(img_path)
        img = np.array(img).flatten() / 255.0
        normalized_images.append(img)

    logging.info("Normalized %d images.", len(normalized_images))


def upload_preprocessed_dataset(raw_dataset_id, processed_dataset_project, processed_dataset_name, queue_name):
    import argparse
    import os
    from pathlib import Path
    import numpy as np
    from PIL import Image
    from clearml import Dataset, Task

    task = Task.init(
        project_name=processed_dataset_project,
        task_name="Dataset Preprocessing",
        task_type=Task.TaskTypes.data_processing,
    )
    task.execute_remotely(queue_name=queue_name, exit_process=True)

    # Access the raw dataset
    raw_dataset = Dataset.get(dataset_id=raw_dataset_id)
    raw_data_path = raw_dataset.get_local_copy()

    # Create directories for preprocessed data
    preprocessed_dir = Path("./preprocessed")
    preprocessed_dir.mkdir(exist_ok=True)

    # Get categories from class folders in the dataset
    categories = os.listdir(raw_data_path)

    # Process images
    for category in categories:
        category_path = Path(raw_data_path) / category
        (preprocessed_dir / category).mkdir(exist_ok=True)

        clean_images(category_path, False)
        preprocess_images(category_path)

        # for image_file in category_path.iterdir():
        #     image_data = preprocess_image(image_file)
        #     np.save(preprocessed_dir / category / (image_file.stem + ".npy"), image_data)

    # Create a new dataset for the preprocessed images
    processed_dataset = Dataset.create(
        dataset_name=processed_dataset_name,
        dataset_project=processed_dataset_project,
        parent_datasets=[raw_dataset_id],
    )

    # Add the preprocessed images to the dataset
    processed_dataset.add_files(str(preprocessed_dir), local_base_folder=str(preprocessed_dir))

    # Upload the dataset to ClearML
    processed_dataset.upload()
    processed_dataset.finalize()

    # Cleanup
    for item in preprocessed_dir.rglob("*"):
        item.unlink()
    preprocessed_dir.rmdir()

    return processed_dataset.id


if __name__ == "__main__":
    # Set logging level to INFO
    logging.basicConfig(level=logging.INFO)

    # Setup arg parse
    parser = argparse.ArgumentParser(description="Clean and preprocess data for model training.")
    parser.add_argument("dataset_id", type=str, help="ID of the raw dataset")
    parser.add_argument("processed_dataset_project", type=str, help="Name of the project for the processed dataset")
    parser.add_argument("processed_dataset_name", type=str, help="Name of the processed dataset")
    parser.add_argument("queue_name", type=str, help="Name of the queue for remote execution")
    parser.add_argument("--no-outliers", action="store_true", help="Avoid detecting outliers")
    parser.add_argument("--resize-dimensions", type=tuple, default=(256, 256), help="Dimensions to resize images to")
    args = parser.parse_args()

    # Access the raw dataset
    raw_dataset = Dataset.get(dataset_id=args.dataset_id)
    raw_data_path = raw_dataset.get_local_copy()

    # Clean images
    clean_images(raw_data_path, remove_outliers=not args.no_outliers)

    # Preprocess images
    preprocess_images(raw_data_path, resize_dimensions=args.resize_dimensions)

    # Upload preprocessed dataset
    processed_dataset_id = upload_preprocessed_dataset(
        raw_dataset_id=args.dataset_id,
        processed_dataset_project=args.processed_dataset_project,
        processed_dataset_name=args.processed_dataset_name,
        queue_name=args.queue_name,
    )

    print(f"Preprocessed dataset uploaded with ID: {processed_dataset_id}")
