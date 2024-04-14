import argparse
import logging
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from clearml import Dataset, Task


def clean_images(folder_path, remove_outliers=False):
    import logging
    import os
    import numpy as np
    from PIL import Image
    from tqdm import tqdm

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


# def preprocess_images(folder_path, resize_dimensions=(224, 224)):
#     """Preprocess images in the folder.

#     Args:
#         folder_path: Path to folder containing images.
#         resize_dimensions: Dimensions to resize images to.
#     """
#     logging.info("Preprocessing images…")

#     # Resize images
#     resized_images = []
#     for file in tqdm(os.listdir(folder_path), "Resizing images"):
#         img_path = os.path.join(folder_path, file)
#         img = Image.open(img_path)
#         if img.size != resize_dimensions:
#             img = img.resize(resize_dimensions)
#             img.save(img_path)
#             resized_images.append(img)

#     logging.info("Resized %d images.", len(resized_images))

#     # Normalize images
#     normalized_images = []
#     for file in tqdm(os.listdir(folder_path), "Normalizing images"):
#         img_path = os.path.join(folder_path, file)
#         img = Image.open(img_path)
#         img = np.array(img).flatten() / 255.0
#         normalized_images.append(img)

#     logging.info("Normalized %d images.", len(normalized_images))
#     return normalized_images


def preprocess_images(image_path, resize_dimensions=(224, 224)):
    import numpy as np
    from PIL import Image

    """
        Preprocess a single image.

        Args:
            image_path: Path to the image.
            resize_dimensions: Dimensions to resize images to.
    """
    img = Image.open(image_path)
    if img.size != resize_dimensions:
        img = img.resize(resize_dimensions)
        img.save(image_path)

    img = np.array(img).flatten() / 255.0
    return img


def upload_preprocessed_dataset(raw_dataset_id, project_name, queue_name, train_ratio):
    import argparse
    import logging
    import os
    import numpy as np
    from PIL import Image
    from tqdm import tqdm
    from clearml import Dataset, Task
    from pathlib import Path
    import random
    import shutil

    task = Task.init(
        project_name=project_name,
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

    # Create directories for train and test data
    train_dir = preprocessed_dir / "train"
    test_dir = preprocessed_dir / "test"
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)

    # Get categories from class folders in the dataset
    categories = os.listdir(raw_data_path)

    # Process images
    for category in categories:
        category_path = Path(raw_data_path) / category
        (train_dir / category).mkdir(exist_ok=True)
        (test_dir / category).mkdir(exist_ok=True)

        clean_images(category_path, False)

        # Split images into train and test sets
        image_files = list(category_path.iterdir())
        random.shuffle(image_files)
        train_files = image_files[: int(len(image_files) * train_ratio)]
        test_files = image_files[int(len(image_files) * train_ratio) :]

        # Save preprocessed images to train and test directories
        for image_file in train_files:
            image_data = preprocess_images(image_file)
            np.save(train_dir / category / (image_file.stem + ".npy"), image_data)
        for image_file in test_files:
            image_data = preprocess_images(image_file)
            np.save(test_dir / category / (image_file.stem + ".npy"), image_data)

    # Create a new dataset for the preprocessed training images
    processed_train_dataset = Dataset.create(
        dataset_name=raw_dataset.name + "_train",
        dataset_project=project_name,
        parent_datasets=[raw_dataset_id],
    )

    # Add the preprocessed training images to the dataset
    processed_train_dataset.add_files(str(train_dir), local_base_folder=str(train_dir))

    # Upload the training dataset to ClearML
    processed_train_dataset.upload()
    processed_train_dataset.finalize()

    # Create a new dataset for the preprocessed testing images
    processed_test_dataset = Dataset.create(
        dataset_name=raw_dataset.name + "_test",
        dataset_project=project_name,
        parent_datasets=[raw_dataset_id],
    )

    # Add the preprocessed testing images to the dataset
    processed_test_dataset.add_files(str(test_dir), local_base_folder=str(test_dir))

    # Upload the testing dataset to ClearML
    processed_test_dataset.upload()
    processed_test_dataset.finalize()

    # Cleanup
    for item in preprocessed_dir.rglob("*"):
        item.unlink()
    preprocessed_dir.rmdir()

    return processed_train_dataset.id, processed_test_dataset.id


if __name__ == "__main__":
    # Set logging level to INFO
    logging.basicConfig(level=logging.INFO)

    # Setup arg parse
    parser = argparse.ArgumentParser(description="Clean and preprocess data for model training.")
    parser.add_argument("--dataset_id", type=str, help="ID of the raw dataset")
    parser.add_argument("--clearml_project", type=str, help="Name of the project for the processed dataset")
    parser.add_argument("--queue_name", type=str, help="Name of the queue for remote execution")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Ratio of data to be used for training")
    args = parser.parse_args()

    # Upload preprocessed datasets
    processed_train_dataset_id, processed_test_dataset_id = upload_preprocessed_dataset(
        raw_dataset_id=args.dataset_id,
        project_name=args.clearml_project,
        queue_name=args.queue_name,
        train_ratio=args.train_ratio,
    )

    print(f"Preprocessed training dataset uploaded with ID: {processed_train_dataset_id}")
    print(f"Preprocessed testing dataset uploaded with ID: {processed_test_dataset_id}")
