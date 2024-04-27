import os
import logging
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
from clearml import Dataset, Task


def preprocess_images(dataset_dir, preprocessed_dir, remove_outliers=False):
    import os
    import logging
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
    for file in tqdm(os.listdir(dataset_dir), desc="Removing non-jpg files"):
        if not file.endswith(".jpg"):
            os.remove(os.path.join(dataset_dir, file))
            count += 1
    print(f"Removed {count} non-jpg files.")

    # Remove corrupt images
    removed_files = []
    for file in tqdm(os.listdir(dataset_dir), desc="Check for corrupt images"):
        img_path = os.path.join(dataset_dir, file)
        try:
            img = Image.open(img_path)
            np.array(img).flatten()
        except (IOError, SyntaxError):
            removed_files.append(file)
            os.remove(img_path)
            logging.info("Removed corrupt image: %s", file)

    # Validate image dimensions
    dimensions = []
    for file in tqdm(os.listdir(dataset_dir), desc="Validating image dimensions"):
        img = Image.open(os.path.join(dataset_dir, file))
        dimensions.append(img.size)

    if len(set(dimensions)) > 1:
        print("Images have different dimensions.")
    else:
        print("All images have the dimension " + str(dimensions[0]))

    # Detect and remove outliers
    if remove_outliers:
        logging.info("Detecting outliers…")
        # Load images
        images = [np.array(Image.open(os.path.join(dataset_dir, file)).flatten()) for file in os.listdir(dataset_dir)]

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

    # Create a subdirectory in the preprocessed directory for this category
    category = os.path.basename(dataset_dir)
    preprocessed_category_dir = preprocessed_dir / category
    preprocessed_category_dir.mkdir(exist_ok=True)

    # Resize images and save in the preprocessed directory
    for file in tqdm(os.listdir(dataset_dir), desc="Resizing images"):
        img = Image.open(os.path.join(dataset_dir, file))
        img = img.resize((224, 224))
        img.save(os.path.join(preprocessed_category_dir, file))

    new_count = len(os.listdir(preprocessed_category_dir))
    print(f"Images in {category}: {new_count} (from {len(os.listdir(dataset_dir))})")


def preprocess_dataset(raw_dataset_name, project_name, queue_name):
    """
    Preprocess images in the raw dataset and upload the preprocessed images to ClearML.

    Args:
        raw_dataset_name: Name of the raw dataset.
        project_name: Name of the project for the processed dataset.
        queue_name: Name of the queue for remote execution.

    Returns:
        ID and name of the processed dataset.
    """
    import os
    from pathlib import Path
    from clearml import Dataset, Task

    task = Task.init(
        project_name=project_name,
        task_name="Dataset Preprocessing",
        task_type=Task.TaskTypes.data_processing,
    )
    # task.execute_remotely(queue_name=queue_name, exit_process=True)

    # Access the raw dataset
    raw_dataset = Dataset.get(dataset_name=raw_dataset_name)
    raw_data_path = raw_dataset.get_local_copy()

    # Create directories for preprocessed data
    preprocessed_dir = Path("./preprocessed")
    preprocessed_dir.mkdir(exist_ok=True)

    # Process images
    for category in os.listdir(raw_data_path):
        category_path = Path(raw_data_path) / category
        preprocess_images(category_path, preprocessed_dir)

    # Create a new dataset for the preprocessed images
    processed_dataset = Dataset.create(
        dataset_name=raw_dataset.name + "_preprocessed",
        dataset_project=project_name,
        parent_datasets=[raw_dataset],
    )

    # Add the preprocessed images to the dataset
    processed_dataset.add_files(str(preprocessed_dir), local_base_folder=str(preprocessed_dir))

    # Upload the training dataset to ClearML
    processed_dataset.upload()
    processed_dataset.finalize()

    # Clean up
    for item in preprocessed_dir.rglob("*"):
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            for subitem in item.rglob("*"):
                subitem.unlink()
            item.rmdir()
    preprocessed_dir.rmdir()

    return processed_dataset.id, processed_dataset.name


if __name__ == "__main__":
    import argparse
    import logging

    # Set logging level to INFO
    logging.basicConfig(level=logging.INFO)

    # Setup arg parse
    parser = argparse.ArgumentParser(description="Clean and preprocess data for model training.")
    parser.add_argument("--dataset_name", type=str, default="TomatoDiseaseDataset", help="Name of the raw dataset")
    parser.add_argument("--project_name", type=str, default="CropSpot", help="Name of the project for the processed dataset")
    parser.add_argument("--queue_name", type=str, default="helldiver", help="Name of the queue for remote execution")

    # Parse command-line arguments
    args = parser.parse_args()

    # Upload preprocessed datasets
    processed_dataset_id, processed_dataset_name = preprocess_dataset(
        raw_dataset_name=args.dataset_name,
        project_name=args.project_name,
        queue_name=args.queue_name,
    )

    print(f"Preprocessed dataset uploaded with ID: {processed_dataset_id}")
