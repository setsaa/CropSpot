def preprocess_dataset(dataset_name, project_name, queue_name):
    """
    Preprocess images in the raw dataset and upload the preprocessed images to ClearML.

    Args:
        dataset_name: Name of the raw dataset.
        project_name: Name of the project for the processed dataset.
        queue_name: Name of the queue for remote execution.

    Returns:
        ID and name of the processed dataset.
    """
    import os
    import logging
    import numpy as np
    from PIL import Image
    from pathlib import Path
    from clearml import Dataset, Task

    task = Task.init(project_name=project_name, task_name="Dataset Preprocessing", task_type=Task.TaskTypes.data_processing)

    # Access the raw dataset
    raw_dataset = Dataset.get(dataset_name=dataset_name)

    # Check if the raw dataset is already downloaded. If not, download it and preprocess the images.
    dataset_path = "Dataset/Raw Data"
    if not os.path.exists(dataset_path):
        print("Downloading the dataset...")

        # Create directories for preprocessed data
        preprocessed_dir = Path("Dataset/Preprocessed")
        preprocessed_dir.mkdir(exist_ok=True)

        # Clean up the preprocessed directory
        if preprocessed_dir.exists():
            for item in preprocessed_dir.iterdir():
                if item.is_dir():
                    for subitem in item.iterdir():
                        subitem.unlink()
                    item.rmdir()
            preprocessed_dir.rmdir()

        # Download the dataset
        raw_dataset.get_mutable_local_copy(preprocessed_dir)

        # Process images
        for category in os.listdir(preprocessed_dir):
            category_path = Path(preprocessed_dir) / category

            # Remove non-jpg files
            count = 0
            for file in os.listdir(category_path):
                if not file.endswith(".jpg"):
                    os.remove(os.path.join(category_path, file))
                    count += 1
            print(f"Removed {count} non-jpg files.")

            # Remove corrupt images
            removed_files = []
            for file in os.listdir(category_path):
                img_path = os.path.join(category_path, file)
                try:
                    img = Image.open(img_path)
                    np.array(img).flatten()
                except (IOError, SyntaxError) as e:
                    removed_files.append(file)
                    os.remove(img_path)
                    logging.info(f"Removed corrupt image: {file} due to {e}")

            # Validate image dimensions
            dimensions = []
            for file in os.listdir(category_path):
                img = Image.open(os.path.join(category_path, file))
                dimensions.append(img.size)

            if len(set(dimensions)) > 1:
                print("Images have different dimensions.")
            else:
                print("All images have the dimension " + str(dimensions[0]))
    else:
        print("Preprocessed Dataset already prepared")
        preprocessed_dir = "Dataset/Preprocessed"

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

    # Finalize the dataset
    processed_dataset.finalize()

    return processed_dataset.id, processed_dataset.name


if __name__ == "__main__":
    import logging
    import argparse

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
        dataset_name=args.dataset_name,
        project_name=args.project_name,
        queue_name=args.queue_name,
    )

    print(f"Preprocessed dataset uploaded with ID: {processed_dataset_id}")
