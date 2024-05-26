def preprocess_dataset(dataset_name, project_name):
    """
    Preprocess images in the raw dataset and upload the preprocessed images to ClearML.

    Args:
        dataset_name: Name of the raw dataset.
        project_name: Name of the project for the processed dataset.

    Returns:
        ID and name of the processed dataset.
    """
    from clearml import Dataset, Task
    import os
    import logging
    import numpy as np
    import shutil
    from PIL import Image
    from pathlib import Path

    task = Task.init(project_name=project_name, task_name="Preprocess Uploaded Data")

    # TEMP CHANGE
    prep_dataset = Dataset.get(dataset_name=dataset_name + "_preprocessed")
    if prep_dataset:
        print(f"Preprocessed dataset '{dataset_name}_preprocessed' already exists in project '{project_name}'.")
        return prep_dataset.id, prep_dataset.name

    # Access the raw dataset
    raw_dataset = Dataset.get(dataset_name=dataset_name)

    # Check if the preprocessed dataset is already downloaded. If not, download it and preprocess the images.
    preprocessed_dir = Path(f"Dataset/{dataset_name}_preprocessed")
    if preprocessed_dir.exists():
        print("Dataset already exists")

        # Remove the old preprocessed directory
        print("Removing the old preprocessed directory...")
        shutil.rmtree(preprocessed_dir)
        preprocessed_dir.mkdir()

    print("Downloading the dataset...")
    raw_dataset.get_mutable_local_copy(str(preprocessed_dir))

    # New preprocessed directory
    print("Processing images...")
    for category in os.listdir(preprocessed_dir):
        category_path = Path(preprocessed_dir) / category

        # Remove non-jpg files
        count = 0
        for file in os.listdir(category_path):
            if not file.lower().endswith(".jpg"):
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
