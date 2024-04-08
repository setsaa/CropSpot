import os
import argparse

# import logging
from clearml import Task, Dataset

# # Set up logging
# logging.basicConfig(level=logging.INFO)


def upload_local_dataset(dataset_dir, clearml_project_name, dataset_name):
    """
    Upload a local dataset to a ClearML project.

    Parameters:
    dataset_dir (str): Path to the local dataset directory.
    clearml_project_name (str): Name of the ClearML project.
    dataset_name (str): Name of the dataset in ClearML.

    Returns:
    str: ID of the uploaded dataset.
    """

    # Create a ClearML task
    task = Task.init(project_name=clearml_project_name, task_name="Dataset Upload", task_type=Task.TaskTypes.data_processing)

    # Ensure the dataset directory is valid
    if not os.path.isdir(dataset_dir):
        raise ValueError(f"The specified path '{dataset_dir}' is not a directory or does not exist.")

    # Ensure the dataset directory is not empty
    if not os.listdir(dataset_dir):
        raise ValueError(f"The specified path '{dataset_dir}' is empty.")

    # Create a ClearML dataset
    dataset = Dataset.create(dataset_name=dataset_name, dataset_project=clearml_project_name)

    # Add the dataset directory to the dataset
    dataset.add_files(dataset_dir)

    # Upload the dataset to ClearML
    dataset.upload()

    # Finalize the dataset
    dataset.finalize()

    print(f"Dataset uploaded with ID: {dataset.id}")
    # logging.info(f"Dataset uploaded with ID: {dataset.id}")

    return dataset.id


if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Upload Dataset Directory to ClearML")

    # Add arguments
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--clearml_project_name", type=str, required=True, help="ClearML project name")
    parser.add_argument("--dataset_name", type=str, required=True, help="ClearML dataset name")

    # Parse command-line arguments
    args = parser.parse_args()

    # Call function with the parsed arguments
    upload_local_dataset(args.dataset_dir, args.clearml_project_name, args.dataset_name)
