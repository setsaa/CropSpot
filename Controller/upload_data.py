import os
import argparse
import requests
import zipfile
from clearml import Task, Dataset


"""
    This script uploads a local dataset to ClearML.
"""


def download_dataset(dataset_url, dataset_dir):
    """
    Download dataset from a URL.

    Parameters:
        dataset_url (str): URL of the dataset.
        dataset_dir (str): Directory to save the downloaded dataset.
    """

    response = requests.get(dataset_url, stream=True)

    if response.status_code == 200:
        with open(os.path.join(dataset_dir, "dataset.zip"), "wb") as file:
            file.write(response.content)
    else:
        raise ValueError(f"Failed to download the dataset. HTTP response code: {response.status_code}")


def upload_dataset(project_name, dataset_name, queue_name):
    """
    Upload dataset to a ClearML project.

    Parameters:
        project_name (str): Name of the ClearML project.
        dataset_name (str): Name of the dataset.
        queue_name (str): Name of the queue to execute the task.

    Returns:
        dataset_id (str): ID of the uploaded dataset.
        dataset_name (str): Name of the uploaded dataset.
    """

    # Create a ClearML task
    task = Task.init(project_name=project_name, task_name="Dataset Upload", task_type=Task.TaskTypes.data_processing)
    task.execute_remotely(queue_name=queue_name, exit_process=True)

    dataset_url = "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/bwh3zbpkpv-1.zip"
    dataset_dir = "./"

    # Download the dataset
    download_dataset(dataset_url, dataset_dir)

    # Extract the zip file
    with zipfile.ZipFile(os.path.join(dataset_dir, "dataset.zip"), "r") as zip_ref:
        zip_ref.extractall(dataset_dir)

    # Ensure the dataset directory is valid or empty
    if not os.path.isdir(dataset_dir):
        raise ValueError(f"The specified path '{dataset_dir}' is not a directory or does not exist.")
    if not os.listdir(dataset_dir):
        raise ValueError(f"The specified path '{dataset_dir}' is empty.")

    # Create a ClearML dataset
    dataset = Dataset.create(dataset_name=dataset_name, dataset_project=project_name)

    # Add the dataset directory to the dataset
    dataset.add_files(dataset_dir)

    # Upload the dataset to ClearML
    dataset.upload()

    # Finalize the dataset
    dataset.finalize()

    print(f"Dataset uploaded with ID: {dataset.id} and name: {dataset.name}")

    return dataset.id, dataset.name


if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Upload Dataset Directory to ClearML")

    # Add arguments
    parser.add_argument("--project_name", type=str, required=True, help="ClearML project name")
    parser.add_argument("--dataset_name", type=str, required=True, help="ClearML dataset name")
    parser.add_argument("--queue_name", type=str, default="default", help="ClearML queue name")

    # Parse command-line arguments
    args = parser.parse_args()

    # Call function with the parsed arguments
    dataset_id, dataset_name = upload_dataset(args.project_name, args.dataset_name, args.queue_name)

    print(f"Dataset uploaded with ID: {dataset_id} and name: {dataset_name}")
