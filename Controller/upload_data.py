import os
import shutil
import argparse
import requests
import zipfile
from tqdm import tqdm
from clearml import Task, Dataset


def download_dataset(dataset_dir, dataset_name):
    import os
    import shutil
    import requests
    import zipfile
    from tqdm import tqdm

    """
    Download and extract dataset from URL.

    Parameters:
        dataset_dir (str): Directory to save and extract the downloaded dataset.
        dataset_name (str): Name of the dataset.
    """

    dataset_url = "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/bwh3zbpkpv-1.zip"

    response = requests.get(dataset_url, stream=True)

    if response.status_code == 200:
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024
        t = tqdm(total=total_size, unit="iB", unit_scale=True)

        zip_path = os.path.join(dataset_dir, f"{dataset_name}.zip")
        with open(zip_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    t.update(len(chunk))
                    file.write(chunk)
        t.close()

        if total_size != 0 and t.n != total_size:
            print("ERROR, something went wrong")

        # Extract the zip file
        zip_path = os.path.join(dataset_dir, f"{dataset_name}.zip")
        extract_dir = os.path.join(dataset_dir, dataset_name)
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            top_level_dir = os.path.commonprefix(zip_ref.namelist())
            with tqdm(total=len(zip_ref.namelist()), desc="Extracting files", unit="file") as pbar:
                for member in zip_ref.namelist():
                    zip_ref.extract(member, extract_dir)
                    pbar.update()

        # Keep only relevant data
        top_level_dir = [folder for folder in os.listdir(extract_dir) if os.path.isdir(os.path.join(extract_dir, folder))][0]
        new_extract_dir = os.path.join(extract_dir, top_level_dir)
        raw_data_dir = os.path.join(new_extract_dir, "Raw Data/CCMT Dataset/Tomato")

        for folder in os.listdir(raw_data_dir):
            os.rename(os.path.join(raw_data_dir, folder), os.path.join(extract_dir, folder))

        # Remove unnecessary folders
        shutil.rmtree(new_extract_dir)

        # Remove the zip file
        os.remove(zip_path)

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
    import os
    import shutil
    from clearml import Task, Dataset

    # Create a ClearML task
    task = Task.init(project_name=project_name, task_name="Dataset Upload", task_type=Task.TaskTypes.data_processing)
    # task.execute_remotely(queue_name=queue_name, exit_process=True)

    dataset_dir = "./"

    # Check if dataset already exists on ClearML
    existing_dataset = Dataset.get(dataset_name=dataset_name)
    if existing_dataset:
        print(f"Dataset '{dataset_name}' already exists in project '{project_name}'.")

        return existing_dataset.id, existing_dataset.name

    # Download the dataset
    download_dataset(dataset_dir, dataset_name)

    # Create a directory with the dataset name if it doesn't exist
    dataset_path = os.path.join(dataset_dir, dataset_name)

    # Create a ClearML dataset
    dataset = Dataset.create(dataset_name=dataset_name, dataset_project=project_name)

    # Add the dataset directory to the dataset
    dataset.add_files(dataset_path)

    # Upload the dataset to ClearML
    dataset.upload()

    # Finalize the dataset
    dataset.finalize()

    # Remove the dataset directory
    shutil.rmtree(dataset_path)

    print(f"Dataset uploaded with ID: {dataset.id} and name: {dataset.name}")

    return dataset.id, dataset.name


if __name__ == "__main__":
    # import argparse

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
