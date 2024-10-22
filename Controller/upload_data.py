def upload_dataset(project_name, dataset_name):
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
    from clearml import Task, Dataset

    task = Task.init(project_name=project_name, task_name="Upload Raw Data")

    dataset_dir = "./Dataset/TomatoDiseaseDatasetV2"

    # Check if dataset already exists on ClearML
    existing_dataset = Dataset.get(dataset_name=dataset_name)
    if existing_dataset:
        print(f"Dataset '{dataset_name}' already exists in project '{project_name}'.")

        return existing_dataset.id, existing_dataset.name

    # Create a ClearML dataset
    dataset = Dataset.create(dataset_name=dataset_name, dataset_project=project_name)

    # Add the dataset directory to the dataset
    dataset.add_files(dataset_dir)

    # Upload the dataset to ClearML
    dataset.upload()

    # Finalize the dataset
    dataset.finalize()

    # # Remove the dataset directory
    # shutil.rmtree(dataset_dir)

    print(f"Dataset uploaded with ID: {dataset.id} and name: {dataset.name}")

    return dataset.id, dataset.name


def download_dataset(dataset_dir, dataset_name):
    """
    Download and extract dataset from URL.

    Parameters:
        dataset_dir (str): Directory to save and extract the downloaded dataset.
        dataset_name (str): Name of the dataset.
    """
    import os
    import shutil
    import requests
    import zipfile
    from tqdm import tqdm

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
