import os
import argparse
from clearml import PipelineController, Task
from upload_data import upload_dataset, download_dataset
from preprocess_data import preprocess_dataset, preprocess_images
from model_training import train_model


def create_CropSpot_pipeline(pipeline_name, project_name, dataset_name, queue_name, pipeline):
    from clearml import PipelineController, Task
    from upload_data import upload_dataset, download_dataset
    from preprocess_data import preprocess_dataset, preprocess_images
    from model_training import train_model

    """
    Create a ClearML pipeline for the CropSpot project.

    Parameters:
        pipeline_name (str): Name of the pipeline.
        project_name (str): Name of the ClearML project.
        dataset_name (str): Name of the dataset.
        queue_name (str): Name of the queue to execute the pipeline.

    Returns:
        None
    """

    # # Initialize a new pipeline controller task
    # pipeline = PipelineController(
    #     name=pipeline_name,
    #     project=project_name,
    #     version="1.0",
    #     add_pipeline_tags=True,
    #     target_project=project_name,
    #     auto_version_bump=True,
    # )

    # Add pipeline-level parameters with defaults from function arguments
    pipeline.add_parameter(name="project_name", default=project_name)
    pipeline.add_parameter(name="dataset_name", default=dataset_name)
    pipeline.add_parameter(name="queue_name", default=queue_name)

    # Set the default execution queue
    pipeline.set_default_execution_queue(queue_name)

    # Step 1: Upload Data
    pipeline.add_function_step(
        name="Data_Upload",
        task_name="Upload Raw Data",
        function=upload_dataset,
        function_kwargs={
            "project_name": "${pipeline.project_name}",
            "dataset_name": "${pipeline.dataset_name}",
            "queue_name": "${pipeline.queue_name}",
        },
        task_type=Task.TaskTypes.data_processing,
        function_return=["raw_dataset_id", "raw_dataset_name"],
        helper_functions=[download_dataset],
        parents=None,
        project_name=project_name,
        cache_executed_step=False,
    )

    # Step 2: Preprocess Data
    pipeline.add_function_step(
        name="Data_Preprocessing",
        task_name="Preprocess Uploaded Data",
        function=preprocess_dataset,
        function_kwargs={
            "dataset_name": "${Data_Upload.raw_dataset_name}",
            "project_name": "${pipeline.project_name}",
            "queue_name": "${pipeline.queue_name}",
        },
        task_type=Task.TaskTypes.data_processing,
        function_return=["processed_dataset_id", "processed_dataset_name"],
        helper_functions=[preprocess_images],
        parents=["Data_Upload"],
        project_name=project_name,
        cache_executed_step=False,
    )

    # Step 3: Train Model
    pipeline.add_function_step(
        name="Model_Training",
        task_name="Train Model",
        function=train_model,
        function_kwargs={
            "dataset_name": "${Data_Preprocessing.processed_dataset_name}",
            "project_name": "${pipeline.project_name}",
            "queue_name": "${pipeline.queue_name}",
        },
        task_type=Task.TaskTypes.training,
        function_return=["model_file_name"],
        parents=["Data_Preprocessing"],
        project_name=project_name,
        cache_executed_step=False,
    )

    # # Step 4: Evaluate Model
    # pipeline.add_function_step(
    #     name="Model_Evaluation",
    #     task_name="Evaluate Model",
    #     function=evaluate_model,
    #     function_kwargs={
    #         "dataset_name": "${pipeline.dataset_name}",
    #         "project_name": "${pipeline.project_name}",
    #         "queue_name": "${pipeline.queue_name}",
    #     },
    #     task_type=Task.TaskTypes.testing,
    #     project_name=project_name,
    #     parents=["Model_Training"],
    #     cache_executed_step=False,
    # )

    # # Step 5: Update Model in GitHub Repository
    # pipeline.add_function_step(
    #     name="Model_Update",
    #     task_name="Update Model",
    #     function=update_model,
    #     function_kwargs={
    #         "dataset_name": "${pipeline.dataset_name}",
    #         "project_name": "${pipeline.project_name}",
    #         "queue_name": "${pipeline.queue_name}",
    #     },
    #     task_type=Task.TaskTypes.system,
    #     project_name=project_name,
    #     parents=["Model_Evaluation"],
    #     cache_executed_step=False,
    # )

    # Start the pipeline
    print("CropSpot Data Pipeline initiated. Check ClearML for progress.")
    # pipeline.start_locally(run_pipeline_steps_locally=True)
    pipeline.start(queue_name)
