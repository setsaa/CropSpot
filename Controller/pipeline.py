def create_CropSpot_pipeline(
    pipeline_name,
    project_name,
    dataset_name,
    queue_name,
    model_path,
    model_history_path,
    test_data_dir,
    repo_path,
    branch,
    commit_message,
    model_name,
):
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

    from clearml import PipelineController, Task
    from upload_data import upload_dataset, download_dataset
    from preprocess_data import preprocess_dataset
    from model_training import train_model
    from model_evaluation import evaluate_model
    from update_model import update_repository

    # Initialize a new pipeline controller task
    pipeline = PipelineController(
        name=pipeline_name,
        project=project_name,
        version="1.0",
        add_pipeline_tags=True,
        target_project=project_name,
        auto_version_bump=True,
    )

    # Add pipeline-level parameters with defaults from function arguments
    pipeline.add_parameter(name="project_name", default=project_name)
    pipeline.add_parameter(name="dataset_name", default=dataset_name)
    pipeline.add_parameter(name="queue_name", default=queue_name)
    pipeline.add_parameter(name="model_path", default=model_path)
    pipeline.add_parameter(name="model_history_path", default=model_history_path)
    pipeline.add_parameter(name="test_data_dir", default=test_data_dir)
    pipeline.add_parameter(name="repo_path", default=repo_path)
    pipeline.add_parameter(name="branch", default=branch)
    pipeline.add_parameter(name="commit_message", default=commit_message)
    pipeline.add_parameter(name="model_name", default=model_name)

    # Set the default execution queue
    pipeline.set_default_execution_queue(queue_name)

    # Step 1: Upload Data
    pipeline.add_function_step(
        name="Data_Upload",
        task_name="Upload Raw Data",
        function=upload_dataset,
        function_kwargs=dict(
            project_name="${pipeline.project_name}",
            dataset_name="${pipeline.dataset_name}",
            queue_name="${pipeline.queue_name}",
        ),
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
        function_kwargs=dict(
            dataset_name="${pipeline.dataset_name}",
            project_name="${pipeline.project_name}",
            queue_name="${pipeline.queue_name}",
        ),
        task_type=Task.TaskTypes.data_processing,
        function_return=["processed_dataset_id", "processed_dataset_name"],
        parents=["Data_Upload"],
        project_name=project_name,
        cache_executed_step=False,
    )

    # Step 3: Train Model
    pipeline.add_function_step(
        name="Model_Training",
        task_name="Train Model",
        function=train_model,
        function_kwargs=dict(
            dataset_name="${pipeline.dataset_name}",
            project_name="${pipeline.project_name}",
            queue_name="${pipeline.queue_name}",
        ),
        task_type=Task.TaskTypes.training,
        function_return=["model_id"],
        parents=["Data_Preprocessing"],
        project_name=project_name,
        cache_executed_step=False,
    )

    # Step 4: Evaluate Model
    pipeline.add_function_step(
        name="Model_Evaluation",
        task_name="Evaluate Model",
        function=evaluate_model,
        function_kwargs={
            "model_path": "${pipeline.model_path}",
            "history_path": "${pipeline.model_history_path}",
            "test_data_dir": "${pipeline.test_data_dir}",
        },
        task_type=Task.TaskTypes.testing,
        function_return=["f1_score"],
        parents=["Model_Training"],
        project_name=project_name,
        cache_executed_step=False,
    )

    # Step 5: Update Model in GitHub Repository
    pipeline.add_function_step(
        name="GitHub_Update",
        task_name="Update Model Weights in GitHub Repository",
        function=update_repository,
        function_kwargs={
            "repo_path": "${pipeline.repo_path}",
            "branch_name": "${pipeline.branch}",
            "commit_message": "${pipeline.commit_message}",
            "project_name": "${pipeline.project_name}",
            "model_name": "${pipeline.model_name}",
        },
        task_type=Task.TaskTypes.service,
        parents=["Model_Training", "Model_Evaluation"],
        project_name=project_name,
        cache_executed_step=False,
    )

    # Start the pipeline
    print("CropSpot Data Pipeline initiated. Check ClearML for progress.")
    pipeline.start_locally(run_pipeline_steps_locally=True)


if __name__ == "__main__":
    import argparse

    # Create the parser
    parser = argparse.ArgumentParser(description="Run CropSpot Data Pipeline")

    # Add arguments
    parser.add_argument(
        "--pipeline_name",
        type=str,
        required=False,
        default="CropSpot Pipeline",
        help="Name of the pipeline",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        required=False,
        default="CropSpot",
        help="Project name for datasets",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=False,
        default="TomatoDiseaseDataset",
        help="Name for the raw dataset",
    )
    parser.add_argument(
        "--queue_name",
        type=str,
        required=False,
        default="default",
        help="ClearML queue name",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        default="Trained Models/CropSpot_Model.h5",
        help="Local model path",
    )
    parser.add_argument(
        "--model_history_path",
        type=str,
        required=False,
        default="Trained Models/CropSpot_Model_History.pkl",
        help="Local model history path",
    )
    parser.add_argument(
        "--test_data_dir",
        type=str,
        required=False,
        default="Dataset/Preprocessed",
        help="Directory containing test data",
    )
    parser.add_argument(
        "--repo_path",
        type=str,
        required=False,
        default=".",
        help="Path to the local Git repository",
    )
    parser.add_argument(
        "--branch",
        type=str,
        required=False,
        default="CROP-28-AUTOMATED",
        help="The branch to commit and push changes to",
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        required=False,
        default="Automated commit of model changes",
        help="Commit message",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        default="CropSpot_Model",
        help="ClearML trained model",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the parsed arguments
    create_CropSpot_pipeline(
        pipeline_name=args.pipeline_name,
        project_name=args.dataset_project,
        dataset_name=args.dataset_name,
        queue_name=args.queue_name,
    )
