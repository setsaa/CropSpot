def create_data_pipeline(
    pipeline_name: str = "CropSpot Data Pipeline",
    dataset_project: str = "CropSpot",
    raw_dataset_name: str = "TomatoDiseaseDataset",
    processed_dataset_name: str = "TomatoDiseaseDataset_preprocessed",
    queue_name: str = "helldiver",
):
    from clearml import PipelineController, Task
    from cropspot.upload_data import upload_local_dataset
    from cropspot.preprocess_uploaded_data import upload_preprocessed_dataset, preprocess_images

    # Initialize a new pipeline controller task
    pipeline = PipelineController(
        name=pipeline_name,
        project=dataset_project,
        version="1.0",
        add_pipeline_tags=True,
        auto_version_bump=True,
        target_project=dataset_project,
    )

    # Add pipeline-level parameters with defaults from function arguments
    pipeline.add_parameter(name="dataset_project", default=dataset_project)
    pipeline.add_parameter(name="raw_dataset_name", default=raw_dataset_name)
    pipeline.add_parameter(name="processed_dataset_name", default=processed_dataset_name)
    pipeline.add_parameter(name="queue_name", default=queue_name)

    # Step 1: Upload Raw Data
    pipeline.add_function_step(
        name="Data Upload",
        function=upload_local_dataset,
        function_kwargs={
            "dataset_dir": "CropSpot\Dataset\Raw Data",
            "clearml_project_name": "${pipeline.dataset_project}",
            "dataset_name": "${pipeline.raw_dataset_name}",
        },
        task_type=Task.TaskTypes.data_processing,
        task_name="Upload Raw Data",
        function_return=["raw_dataset_id"],
        cache_executed_step=False,
    )

    # Step 2: Preprocess Data
    pipeline.add_function_step(
        name="Data Preprocessing",
        function=upload_preprocessed_dataset,
        function_kwargs={
            "dataset_name": "${pipeline.raw_dataset_name}",
            "project_name": "${pipeline.dataset_project}",
            "queue_name": "${pipeline.queue_name}",
        },
        task_type=Task.TaskTypes.data_processing,
        task_name="Preprocess Uploaded Data",
        function_return=["processed_dataset_id"],
        helper_functions=[preprocess_images],
        cache_executed_step=False,
    )

    # Start the pipeline
    pipeline.start(queue=queue_name)
    print("CropSpot pipeline initiated. Check ClearML for progress.")


if __name__ == "__main__":
    import argparse

    # Create the parser
    parser = argparse.ArgumentParser(description="Run CropSpot Data Pipeline")
    parser.add_argument(
        "--pipeline_name",
        type=str,
        default="CropSpot Data Pipeline",
        help="Name of the pipeline",
    )
    parser.add_argument(
        "--dataset_project",
        type=str,
        default="CropSpot",
        help="Project name for datasets",
    )
    parser.add_argument(
        "--raw_dataset_name",
        type=str,
        default="TomatoDiseaseDataset",
        help="Name for the raw dataset",
    )
    parser.add_argument(
        "--processed_dataset_name",
        type=str,
        default="TomatoDiseaseDataset_preprocessed",
        help="Name for the processed dataset",
    )
    parser.add_argument(
        "--queue_name",
        type=str,
        required=True,
        default="helldiver",
        help="ClearML queue name",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the parsed arguments
    create_data_pipeline(
        pipeline_name=args.pipeline_name,
        dataset_project=args.dataset_project,
        raw_dataset_name=args.raw_dataset_name,
        processed_dataset_name=args.processed_dataset_name,
        queue_name=args.queue_name,
    )
