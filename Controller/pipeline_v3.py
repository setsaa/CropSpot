import os
import argparse
from clearml import PipelineController, Task
from upload_data import upload_dataset
from preprocess_data import preprocess_dataset
from model_training import train_model


def step_one(project_name, dataset_name, queue_name):
    return upload_dataset(project_name, dataset_name, queue_name)


def step_two(project_name, dataset_name, queue_name):
    return preprocess_dataset(project_name, dataset_name, queue_name)


def step_three(dataset_name, project_name, queue_name):
    return train_model(dataset_name, project_name, queue_name)


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Run CropSpot Pipeline")

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
        default="helldiver",
        help="ClearML queue name",
    )

    # Parse the arguments
    args = parser.parse_args()

    # create the pipeline controller
    pipeline = PipelineController(
        project=args.project_name,
        name=args.pipeline_name,
        version="1.0",
        add_pipeline_tags=True,
        target_project=args.project_name,
        auto_version_bump=True,
    )

    # set the default execution queue to be used (per step we can override the execution)
    pipeline.set_default_execution_queue(args.queue_name)

    # add pipeline components
    pipeline.add_parameter(name="project_name", default=args.project_name)
    pipeline.add_parameter(name="dataset_name", default=args.dataset_name)
    pipeline.add_parameter(name="queue_name", default=args.queue_name)

    pipeline.add_function_step(
        name="step_one",
        function=step_one,
        function_kwargs=dict(
            project_name="${pipeline.project_name}",
            dataset_name="${pipeline.dataset_name}",
            queue_name="${pipeline.queue_name}",
        ),
        function_return=["raw_dataset_id", "raw_dataset_name"],
        cache_executed_step=True,
    )
    pipeline.add_function_step(
        name="step_two",
        function=step_two,
        function_kwargs=dict(
            project_name="${pipeline.project_name}",
            dataset_name="${step_one.raw_dataset_name}",
            queue_name="${pipeline.queue_name}",
        ),
        function_return=["processed_dataset_id", "processed_dataset_name"],
        cache_executed_step=True,
    )
    pipeline.add_function_step(
        name="step_three",
        function=step_three,
        function_kwargs=dict(
            project_name="${pipeline.project_name}",
            dataset_name="${step_two.processed_dataset_name}",
            queue_name="${pipeline.queue_name}",
        ),
        function_return=["model_id"],
        cache_executed_step=True,
    )

    pipeline.start(queue=args.queue_name)

    print("Pipeline completed")
