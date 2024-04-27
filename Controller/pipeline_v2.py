import os
import argparse
from clearml import PipelineDecorator, Task
from upload_data import upload_dataset
from preprocess_data import preprocess_dataset
from model_training import train_model


@PipelineDecorator.component(return_values=["raw_dataset_id", "raw_dataset_name"], cache=False, task_type=Task.TaskTypes.data_processing, parents=None, helper_functions=[upload_dataset])
def step_one(project_name, dataset_name, queue_name):
    return upload_dataset(project_name, dataset_name, queue_name)


@PipelineDecorator.component(return_values=["prep_dataset_id", "prep_dataset_name"], cache=False, task_type=Task.TaskTypes.data_processing, parents=[step_one], helper_functions=[preprocess_dataset])
def step_two(project_name, raw_dataset_name, queue_name):
    return preprocess_dataset(project_name, raw_dataset_name, queue_name)


@PipelineDecorator.component(return_values=["model_id"], cache=False, task_type=Task.TaskTypes.training, parents=[step_two], helper_functions=[train_model])
def step_three(prep_dataset_name, project_name, queue_name):
    return train_model(prep_dataset_name, project_name, queue_name)


@PipelineDecorator.pipeline(name="CropSpot Pipeline", project="CropSpot", version="1.0", add_pipeline_tags=True, target_project="CropSpot", pipeline_execution_queue="helldiver")
def executing_pipeline(project_name, dataset_name, queue_name):

    print("Step 1: Upload Data")
    raw_dataset_id, raw_dataset_name = step_one(project_name, dataset_name, queue_name)

    print("Step 2: Preprocess Data")
    prep_dataset_id, prep_dataset_name = step_two(project_name, raw_dataset_name, queue_name)

    print("Step 3: Train Model")
    model_id = step_three(prep_dataset_name, project_name, queue_name)

    print("Model ID: ", model_id)


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

    PipelineDecorator.set_default_execution_queue(args.queue_name)

    executing_pipeline(args.project_name, args.dataset_name, args.queue_name)

    print("Pipeline executed successfully.")
