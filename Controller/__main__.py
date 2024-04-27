import argparse

# Import the function to create the pipeline
from clearml import PipelineController
from pipeline import create_CropSpot_pipeline

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
        # default="default",
        default="helldiver",
        help="ClearML queue name",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Initialize a new pipeline controller task
    pipeline = PipelineController(
        name=args.pipeline_name,
        project=args.project_name,
        version="1.0",
        add_pipeline_tags=True,
        target_project=args.project_name,
        auto_version_bump=True,
    )

    # Call the function with the parsed arguments
    create_CropSpot_pipeline(
        pipeline_name=args.pipeline_name,
        project_name=args.project_name,
        dataset_name=args.dataset_name,
        queue_name=args.queue_name,
        pipeline=pipeline,
    )
