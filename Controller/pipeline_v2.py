from clearml import PipelineDecorator, Task
from upload_data import upload_dataset, download_dataset
from preprocess_data import preprocess_dataset
from resnet_train import resnet_train
from densenet_train import densenet_train
from cnn_train import custom_cnn_train
from model_evaluation import evaluate_model
from update_model import update_repository


# Step 1: Upload Data
@PipelineDecorator.component(return_values=["dataset_id", "dataset_name"], cache=False, task_type=Task.TaskTypes.data_processing)
def upload_data_pipeline(project_name, dataset_name, queue_name):
    return upload_dataset(project_name=project_name, dataset_name=dataset_name, queue_name=queue_name)


# Step 2: Preprocess Data
@PipelineDecorator.component(return_values=["preprocessed_dataset_id", "preprocessed_dataset_name"], cache=False, task_type=Task.TaskTypes.data_processing)
def preprocess_data_pipeline(dataset_name, project_name, queue_name):
    return preprocess_dataset(dataset_name=dataset_name, project_name=project_name, queue_name=queue_name)


# Step 3(a): Train ResNet Model
@PipelineDecorator.component(return_values=["resnet_model_id"], cache=False, task_type=Task.TaskTypes.training)
def resnet_train_pipeline(dataset_name, project_name, queue_name):
    return resnet_train(dataset_name=dataset_name, project_name=project_name, queue_name=queue_name)


# Step 3(b): Train DenseNet Model
@PipelineDecorator.component(return_values=["densenet_model_id"], cache=False, task_type=Task.TaskTypes.training)
def densenet_train_pipeline(dataset_name, project_name, queue_name):
    return densenet_train(dataset_name=dataset_name, project_name=project_name, queue_name=queue_name)


# Step 3(c): Train CNN Model
@PipelineDecorator.component(return_values=["custom_cnn_model_id"], cache=False, task_type=Task.TaskTypes.training)
def custom_cnn_train_pipeline(dataset_name, project_name, queue_name):
    return custom_cnn_train(dataset_name=dataset_name, project_name=project_name, queue_name=queue_name)


# Step 4: Evaluate Model
@PipelineDecorator.component(return_values=["f1_score"], cache=False, task_type=Task.TaskTypes.testing)
def evaluate_model_pipeline(model_path, history_path, test_data_dir, queue_name):
    return evaluate_model(model_path=model_path, history_path=history_path, test_data_dir=test_data_dir, queue_name=queue_name)


# # Step 5: Update Model in GitHub Repository
# @PipelineDecorator(name="Update Model in GitHub Repository", project_name="CropSpot", task_name="Update Model in GitHub Repository", task_type=PipelineDecorator.TaskTypes.training)
# def update_model_pipeline(repo_path, branch, commit_message, project_name, model_name):
#     return update_repository(repo_path=repo_path, branch_name=branch, commit_message=commit_message, project_name=project_name, model_name=model_name)


# Create a ClearML pipeline for the CropSpot project.
@PipelineDecorator.pipeline(name="CropSpot Data Pipeline", project="CropSpot", version="1.0")
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
    # Step 1: Upload Data
    upload_data_pipeline(project_name=project_name, dataset_name=dataset_name, queue_name=queue_name)

    # Step 2: Preprocess Data
    preprocess_data_pipeline(dataset_name=dataset_name, project_name=project_name, queue_name=queue_name)

    # Step 3(a): Train ResNet Model
    resnet_train_pipeline(dataset_name=dataset_name, project_name=project_name, queue_name=queue_name)

    # Step 3(b): Train DenseNet Model
    densenet_train_pipeline(dataset_name=dataset_name, project_name=project_name, queue_name=queue_name)

    # Step 3(c): Train CNN Model
    custom_cnn_train_pipeline(dataset_name=dataset_name, project_name=project_name, queue_name=queue_name)

    # Step 4: Evaluate Model
    evaluate_model_pipeline(model_path=model_path, history_path=model_history_path, test_data_dir=test_data_dir, queue_name=queue_name)

    # # Step 5: Update Model in GitHub Repository
    # update_model_pipeline(repo_path=repo_path, branch=branch, commit_message=commit_message, project_name=project_name, model_name=model_name)

    print("CropSpot Data Pipeline initiated. Check ClearML for progress.")
    return


if __name__ == "__main__":
    import argparse

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
    parser.add_argument(
        "--model_path_1",
        type=str,
        required=False,
        default="Trained Models/cropspot_resnet_model.h5",
        help="Local model path",
    )
    parser.add_argument(
        "--model_path_2",
        type=str,
        required=False,
        default="Trained Models/cropspot_densenet_model.h5",
        help="Local model path",
    )
    parser.add_argument(
        "--model_path_3",
        type=str,
        required=False,
        default="Trained Models/cropspot_CNN_model.h5",
        help="Local model path",
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
        default="Crop-33-Deploy-MLOPs-pipeline",
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

    PipelineDecorator.set_default_execution_queue(args.queue_name)

    # Call the function with the parsed arguments
    create_CropSpot_pipeline(
        pipeline_name=args.pipeline_name,
        project_name=args.project_name,
        dataset_name=args.dataset_name,
        queue_name=args.queue_name,
        model_path=args.model_path,
        test_data_dir=args.test_data_dir,
        repo_path=args.repo_path,
        branch=args.branch,
        commit_message=args.commit_message,
        model_name=args.model_name,
    )

    PipelineDecorator.start(queue=args.queue_name)
