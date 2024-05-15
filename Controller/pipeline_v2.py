from clearml import PipelineDecorator, Task
from upload_data import upload_dataset, download_dataset
from preprocess_data import preprocess_dataset
from resnet_train import resnet_train
from densenet_train import densenet_train
from cnn_train import custom_cnn_train
from model_evaluation import evaluate_model
from update_model import update_repository
from compare_models import compare_models


# Step 1: Upload Data
@PipelineDecorator.component(name="Upload_Data", return_values=["dataset_id", "dataset_name"], cache=False, task_type=Task.TaskTypes.data_processing)
def upload_data_pipeline(project_name, dataset_name, queue_name):
    return upload_dataset(project_name=project_name, dataset_name=dataset_name, queue_name=queue_name)


# Step 2: Preprocess Data
@PipelineDecorator.component(name="Preprocess_Data", return_values=["preprocessed_dataset_id", "preprocessed_dataset_name"], cache=False, task_type=Task.TaskTypes.data_processing)
def preprocess_data_pipeline(dataset_name, project_name, queue_name):
    return preprocess_dataset(dataset_name=dataset_name, project_name=project_name, queue_name=queue_name)


# Step 3(a): Train ResNet Model
@PipelineDecorator.component(name="Train_ResNet", return_values=["resnet_model_id"], cache=False, task_type=Task.TaskTypes.training)
def resnet_train_pipeline(dataset_name, project_name, queue_name):
    return resnet_train(dataset_name=dataset_name, project_name=project_name, queue_name=queue_name)


# Step 3(b): Train DenseNet Model
@PipelineDecorator.component(name="Train_DenseNet", return_values=["densenet_model_id"], cache=False, task_type=Task.TaskTypes.training)
def densenet_train_pipeline(dataset_name, project_name, queue_name):
    return densenet_train(dataset_name=dataset_name, project_name=project_name, queue_name=queue_name)


# Step 3(c): Train CNN Model
@PipelineDecorator.component(name="Train_CNN", return_values=["custom_cnn_model_id"], cache=False, task_type=Task.TaskTypes.training)
def custom_cnn_train_pipeline(dataset_name, project_name, queue_name):
    return custom_cnn_train(dataset_name=dataset_name, project_name=project_name, queue_name=queue_name)


# Step 4(a): Evaluate ResNet Model
@PipelineDecorator.component(name="Eval_ResNet", return_values=["resnet_evaluation_id"], cache=False, task_type=Task.TaskTypes.testing)
def evaluate_model_pipeline(model_path, history_path, test_data_dir, queue_name):
    return evaluate_model(model_path=model_path, history_path=history_path, test_data_dir=test_data_dir, queue_name=queue_name)


# Step 4(b): Evaluate DenseNet Model
@PipelineDecorator.component(name="Eval_DenseNet", return_values=["densenet_evaluation_id"], cache=False, task_type=Task.TaskTypes.testing)
def evaluate_model_pipeline(model_path, history_path, test_data_dir, queue_name):
    return evaluate_model(model_path=model_path, history_path=history_path, test_data_dir=test_data_dir, queue_name=queue_name)


# Step 4(c): Evaluate CNN Model
@PipelineDecorator.component(name="Eval_CNN", return_values=["cnn_evaluation_id"], cache=False, task_type=Task.TaskTypes.testing)
def evaluate_model_pipeline(model_path, history_path, test_data_dir, queue_name):
    return evaluate_model(model_path=model_path, history_path=history_path, test_data_dir=test_data_dir, queue_name=queue_name)


# Step 5: Compare Models
@PipelineDecorator.component(name="Compare_Models", return_values=["model_comparison_id"], cache=False, task_type=Task.TaskTypes.testing)
def compare_models(model_path_1, model_path_2, model_path_3, test_data_dir, queue_name):
    return compare_models(model_path_1=model_path_1, model_path_2=model_path_2, model_path_3=model_path_3, test_data_dir=test_data_dir, queue_name=queue_name)


# # Step 5: Update Model in GitHub Repository
# @PipelineDecorator(name="Update Model in GitHub Repository", project_name="CropSpot", task_name="Update Model in GitHub Repository", task_type=PipelineDecorator.TaskTypes.training)
# def update_model_pipeline(repo_path, branch, commit_message, project_name, model_name):
#     return update_repository(repo_path=repo_path, branch_name=branch, commit_message=commit_message, project_name=project_name, model_name=model_name)


# Create a ClearML pipeline for the CropSpot project.
@PipelineDecorator.pipeline(name="CropSpot Pipeline", project="CropSpot", version="1.0", pipeline_execution_queue="helldiver_2")
def create_CropSpot_pipeline(
    pipeline_name,
    project_name,
    dataset_name,
    queue_name,
    model_path_1,
    model_path_2,
    model_path_3,
    test_data_dir,
    repo_path,
    branch,
    commit_message,
    model_name,
):
    # Step 1: Upload Data
    uploaded_dataset_id, uploaded_dataset_name = upload_data_pipeline(project_name=project_name, dataset_name=dataset_name, queue_name=queue_name)

    # Step 2: Preprocess Data
    prep_dataset_id, prep_dataset_name = preprocess_data_pipeline(dataset_name=uploaded_dataset_name, project_name=project_name, queue_name=queue_name)

    # Step 3(a): Train ResNet Model
    resnet_id = resnet_train_pipeline(dataset_name=prep_dataset_name, project_name=project_name, queue_name=queue_name)

    # Step 3(b): Train DenseNet Model
    densenet_id = densenet_train_pipeline(dataset_name=prep_dataset_name, project_name=project_name, queue_name=queue_name)

    # Step 3(c): Train CNN Model
    cnn_id = custom_cnn_train_pipeline(dataset_name=prep_dataset_name, project_name=project_name, queue_name=queue_name)

    # Step 4(a): Evaluate ResNet Model
    resnet_f1 = evaluate_model_pipeline(model_path=resnet_id, history_path="Trained Models/resnet_history.json", test_data_dir=test_data_dir, queue_name=queue_name)

    # Step 4(b): Evaluate DenseNet Model
    densenet_f1 = evaluate_model_pipeline(model_path=densenet_id, history_path="Trained Models/densenet_history.json", test_data_dir=test_data_dir, queue_name=queue_name)

    # Step 4(c): Evaluate CNN Model
    cnn_f1 = evaluate_model_pipeline(model_path=cnn_id, history_path="Trained Models/cnn_history.json", test_data_dir=test_data_dir, queue_name=queue_name)

    # Step 5: Compare Models
    best_id = compare_models(model_path_1=resnet_id, model_path_2=densenet_id, model_path_3=cnn_id, test_data_dir=test_data_dir, queue_name=queue_name)

    # # Step 5: Update Model in GitHub Repository
    # update_model_pipeline(repo_path=repo_path, branch=branch, commit_message=commit_message, project_name=project_name, model_name=model_name)

    print("CropSpot Pipeline initiated. Check ClearML for progress.")
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
        default="helldiver_2",
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

    # PipelineDecorator.set_default_execution_queue(args.queue_name)

    # Call the function with the parsed arguments
    create_CropSpot_pipeline(
        pipeline_name=args.pipeline_name,
        project_name=args.project_name,
        dataset_name=args.dataset_name,
        queue_name=args.queue_name,
        model_path_1=args.model_path_1,
        model_path_2=args.model_path_2,
        model_path_3=args.model_path_3,
        test_data_dir=args.test_data_dir,
        repo_path=args.repo_path,
        branch=args.branch,
        commit_message=args.commit_message,
        model_name=args.model_name,
    )

    PipelineDecorator.start()
