def compare_models(model_name_1, model_score_1, model_name_2, model_score_2, model_name_3, model_score_3, project_name):
    from clearml import Task, Dataset, OutputModel, InputModel

    task = Task.init(project_name="CropSpot", task_name="Compare Models")
    # task.execute_remotely(queue_name=queue_name, exit_process=True)

    # Find best models based on the F1 score
    best_model = None
    best_score = 0
    for model_name, model_score in zip([model_name_1, model_name_2, model_name_3], [model_score_1, model_score_2, model_score_3]):
        if model_score > best_score:
            best_model = model_name
            best_score = model_score

    # Load the best model
    model = InputModel(name=best_model[:-3], project=project_name, only_published=True)
    model.connect(task=task)

    return model.id


if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Compare the published models")

    # Add arguments
    parser.add_argument("--model_path_1", type=str, required=False, default="cropspot_resnet_model.h5", help="Path to the 1st trained model file")
    parser.add_argument("--model_path_2", type=str, required=False, default="cropspot_densenet_model.h5", help="Path to the 2nd trained model file")
    parser.add_argument("--model_path_3", type=str, required=False, default="cropspot_CNN_model.h5", help="Path to the 3rd trained model file")
    parser.add_argument("--queue_name", type=str, required=False, default="helldiver", help="ClearML queue name")

    args = parser.parse_args()

    # Evaluate the model
    compare_models(args.model_path_1, args.model_path_2, args.model_path_3, args.queue_name)
