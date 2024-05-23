def compare_models(model_name_1, model_name_2, model_name_3, queue_name):
    from clearml import Task, Dataset, OutputModel, InputModel

    task = Task.init(project_name="CropSpot", task_name="Compare Models")
    # task.execute_remotely(queue_name=queue_name, exit_process=True)

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import tensorflow as tf
    from keras.preprocessing.image import ImageDataGenerator
    from keras.models import Model, load_model
    from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc
    from sklearn.preprocessing import label_binarize
    from itertools import cycle
    from math import ceil
    import pickle as pkl

    # Load the model
    model1 = InputModel(name=model_name_1).connect()
    model2 = InputModel(name=model_name_2).connect()
    model3 = InputModel(name=model_name_3).connect()

    # Load score from clearml
    model_1_score = model1.get_metric("val_accuracy")
    model_2_score = model2.get_metric("val_accuracy")
    model_3_score = model3.get_metric("val_accuracy")

    # Get best model
    best_model = max(model_1_score, model_2_score, model_3_score)

    # Return the id of best model based on f1 score
    model = [model_1_score, model_2_score, model_3_score]
    best_model = model.index(max(model))

    return best_model.id


if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Compare the published models")

    # Add arguments
    parser.add_argument("--model_path_1", type=str, required=False, default="Trained Models/cropspot_resnet_model.h5", help="Path to the 1st trained model file")
    parser.add_argument("--model_path_2", type=str, required=False, default="Trained Models/cropspot_densenet_model.h5", help="Path to the 2nd trained model file")
    parser.add_argument("--model_path_3", type=str, required=False, default="Trained Models/cropspot_CNN_model.h5", help="Path to the 3rd trained model file")
    parser.add_argument("--queue_name", type=str, required=False, default="helldiver", help="ClearML queue name")

    args = parser.parse_args()

    # Evaluate the model
    compare_models(args.model_path_1, args.model_path_2, args.model_path_3, args.queue_name)
