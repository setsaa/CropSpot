def evaluate_model(model_name, test_dataset, task_name, project_name):
    from clearml import Task, Dataset, OutputModel, InputModel

    task = Task.init(project_name="CropSpot", task_name=task_name)
    # task.execute_remotely(queue_name=queue_name, exit_process=True)

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from keras.preprocessing.image import ImageDataGenerator
    from keras.models import Model, load_model
    from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc
    from sklearn.preprocessing import label_binarize
    from itertools import cycle
    from math import ceil
    import pickle as pkl

    # Load the model
    input_model = InputModel(name=model_name[:-3], project=project_name, only_published=True)
    input_model.connect(task=task)
    local_model = input_model.get_local_copy()
    model = load_model(local_model)

    dataset = Dataset.get(dataset_name=test_dataset)

    # Check if the dataset is already downloaded. If not, download it. Otherwise, use the existing dataset.
    dataset_path = f"Dataset/{test_dataset}"
    if not os.path.exists(dataset_path):
        dataset.get_mutable_local_copy(dataset_path)

    # # Get image size from the first image from the healthy directory
    # first_category = os.listdir(dataset_path)[0]
    # first_image_file = os.listdir(f"{dataset_path}/{first_category}")[0]
    # img = plt.imread(f"{dataset_path}/{first_category}/{first_image_file}")
    # img_height, img_width, _ = img.shape
    # img_size = min(img_height, img_width)
    img_size = 224

    batch_size = 16

    # Data generator for evaluation
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_generator = test_datagen.flow_from_directory(dataset_path, target_size=(img_size, img_size), batch_size=batch_size, class_mode="categorical", shuffle=True)

    # Calculate the correct number of steps per epoch
    steps = ceil(test_generator.samples / test_generator.batch_size)

    # Generate predictions
    predictions = model.predict(test_generator, steps=steps)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes[: len(y_pred)]

    # Evaluate the model
    score = model.evaluate(test_generator)
    print(f"Test loss: {score[0]:.3f}")
    print(f"Test accuracy: {score[1]:.3f}")

    # Printing the f1 score
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"F1 Score: {f1}")

    # Plotting the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix")
    plt.show()

    # Prepare for ROC curve plot
    y_test_binarized = label_binarize(y_true, classes=np.arange(test_generator.num_classes))
    n_classes = y_test_binarized.shape[1]
    fpr, tpr, roc_auc = dict(), dict(), dict()
    colors = cycle(["blue", "red", "green"])
    plt.figure()
    for i, color in zip(range(n_classes), colors):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-class ROC")
    plt.legend(loc="lower right")
    plt.show()

    return score[1]


if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate the model")

    # Add arguments
    parser.add_argument("--model_path", type=str, required=False, default="Trained Models/CropSpot_Model.h5", help="Path to the trained model file")
    parser.add_argument("--history_path", type=str, required=False, default="Trained Models/CropSpot_Model_History.pkl", help="Path to the training history file")
    parser.add_argument("--test_dataset", type=str, required=False, default="Dataset/Raw Data", help="Directory containing data")

    args = parser.parse_args()

    # Evaluate the model
    evaluate_model(args.model_path, args.history_path, args.test_dataset)
