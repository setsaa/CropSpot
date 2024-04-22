import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
import argparse
from scipy import interp
from itertools import cycle


def evaluate_model(model_path, history_path, test_data_dir, batch_size, img_size):
    import torch
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pickle as pkl
    import argparse
    from scipy import interp
    from itertools import cycle

    """
    Evaluate a PyTorch model on a test dataset.
    
    Parameters:
        model_path (str): Path to the model file.
        history_path (str): Path to the training history file.
        test_data_dir (str): Directory with test data images.
        batch_size (int): Batch size for testing.
        img_size (int): Size of the input images.
        
    Returns:
        f1 (float): F1 Score of the model.
    """

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path)
    model = model.to(device)
    model.eval()

    # Load history
    with open(history_path, "rb") as file:
        history = pkl.load(file)

    # Data preprocessing
    transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # Setup DataLoader
    test_dataset = torchvision.datasets.ImageFolder(root=test_data_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Prediction and Metrics
    all_labels = []
    all_preds = []
    all_outputs = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_outputs.extend(outputs.cpu().numpy())

    # Compute metrics
    cm = confusion_matrix(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    # Plotting confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = len(test_dataset.classes)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(np.array(all_labels) == i, np.array(all_outputs)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure()
    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label="ROC curve of class {0} (area = {1:0.2f})" "".format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic for multi-class data")
    plt.legend(loc="lower right")
    plt.show()

    return f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a PyTorch Model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model file")
    parser.add_argument("--history_path", type=str, required=True, help="Path to the training history file")
    parser.add_argument("--test_data_dir", type=str, required=True, help="Directory with test data images")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for testing")
    parser.add_argument("--img_size", type=int, default=224, help="Size of the input images")
    args = parser.parse_args()

    f1 = evaluate_model(args.model_path, args.history_path, args.test_data_dir, args.batch_size, args.img_size)

    print(f"F1 Score: {f1}")
