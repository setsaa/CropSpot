import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader, Subset
from clearml import Task, Dataset, OutputModel


def train_model(dataset_name, project_name, queue_name):
    # import os
    # import argparse
    # import numpy as np
    # from sklearn.model_selection import train_test_split
    # import torch
    # import torch.nn as nn
    # import torch.optim as optim
    # from torchvision import transforms, models, datasets
    # from torch.utils.data import DataLoader, Subset
    # from clearml import Task, Dataset, OutputModel

    # # Create ClearML task
    # task = Task.init(project_name=project_name, task_name="Model Training", task_type=Task.TaskTypes.training)
    # task.execute_remotely(queue_name=queue_name, exit_process=True)

    # Parameters
    img_size = 224
    batch_size = 16
    epochs = 200
    patience = 10

    # Data augmentation and loading
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(45),
            transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load the dataset
    dataset = Dataset.get(dataset_name=dataset_name)
    dataset_path = dataset.get_local_copy()
    # print(f"Dataset path: {dataset_path}")

    # Get class folders from the dataset path
    classes = [folder for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))]
    # print(f"Classes: {classes}")

    # Load the dataset into an ImageFolder
    img_data = datasets.ImageFolder(root=dataset_path, transform=transform)

    # Get the targets and calculate the train and test split indices
    targets = np.array(img_data.targets)
    train_indices, test_indices = train_test_split(np.arange(len(targets)), test_size=0.9, stratify=targets, random_state=42)

    # Create the train and test datasets
    train_dataset = Subset(img_data, train_indices)
    test_dataset = Subset(img_data, test_indices)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # print(f"Train dataset size: {len(train_loader.dataset)}")
    # print(f"Test dataset size: {len(test_loader.dataset)}")

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model setup
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_ftrs, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, len(train_loader.dataset.dataset.classes)))

    # Move the model to device
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Early stopping initialization
    best_validation_loss = float("inf")
    patience_counter = 0

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation loop
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                validation_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch+1} Training Loss: {running_loss / len(train_loader)} Validation Loss: {validation_loss / len(test_loader)}")

        # Early stopping
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping")
            break

    print("Finished Training")

    # Save the model
    model_file_name = "CropSpot_Model.pth"
    torch.save(model.state_dict(), model_file_name)

    # output_model = OutputModel(model=model, model_desc="CropSpot's Image Classification Model", task=task)
    # output_model.update_weights(model.state_dict())
    # output_model.publish()

    # # Upload the model artifact
    # task.upload_artifact("Trained CropSpot Model", artifact_object=model_file_name)

    return model_file_name
    # return output_model.id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CropSpot's PyTorch model on AWS SageMaker with ClearML")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the preprocessed dataset")
    parser.add_argument("--project_name", type=str, required=True, help="Name of the ClearML project")
    parser.add_argument("--queue_name", type=str, required=True, help="Name of the ClearML queue for remote execution")

    args = parser.parse_args()

    model_id = train_model(args.dataset_name, args.project_name, args.queue_name)

    print(f"Model trained with ID: {model_id}")
