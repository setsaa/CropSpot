def train_model(dataset_name, project_name, queue_name):
    """
    Train the CropSpot model using the preprocessed dataset.

    Args:
        preprocessed_dataset_name (str): Name of the preprocessed dataset
        project_name (str): Name of the ClearML project
        queue_name (str): Name of the ClearML queue for remote execution

    Returns:
        ID of the trained model
    """
    import os
    import matplotlib.pyplot as plt
    from tensorflow.keras.models import Model
    from tensorflow.keras.callbacks import LambdaCallback
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications import ResNet50V2
    from tensorflow.keras.applications.resnet_v2 import preprocess_input
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Activation, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from clearml import Task, Dataset, OutputModel
    import pickle

    task = Task.init(project_name=project_name, task_name="Model Training", task_type=Task.TaskTypes.training)

    # Load preprocessed dataset
    dataset = Dataset.get(dataset_name=dataset_name + "_preprocessed")

    # Check if the dataset is already downloaded. If not, download it. Otherwise, use the existing dataset.
    dataset_path = "Dataset/Preprocessed"
    if not os.path.exists(dataset_path):
        dataset.get_mutable_local_copy(dataset_path)

    # Get first category
    first_category = os.listdir(dataset_path)[0]

    # Get image size from the first image from the healthy directory
    first_image_file = os.listdir(f"{dataset_path}/{first_category}")[0]
    img = plt.imread(f"{dataset_path}/{first_category}/{first_image_file}")
    img_height, img_width, _ = img.shape
    img_size = min(img_height, img_width)

    # Set batch size
    batch_size = 32

    # Data augmentation and preprocessing
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        # preprocessing_function=preprocess_input,
        rotation_range=45,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.25,
        shear_range=0.2,
        brightness_range=[0.2, 1.0],
        validation_split=0.2,
    )

    train_generator = datagen.flow_from_directory(dataset_path, target_size=(img_size, img_size), batch_size=batch_size, class_mode="categorical", shuffle=True, seed=42, subset="training")
    test_generator = datagen.flow_from_directory(dataset_path, target_size=(img_size, img_size), batch_size=batch_size, class_mode="categorical", shuffle=True, seed=42, subset="validation")

    # Configurations
    epochs = 200
    num_classes = len(train_generator.class_indices)
    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    early_stopping = EarlyStopping(monitor="val_accuracy", patience=10, min_delta=0.001, restore_best_weights=True)
    learning_rate_reduction = ReduceLROnPlateau(monitor="val_accuracy", patience=3, verbose=1, factor=0.75, min_lr=0.00001)

    # Load ResNet50V2 model
    base_resNet_model = ResNet50V2(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))

    # Freeze the base model
    for layer in base_resNet_model.layers:
        layer.trainable = False

    # Add custom layers
    x = base_resNet_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    # Create model
    cropspot_model = Model(inputs=base_resNet_model.input, outputs=predictions)

    # Compile model
    cropspot_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    # Manual logging within model.fit() callback
    logger = task.get_logger()
    clearml_log_callbacks = [
        LambdaCallback(
            on_epoch_end=lambda epoch, logs: [
                logger.report_scalar("loss", "train", iteration=epoch, value=logs["loss"]),
                logger.report_scalar("accuracy", "train", iteration=epoch, value=logs["accuracy"]),
                logger.report_scalar("val_loss", "validation", iteration=epoch, value=logs["val_loss"]),
                logger.report_scalar(
                    "val_accuracy",
                    "validation",
                    iteration=epoch,
                    value=logs["val_accuracy"],
                ),
            ]
        )
    ]

    # Train the model
    train_history = cropspot_model.fit(
        train_generator,
        epochs=epochs,
        validation_data=test_generator,
        callbacks=[learning_rate_reduction, early_stopping, clearml_log_callbacks],
    )

    # Save and upload the model to ClearML
    model_dir = "Trained Models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_file_name = "CropSpot_Model.h5"
    cropspot_model.save(model_dir + "/" + model_file_name)

    model_history_file_name = "CropSpot_Model_History.pkl"
    with open(model_dir + "/" + model_history_file_name, "wb") as file:
        pickle.dump(train_history.history, file)

    output_model = OutputModel(task=task, name="CropSpot_Model", framework="Tensorflow")

    # Upload the model weights to ClearML
    output_model.update_weights(model_file_name, upload_uri="https://files.clear.ml")

    # Make sure the model is accessible
    output_model.publish()

    task.upload_artifact("Trained Model", artifact_object=model_file_name)
    task.upload_artifact("Trained Model History", artifact_object=model_history_file_name)

    return output_model.id


if __name__ == "__main__":
    import argparse

    # Create the parser
    parser = argparse.ArgumentParser(description="Train CropSpot's PyTorch model on AWS SageMaker with ClearML")

    # Add arguments
    parser.add_argument("--dataset_name", type=str, required=False, default="TomatoDiseaseDataset", help="Name of the preprocessed dataset")
    parser.add_argument("--project_name", type=str, required=False, default="CropSpot", help="Name of the ClearML project")
    parser.add_argument("--queue_name", type=str, required=False, default="helldiver", help="Name of the ClearML queue for remote execution")

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the parsed arguments
    model_id = train_model(args.dataset_name, args.project_name, args.queue_name)

    print(f"Model trained with ID: {model_id}")
