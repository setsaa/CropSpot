def resnet_train(dataset_name, project_name, queue_name):
    """
    Train the CropSpot model using the preprocessed dataset.

    Args:
        preprocessed_dataset_name (str): Name of the preprocessed dataset
        project_name (str): Name of the ClearML project
        queue_name (str): Name of the ClearML queue for remote execution

    Returns:
        ID of the trained model
    """
    from clearml import Task, Dataset, OutputModel

    task = Task.create(project_name=project_name, task_name="ResNet Model Training", task_type=Task.TaskTypes.training)
    # task.execute_remotely(queue_name=queue_name, exit_process=True)

    import os
    import matplotlib.pyplot as plt
    from keras.models import Model, load_model
    from keras.callbacks import LambdaCallback
    from keras.preprocessing.image import ImageDataGenerator
    from keras.applications import ResNet50V2
    from keras.applications.resnet_v2 import preprocess_input
    from keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Activation, Dropout
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    import pickle

    model_file_name = "cropspot_resnet_model.h5"
    model_history_file_name = "cropspot_resnet_model_History.pkl"

    # Load preprocessed dataset
    prep_dataset_name = dataset_name + "_preprocessed"
    dataset = Dataset.get(dataset_name=prep_dataset_name)

    # Check if the dataset is already downloaded. If not, download it. Otherwise, use the existing dataset.
    dataset_path = f"Dataset/{prep_dataset_name}"
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
    batch_size = 64

    # Data augmentation and preprocessing
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        # rotation_range=45,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        # horizontal_flip=True,
        # vertical_flip=True,
        # zoom_range=0.25,
        # shear_range=0.2,
        # brightness_range=[0.2, 1.0],
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
    resnet_model = Model(inputs=base_resNet_model.input, outputs=predictions)

    # Compile model
    resnet_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

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
    resnet_model.fit(
        train_generator,
        epochs=epochs,
        validation_data=test_generator,
        callbacks=[learning_rate_reduction, early_stopping, clearml_log_callbacks],
    )

    trained_model_dir = "Trained Models"

    # Save and upload the model to ClearML
    if not os.path.exists(trained_model_dir):
        os.makedirs(trained_model_dir)

    model_file_name = "cropspot_resnet_model.h5"
    resnet_model.save(trained_model_dir + "/" + model_file_name)

    output_model = OutputModel(task=task, name="cropspot_resnet_model", framework="Tensorflow")

    # Upload the model weights to ClearML
    output_model.update_weights("Trained Models\cropspot_resnet_model.h5", upload_uri="https://files.clear.ml", auto_delete_file=False)

    # Make sure the model is accessible
    output_model.publish()

    task.upload_artifact("ResNet Model", artifact_object=model_file_name)

    return output_model.id


if __name__ == "__main__":
    import argparse

    # Create the parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--dataset_name", type=str, required=False, default="TomatoDiseaseDatasetV2", help="Name of the preprocessed dataset")
    parser.add_argument("--project_name", type=str, required=False, default="CropSpot", help="Name of the ClearML project")
    parser.add_argument("--queue_name", type=str, required=False, default="helldiver", help="Name of the ClearML queue for remote execution")

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the parsed arguments
    model_id = resnet_train(args.dataset_name, args.project_name, args.queue_name)

    print(f"Model trained with ID: {model_id}")
