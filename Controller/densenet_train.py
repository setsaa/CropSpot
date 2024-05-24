def densenet_train(dataset_name, project_name):
    """
    Train the model using DenseNet architecture with preprocessed dataset.

    Args:
        dataset_name (str): Name of the preprocessed dataset
        project_name (str): Name of the ClearML project

    Returns:
        ID of the trained model
    """
    from clearml import Task, Dataset, OutputModel, InputModel

    task = Task.init(project_name=project_name, task_name="DenseNet Train Model")

    import os
    import matplotlib.pyplot as plt
    from keras.models import Model
    from keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Activation, Dropout
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau, LambdaCallback
    from keras.regularizers import L2
    from keras.applications import DenseNet121
    from keras.applications.densenet import preprocess_input
    from keras.preprocessing.image import ImageDataGenerator

    # # TEMP
    # model_file_name = "cropspot_densenet_model.h5"
    # existing_model = InputModel(name=model_file_name[:-3], project=project_name, only_published=True)
    # existing_model.connect(task=task)
    # if existing_model:
    #     print(f"Model '{model_file_name}' already exists in project '{project_name}'.")
    #     return existing_model.id

    # Load preprocessed dataset
    prep_dataset_name = dataset_name
    dataset = Dataset.get(dataset_name=prep_dataset_name)

    # Check if the dataset is already downloaded. If not, download it. Otherwise, use the existing dataset.
    dataset_path = f"Dataset/{prep_dataset_name}"
    if not os.path.exists(dataset_path):
        dataset.get_mutable_local_copy(dataset_path)

    img_size = 224

    batch_size = 64

    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2,
    )

    train_generator = datagen.flow_from_directory(dataset_path, target_size=(img_size, img_size), batch_size=batch_size, class_mode="categorical", shuffle=True, seed=42, subset="training")
    test_generator = datagen.flow_from_directory(dataset_path, target_size=(img_size, img_size), batch_size=batch_size, class_mode="categorical", shuffle=True, seed=42, subset="validation")

    epochs = 200
    num_classes = len(train_generator.class_indices)
    optimizer = Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999)
    early_stopping = EarlyStopping(monitor="val_accuracy", patience=25, min_delta=0.001, restore_best_weights=True)
    learning_rate_reduction = ReduceLROnPlateau(monitor="val_accuracy", patience=6, verbose=1, factor=0.5, min_lr=0.00001)

    base_densenet_model = DenseNet121(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))

    for layer in base_densenet_model.layers:
        layer.trainable = False

    x = base_densenet_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, kernel_regularizer=L2(0.01))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.2)(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    densenet_model = Model(inputs=base_densenet_model.input, outputs=predictions)
    densenet_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    # Manual logging within model.fit() callback
    logger = task.get_logger()
    clearml_log_callbacks = [
        LambdaCallback(
            on_epoch_end=lambda epoch, logs: [
                logger.report_scalar("loss", "train", iteration=epoch, value=logs["loss"]),
                logger.report_scalar("accuracy", "train", iteration=epoch, value=logs["accuracy"]),
                logger.report_scalar("val_loss", "validation", iteration=epoch, value=logs["val_loss"]),
                logger.report_scalar("val_accuracy", "validation", iteration=epoch, value=logs["val_accuracy"]),
            ]
        )
    ]

    densenet_model.fit(
        train_generator,
        epochs=epochs,
        validation_data=test_generator,
        callbacks=[clearml_log_callbacks, early_stopping, learning_rate_reduction],
    )

    trained_model_dir = "Trained Models"
    if not os.path.exists(trained_model_dir):
        os.makedirs(trained_model_dir)
    densenet_model.save(os.path.join(trained_model_dir, "cropspot_densenet_model.h5"))

    output_model = OutputModel(task=task, name="cropspot_densenet_model", framework="Tensorflow")
    output_model.update_weights(os.path.join(trained_model_dir, "cropspot_densenet_model.h5"), upload_uri="https://files.clear.ml", auto_delete_file=False)

    task.upload_artifact("DenseNet Model", artifact_object="cropspot_densenet_model.h5")

    output_model.publish()

    return output_model.id


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a DenseNet model with ClearML on the preprocessed dataset")
    parser.add_argument("--dataset_name", type=str, required=False, default="YourDatasetName", help="Name of the preprocessed dataset")
    parser.add_argument("--project_name", type=str, required=False, default="YourProjectName", help="Name of the ClearML project")

    args = parser.parse_args()

    model_id = densenet_train(args.dataset_name, args.project_name)

    print(f"Model trained with ID: {model_id}")
