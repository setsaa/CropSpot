def custom_cnn_train(dataset_name, project_name, queue_name):
    """
    Train the model using a custom CNN architecture with preprocessed dataset.

    Args:
        dataset_name (str): Name of the preprocessed dataset
        project_name (str): Name of the ClearML project
        queue_name (str): Name of the ClearML queue for remote execution

    Returns:
        ID of the trained model
    """
    import os
    import pickle
    import matplotlib.pyplot as plt
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LambdaCallback
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from clearml import Task, Dataset, OutputModel

    task = Task.init(project_name=project_name, task_name="Custom CNN Model Training", task_type=Task.TaskTypes.training)
    task.execute_remotely(queue_name=queue_name)

    os.system("pip install -r requirements.txt")

    trained_model_dir = "Trained Models"

    dataset = Dataset.get(dataset_name=dataset_name + "_preprocessed")
    dataset_path = "Dataset/Preprocessed"
    dataset.get_mutable_local_copy(dataset_path)

    first_category = os.listdir(dataset_path)[0]
    first_image_file = os.listdir(f"{dataset_path}/{first_category}")[0]
    img = plt.imread(f"{dataset_path}/{first_category}/{first_image_file}")
    img_height, img_width, _ = img.shape
    img_size = min(img_height, img_width)

    batch_size = 64
    datagen = ImageDataGenerator(
        rescale=1.0 / 255, rotation_range=45, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True, vertical_flip=True, zoom_range=0.25, shear_range=0.2, brightness_range=[0.2, 1.0], validation_split=0.2
    )

    train_generator = datagen.flow_from_directory(dataset_path, target_size=(img_size, img_size), batch_size=batch_size, class_mode="categorical", shuffle=True, seed=42, subset="training")
    test_generator = datagen.flow_from_directory(dataset_path, target_size=(img_size, img_size), batch_size=batch_size, class_mode="categorical", shuffle=True, seed=42, subset="validation")

    epochs = 200
    num_classes = len(train_generator.class_indices)
    optimizer = Adam(learning_rate=0.001)

    inputs = Input(shape=(img_size, img_size, 3))
    x = Conv2D(32, (3, 3), activation="relu")(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, (3, 3), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, (3, 3), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    cnn_model = Model(inputs=inputs, outputs=predictions)
    cnn_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

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

    train_history = cnn_model.fit(train_generator, epochs=epochs, validation_data=test_generator, callbacks=[ReduceLROnPlateau(), EarlyStopping(), clearml_log_callbacks])

    if not os.path.exists(trained_model_dir):
        os.makedirs(trained_model_dir)
    cnn_model.save(os.path.join(trained_model_dir, "cropspot_CNN_model.h5"))
    with open(os.path.join(trained_model_dir, "cropspot_CNN_model_History.pkl"), "wb") as file:
        pickle.dump(train_history.history, file)

    output_model = OutputModel(task=task, name="cropspot_CNN_model", framework="Tensorflow")
    output_model.update_weights(os.path.join(trained_model_dir, "cropspot_CNN_model.h5"))
    output_model.publish()
    task.upload_artifact("Trained Model", artifact_object="cropspot_CNN_model.h5")
    task.upload_artifact("Trained Model History", artifact_object="cropspot_CNN_model_History.pkl")

    return output_model.id


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a custom CNN model with ClearML on the preprocessed dataset")
    parser.add_argument("--dataset_name", type=str, required=False, default="YourDatasetName", help="Name of the preprocessed dataset")
    parser.add_argument("--project_name", type=str, required=False, default="YourProjectName", help="Name of the ClearML project")
    parser.add_argument("--queue_name", type=str, required=False, default="YourQueueName", help="Name of the ClearML queue for remote execution")

    args = parser.parse_args()

    model_id = custom_cnn_train(args.dataset_name, args.project_name, args.queue_name)
    print(f"Model trained with ID: {model_id}")
