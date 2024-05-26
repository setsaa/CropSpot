def densenet_train(dataset_name, project_name):
    """
    Train the model using DenseNet architecture with preprocessed dataset.

    Args:
        dataset_name (str): Name of the preprocessed dataset
        project_name (str): Name of the ClearML project

    Returns:
        ID of the trained model
    """
    import os
    from clearml import Task, Dataset, OutputModel, InputModel
    from keras.models import Model
    from keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Activation, Dropout
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau, LambdaCallback
    from keras.optimizers import Adam, RMSprop, SGD
    from keras_tuner import HyperModel, HyperParameters
    from keras_tuner.tuners import Hyperband
    from keras.applications import DenseNet121
    from keras.preprocessing.image import ImageDataGenerator

    task = Task.init(project_name=project_name, task_name="DenseNet Train Model")

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
        rescale=1.0 / 255,
        validation_split=0.2,
    )

    train_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=42,
        subset="training"
    )
    test_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=42,
        subset="validation"
    )

    num_classes = len(train_generator.class_indices)

    class DenseNetHyperModel(HyperModel):
        def __init__(self, input_shape, num_classes):
            self.input_shape = input_shape
            self.num_classes = num_classes

        def build(self, hp):
            base_densenet_model = DenseNet121(weights="imagenet", include_top=False, input_shape=self.input_shape)

            # Freeze the base model
            for layer in base_densenet_model.layers:
                layer.trainable = False

            x = base_densenet_model.output
            x = GlobalAveragePooling2D()(x)

            # Hyperparameters for the fully connected layers
            x = Dense(units=hp.Int("units_1", min_value=128, max_value=1024, step=128))(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            x = Dropout(rate=hp.Float("dropout_1", min_value=0.0, max_value=0.5, step=0.1))(x)

            x = Dense(units=hp.Int("units_2", min_value=128, max_value=1024, step=128))(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            x = Dropout(rate=hp.Float("dropout_2", min_value=0.0, max_value=0.5, step=0.1))(x)

            x = Dense(units=hp.Int("units_3", min_value=128, max_value=1024, step=128))(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            x = Dropout(rate=hp.Float("dropout_3", min_value=0.0, max_value=0.5, step=0.1))(x)

            predictions = Dense(self.num_classes, activation="softmax")(x)

            model = Model(inputs=base_densenet_model.input, outputs=predictions)

            # Hyperparameter: Optimizer selection
            optimizer_name = hp.Choice("optimizer", ["adam", "rmsprop", "sgd"])
            learning_rate = hp.Float("learning_rate", min_value=1e-5, max_value=1e-2, sampling="LOG")

            if optimizer_name == "adam":
                optimizer = Adam(learning_rate=learning_rate)
            elif optimizer_name == "rmsprop":
                optimizer = RMSprop(learning_rate=learning_rate)
            elif optimizer_name == "sgd":
                optimizer = SGD(learning_rate=learning_rate)
            else:
                raise Exception(f"Illegal optimizer name given: {optimizer_name}")

            model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

            return model

    hypermodel = DenseNetHyperModel(input_shape=(img_size, img_size, 3), num_classes=num_classes)

    tuner = Hyperband(
        hypermodel,
        objective="val_accuracy",
        max_epochs=10,
        factor=3,
        hyperband_iterations=1,
        directory=f"densenet_keras_tuner",
        project_name=f"densenet_tuning"
    )

    # Search for the best hyperparameters
    tuner.search_space_summary()

    logger = task.get_logger()

    tuner.search(train_generator, epochs=10, validation_data=test_generator)

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Best hyperparameters: {best_hps.values}")

    # Build the model with the best hyperparameters and train it on the data for 50 epochs
    densenet_model = tuner.hypermodel.build(best_hps)

    epochs = 60
    densenet_model.fit(
        train_generator,
        epochs=epochs,
        validation_data=test_generator,
        callbacks=[
            EarlyStopping(monitor="val_accuracy", patience=10, min_delta=0.001, restore_best_weights=True),
            LambdaCallback(
                on_epoch_end=lambda epoch, logs: [
                    logger.report_scalar("loss", "train", iteration=epoch, value=logs["loss"]),
                    logger.report_scalar("accuracy", "train", iteration=epoch, value=logs["accuracy"]),
                    logger.report_scalar("val_loss", "validation", iteration=epoch, value=logs["val_loss"]),
                    logger.report_scalar("val_accuracy", "validation", iteration=epoch, value=logs["val_accuracy"]),
                ]
            ),
        ],
    )

    trained_model_dir = "Trained Models"
    if not os.path.exists(trained_model_dir):
        os.makedirs(trained_model_dir)
    densenet_model.save(os.path.join(trained_model_dir, "cropspot_densenet_model.h5"))

    output_model = OutputModel(task=task, name="cropspot_densenet_model", framework="Tensorflow")
    output_model.update_weights(
        os.path.join(trained_model_dir, "cropspot_densenet_model.h5"),
        upload_uri="https://files.clear.ml",
        auto_delete_file=False
    )

    task.upload_artifact("DenseNet Model", artifact_object="cropspot_densenet_model.h5")

    output_model.publish()

    return output_model.id
