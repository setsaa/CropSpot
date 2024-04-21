import argparse


def train_model(preprocessed_dataset_id, split_ratio, project_name, queue_name):
    import os
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications import ResNet50V2
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Activation, Dropout
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from clearml import Task, Dataset, OutputModel

    task = Task.init(project_name=project_name, task_name="Model Training")
    task.execute_remotely(queue_name=queue_name, exit_process=True)

    # Load preprocessed dataset
    dataset = Dataset.get(dataset_id=preprocessed_dataset_id)
    dataset_path = dataset.get_local_copy()

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
        rescale=1.0 / 255, rotation_range=45, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True, vertical_flip=True, zoom_range=0.25, shear_range=0.2, brightness_range=[0.2, 1.0], validation_split=split_ratio
    )

    train_generator = datagen.flow_from_directory(dataset_path, target_size=(img_size, img_size), batch_size=batch_size, class_mode="categorical", shuffle=True, seed=42)
    test_generator = datagen.flow_from_directory(dataset_path, target_size=(img_size, img_size), batch_size=batch_size, class_mode="categorical", shuffle=True, seed=42)

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

    # Train the model
    resnet_model.fit(
        train_generator,
        epochs=epochs,
        validation_data=test_generator,
        callbacks=[learning_rate_reduction, early_stopping],
    )

    # Save and upload the model to ClearML
    model_file_name = "CropSpot_Model.h5"
    resnet_model.save(model_file_name)

    output_model = OutputModel(task=task)

    # Upload the model weights to ClearML
    output_model.update_weights(model_file_name, upload_uri="https://files.clear.ml")

    # Make sure the model is accessible
    output_model.publish()

    task.upload_artifact("Trained Model", artifact_object=model_file_name)

    if os.path.exists("CropSpot_Model.h5"):
        os.remove("CropSpot_Model.h5")

    return output_model.id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CropSpot Model")
    parser.add_argument("--preprocessed_dataset_id", type=str, required=True, help="ID of the preprocessed dataset")
    parser.add_argument("--project_name", type=str, required=True, help="ClearML project name")
    parser.add_argument("--queue_name", type=str, required=True, help="ClearML queue name")
    parser.add_argument("--split_ratio", type=float, default=0.2, help="Validation split ratio")

    args = parser.parse_args()

    print(args)

    model_id = train_model(args.preprocessed_dataset_id, args.split_ratio, args.project_name, args.queue_name)

    print(f"Model trained with ID: {model_id}")
