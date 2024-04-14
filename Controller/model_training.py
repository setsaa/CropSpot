import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from clearml import Task, Dataset, OutputModel
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Activation, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def train_model(train_dataset_id, test_dataset_id, project_name, queue_name):
    import argparse
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    from clearml import Task, Dataset, OutputModel
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications import ResNet50V2
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Activation, Dropout
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    task = Task.init(project_name=project_name, task_name="train_model")
    task.execute_remotely(queue_name=queue_name, exit_process=True)

    # Load the training dataset
    train_dataset = Dataset.get(dataset_id=train_dataset_id)
    train_dataset_path = train_dataset.get_local_copy()

    # Load the testing dataset
    test_dataset = Dataset.get(dataset_id=test_dataset_id)
    test_dataset_path = test_dataset.get_local_copy()

    # Get image size from the first image from the healthy directory
    first_image_file = os.listdir(f"{train_dataset_path}/healthy")[0]
    img = plt.imread(f"{train_dataset_path}/healthy/{first_image_file}")
    img_height, img_width, _ = img.shape
    img_size = min(img_height, img_width)

    # Set batch size
    batch_size = 32

    # Data augmentation and preprocessing
    train_datagen = ImageDataGenerator(rescale=1.0 / 255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True, vertical_flip=True, zoom_range=0.2, shear_range=0.2, brightness_range=[0.2, 1.0])
    test_datagen = ImageDataGenerator(rescale=1.0 / 255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True, vertical_flip=True, zoom_range=0.2, shear_range=0.2, brightness_range=[0.2, 1.0])

    train_generator = train_datagen.flow_from_directory(train_dataset_path, target_size=(img_size, img_size), batch_size=batch_size, class_mode="categorical", shuffle=True, seed=42)
    test_generator = test_datagen.flow_from_directory(test_dataset_path, target_size=(img_size, img_size), batch_size=batch_size, class_mode="categorical", shuffle=True, seed=42)

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

    task.upload_artifact("trained_model", artifact_object=model_file_name)

    if os.path.exists("CropSpot_Model.h5"):
        os.remove("CropSpot_Model.h5")

    return output_model.id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CropSpot Model")
    parser.add_argument("--train_dataset_id", type=str, required=True, help="ClearML training dataset id")
    parser.add_argument("--test_dataset_id", type=str, required=True, help="ClearML testing dataset id")
    parser.add_argument("--project_name", type=str, required=True, help="ClearML project name")
    parser.add_argument("--queue_name", type=str, required=True, help="ClearML queue name")
    args = parser.parse_args()

    model_id = train_model(args.train_dataset_id, args.test_dataset_id, args.project_name)

    print(f"Model ID: {model_id}")
