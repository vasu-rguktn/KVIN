import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.losses import BinaryFocalCrossentropy

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# -----------------------------
# Configuration
# -----------------------------
IMG_SIZE = 224
BATCH_SIZE = 32
FROZEN_EPOCHS = 20
TOTAL_EPOCHS = 50

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.03),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.15),
        tf.keras.layers.RandomBrightness(0.1),
    ]
)

# -----------------------------
# Load dataset
# -----------------------------
def load_path(path, part):
    """
    Load X-ray dataset for a specific body part.
    Returns a DataFrame with columns: ['body_part', 'patient_id', 'label', 'image_path']
    """
    dataset = []
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if not os.path.isdir(folder_path):
            continue

        for body in os.listdir(folder_path):
            if body != part:
                continue

            body_path = os.path.join(folder_path, body)
            for patient_id in os.listdir(body_path):
                patient_path = os.path.join(body_path, patient_id)
                for lab in os.listdir(patient_path):
                    label = (
                        "fractured" if lab.split("_")[-1] == "positive" else "normal"
                    )
                    lab_path = os.path.join(patient_path, lab)
                    for img_file in os.listdir(lab_path):
                        dataset.append(
                            {
                                "body_part": body,
                                "patient_id": patient_id,
                                "label": label,
                                "image_path": os.path.join(lab_path, img_file),
                            }
                        )
    return pd.DataFrame(dataset)


# -----------------------------
# Image preprocessing
# -----------------------------
def parse_image(filename, label, training=True):
    

    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.keras.applications.resnet50.preprocess_input(image)
    # label = tf.one_hot(label, NUM_CLASSES)

    # if training:
    #     image = data_augmentation(image)
    label = tf.cast(label, tf.float32)
    label = tf.expand_dims(label, axis=-1)
    return image, label


def build_dataset(df, training=True, cache_name=None):

    ds = tf.data.Dataset.from_tensor_slices(
        (df["image_path"].values, df["label"].values)
    )

    ds = ds.map(lambda x, y: parse_image(x, y, training=training), num_parallel_calls=tf.data.AUTOTUNE,)



    if training:
        ds = ds.shuffle(1000)

    if cache_name is not None:
        ds = ds.cache(cache_name)
    ds = ds.batch(
        BATCH_SIZE,
    )
    ds = ds.prefetch(tf.data.AUTOTUNE)

    if training:
        ds = ds.repeat()

    return ds


# -----------------------------
# Training function
# -----------------------------
def train_part(part_name):
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(THIS_FOLDER, "Dataset")
    df = load_path(dataset_path, part_name)
        
    df["label"] = df["label"].map({"normal": 0, "fractured": 1})
    # Train/Validation/Test split
    train_df, test_df = train_test_split(
        df,
        train_size=0.85,
        stratify=df["label"],
        random_state=42,
        shuffle=True,
    )
    train_df, val_df = train_test_split(
        train_df,
        train_size=0.85,
        stratify=train_df["label"],
        random_state=42,
        shuffle=True,
    )


    # Datasets
    train_ds = build_dataset(train_df, training=True,  cache_name="cache_fracture_train_unfreeze_no_aug.tf")
    val_ds = build_dataset(val_df, training=False, cache_name="cache_fracture_val_unfreeze_no_aug.tf")
    test_ds = build_dataset(test_df, training=False, cache_name="cache_fracture_test_unfreeze_no_aug.tf")

    # train_ds = build_dataset(
    #     train_df,
    #     training=True,
    #     part_name=part_name,
    # )

    # val_ds = build_dataset(
    #     val_df,
    #     training=False,
    #     part_name=part_name,
    # )
    # test_ds = build_dataset(
    #     test_df,
    #     training=False,
    #     part_name=part_name,
    # )

    # Compute class weights
    # class_weights_array = compute_class_weight(
    #     "balanced",
    #     classes=np.unique(train_df["label"]),
    #     y=train_df["label"],
    # )
    # class_weights = dict(enumerate(class_weights_array))

    steps_per_epoch = len(train_df) // BATCH_SIZE
    validation_steps = len(val_df) // BATCH_SIZE

    counts = np.bincount(train_df["label"])
    total = counts.sum()
    # class_weights = {i: total/(len(counts)*c) for i, c in enumerate(counts)}
    class_weights = {}
    for i, c in enumerate(counts):
        if c > 0:
            class_weights[i] = total / (len(counts) * c)
        else:
            class_weights[i] = 1.0


    # -----------------------------
    # Build Model
    # -----------------------------
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )

    # Freeze most layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer=Adam(1e-4),
        # loss="categorical_crossentropy",
        
        loss = BinaryFocalCrossentropy(gamma=2.0, alpha=0.25),
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    # -----------------------------
    # Callbacks
    # -----------------------------
    weight_dir = os.path.join(THIS_FOLDER, "weights")
    os.makedirs(weight_dir, exist_ok=True)
    plot_dir = os.path.join(THIS_FOLDER, "plots", "FractureDetection", part_name)
    os.makedirs(plot_dir, exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-6,
        ),
        ModelCheckpoint(
            os.path.join(weight_dir, f"ResNet50_{part_name}_frac.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    # -----------------------------
    # Train
    # -----------------------------
    history_1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FROZEN_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    for layer in base_model.layers[-40:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    model.compile(
        optimizer=Adam(1e-5),
        # loss="categorical_crossentropy",
        
        loss = BinaryFocalCrossentropy(gamma=2.0, alpha=0.25),
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    history_2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=TOTAL_EPOCHS,
        initial_epoch=FROZEN_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        class_weight=class_weights,
    )
    

    # -----------------------------
    # Evaluate
    # -----------------------------
    model.save(os.path.join(THIS_FOLDER, "weights", f"ResNet50_{part_name}_frac_unfreeze_no_aug.keras"))
    results = model.evaluate(test_ds, verbose=1)
    print(f"\n{part_name} Test Results: {results}")
    print(f"Test Accuracy: {results[1]*100:.2f}%")


    history = history_1.history
    print(history.keys())
    print(history_2.history.keys())

    for key in history_2.history:
        history[key] = history.get(key, []) + history_2.history[key]


    # Predictions for classification report
    # y_true = np.concatenate([y for x, y in test_ds], axis=0).astype(int)
    # y_pred = (model.predict(test_ds) > 0.5).astype(int).flatten()

    y_true = []
    y_pred = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)

        y_true.extend(labels.numpy().flatten())
        y_pred.extend((preds > 0.5).astype(int).flatten())

    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)

    print(classification_report(y_true, y_pred, target_names=["normal", "fractured"]))

    # -----------------------------
    # Plot Accuracy
    # -----------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(history["accuracy"], label="Train Accuracy")
    plt.plot(history["val_accuracy"], label="Validation Accuracy")
    plt.plot(history["precision"], label="Train Precision")
    plt.plot(history["val_precision"], label="Validation Precision")
    plt.plot(history["recall"], label="Train Recall")
    plt.plot(history["val_recall"], label="Validation Recall")
    plt.plot(history["auc"], label="Train AUC")
    plt.plot(history["val_auc"], label="Validation AUC")
    plt.title(f"{part_name} Metrics")
    plt.xlabel("Epoch")
    # plt.ylabel("Accuracy")
    # plt.legend(["train", "validation"])
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "_Metrics_unfreeze_no_aug.jpeg"))
    plt.clf()

    # -----------------------------
    # Plot Loss
    # -----------------------------
    loss_1 = history_1.history.get("loss", []) if history_1 else []
    loss_2 = history_2.history.get("loss", []) if history_2 else []

    val_loss_1 = history_1.history.get("val_loss", []) if history_1 else []
    val_loss_2 = history_2.history.get("val_loss", []) if history_2 else []

    loss = loss_1 + loss_2
    val_loss = val_loss_1 + val_loss_2
    plt.figure(figsize=(10, 6))
    plt.plot(loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.title(f"{part_name} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    # plt.legend(["train", "validation"])
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "_Loss_unfreeze_no_aug.jpeg"))
    plt.clf()


# -----------------------------
# Run training for each body part
# -----------------------------
categories_parts = ["Elbow", "Hand", "Shoulder"]
for part in categories_parts:
    train_part(part)
