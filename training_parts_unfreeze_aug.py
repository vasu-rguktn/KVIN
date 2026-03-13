import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import glob

for f in glob.glob("cache_*"):
    os.remove(f)

IMG_SIZE = 224
BATCH_SIZE = 32
FROZEN_EPOCHS = 50
TOTAL_EPOCHS = 50

# load data from path
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(THIS_FOLDER, "Dataset")

plot_dir = os.path.join(
    THIS_FOLDER, "plots", "FractureDetection", "body_part_identification"
)
os.makedirs("weights", exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)


def load_path(path):
    """
    load X-ray dataset
    """
    dataset = []
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)

        if os.path.isdir(folder_path):

            for body in os.listdir(folder_path):
                path_p = os.path.join(folder_path, body)

                for patient_id in os.listdir(path_p):
                    path_id = os.path.join(path_p, patient_id)

                    for lab in os.listdir(path_id):

                        if lab.split("_")[-1] == "positive":
                            label = "fractured"
                        elif lab.split("_")[-1] == "negative":
                            label = "normal"
                        path_l = os.path.join(path_id, lab)
                        for img in os.listdir(path_l):

                            dataset.append(
                                {"label": body, "image_path": os.path.join(path_l, img)}
                            )
    return pd.DataFrame(dataset)


def parse_image(filename, label, training=True):

    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))

    if training:
        image = data_augmentation(image)

    image = tf.keras.applications.resnet50.preprocess_input(image)

    label = tf.one_hot(label, NUM_CLASSES)

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


data = load_path(image_dir)

train_df, test_df = train_test_split(
    data,
    train_size=0.85,
    stratify=data["label"],
    random_state=42,
    shuffle=True,
)

train_df, val_df = train_test_split(
    train_df, train_size=0.85, stratify=train_df["label"], random_state=42
)

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.03),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.15),
        tf.keras.layers.RandomBrightness(0.1),
    ]
)


le = LabelEncoder()
train_df = train_df.copy()
train_df["label"] = le.fit_transform(train_df["label"])
print(train_df["label"].value_counts())
val_df["label"] = le.transform(val_df["label"])
test_df["label"] = le.transform(test_df["label"])

NUM_CLASSES = len(le.classes_)

train_ds = build_dataset(train_df, training=True,  cache_name="cache_train_unfreeze_aug.tf")
val_ds = build_dataset(val_df, training=False, cache_name="cache_val_unfreeze_aug.tf")
test_ds = build_dataset(test_df, training=False, cache_name="cache_test_unfreeze_aug.tf")


base_model = tf.keras.applications.resnet50.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)

for layer in base_model.layers:
    layer.trainable = False


x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)

x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)

x = Dense(64, activation="relu")(x)
x = Dropout(0.3)(x)

outputs = Dense(NUM_CLASSES, activation="softmax")(x)

model = tf.keras.Model(inputs=base_model.input, outputs=outputs)

print(model.summary())


model.compile(
    optimizer=Adam(1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.CategoricalAccuracy(name="cat_acc"),
        tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc"),
    ],
)


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
        "weights/best_model.keras",
        monitor="val_loss",
        save_best_only=True,
    ),
]

class_weights = compute_class_weight(
    class_weight="balanced", classes=np.unique(train_df["label"]), y=train_df["label"]
)

class_weights = dict(enumerate(class_weights))

steps_per_epoch = len(train_df) // BATCH_SIZE
validation_steps = len(val_df) // BATCH_SIZE

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
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.CategoricalAccuracy(name="cat_acc"),
        tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc"),
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


model.save("weights/ResNet50_BodyParts_unfreeze_aug.keras")


# -----------------------------
# Testing the Model
# -----------------------------
results = model.evaluate(test_ds, verbose=1)
print("Test Results:", results)

history = history_1.history
print(history.keys())
print(history_2.history.keys())

for key in history_2.history:
    history[key] = history.get(key, []) + history_2.history[key]


# y_true = np.concatenate([y for x, y in test_ds], axis=0)
# y_true = np.argmax(y_true, axis=1)

# preds = model.predict(test_ds, verbose=1)
# y_pred = np.argmax(preds, axis=1)

y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)

    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print(classification_report(y_true, y_pred, target_names=le.classes_))


# confusion matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=le.classes_,
    yticklabels=le.classes_,
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Body Part Classification")

plt.tight_layout()

plt.savefig(os.path.join(plot_dir, "confusion_matrix_unfreeze_aug.jpeg"))
plt.close()

cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
cm_df.to_csv(os.path.join(plot_dir, "confusion_matrix_unfreeze_aug.csv"))


# If you run into KeyError: 'precision', print history.history.keys() to confirm the exact names

plt.figure(figsize=(10, 6))
plt.plot(history["accuracy"], label="Train Accuracy")
plt.plot(history["val_accuracy"], label="Validation Accuracy")
plt.plot(history["precision"], label="Train Precision")
plt.plot(history["val_precision"], label="Validation Precision")
plt.plot(history["recall"], label="Train Recall")
plt.plot(history["val_recall"], label="Validation Recall")
plt.plot(history["cat_acc"], label="Train Categorical Acc")
plt.plot(history["val_cat_acc"], label="Val Categorical Acc")
plt.plot(history["top3_acc"], label="Train Top-3 Acc")
plt.plot(history["val_top3_acc"], label="Val Top-3 Acc")

plt.title(f"Body Part Identification Metrics")
plt.xlabel("Epoch")
plt.ylabel("Metric Value")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "_Metrics_unfreeze_aug.jpeg"))
plt.close()


plt.figure(figsize=(10, 6))
plt.plot(history["loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Validation Loss")
plt.title(f"Body Part Identification Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
# plt.legend(["train", "validation"])
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "_Loss_unfreeze_aug.jpeg"))
plt.close()
