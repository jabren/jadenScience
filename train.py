import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------

# Standard image size for most ML models
img_size = (224, 224)

# Number of samples to pass through the network at once
batch_size = 32

# -----------------------------------------------------------
# LOAD DATASET
# -----------------------------------------------------------
# TensorFlow will read images directly from folders
# Folder structure:
# data/train/healthy
# data/train/cancer
# data/val/healthy
# data/val/cancer

train_ds = keras.preprocessing.image_dataset_from_directory(
    "data/train",
    image_size=img_size,
    batch_size=batch_size
)

val_ds = keras.preprocessing.image_dataset_from_directory(
    "data/val",
    image_size=img_size,
    batch_size=batch_size
)

# Get the class names in alphabetical order
# Example: ["cancer", "healthy"]
class_names = train_ds.class_names
print("Classes found:", class_names)

# Speed up training with prefetching
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# -----------------------------------------------------------
# DATA AUGMENTATION (helps prevent overfitting)
# -----------------------------------------------------------
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

# -----------------------------------------------------------
# LOAD A PRETRAINED BASE MODEL (TRANSFER LEARNING)
# -----------------------------------------------------------
# MobileNetV2 is light, fast, and excellent for science projects
base_model = keras.applications.MobileNetV2(
    include_top=False,              # remove original classifier
    input_shape=img_size + (3,),    # 224x224 RGB images
    pooling="avg",                  # create 1D vector
    weights="imagenet"              # pretrained weights
)

# Freeze the base model for the first stage of training
base_model.trainable = False

# -----------------------------------------------------------
# BUILD FULL MODEL
# -----------------------------------------------------------
inputs = keras.Input(shape=img_size + (3,))

# Apply data augmentation
x = data_augmentation(inputs)

# Apply MobileNetV2 preprocessing
x = keras.applications.mobilenet_v2.preprocess_input(x)

# Pass through base model
x = base_model(x, training=False)

# Dropout helps avoid overfitting
x = layers.Dropout(0.2)(x)

# Final layer → 2 classes: "cancer" and "healthy"
outputs = layers.Dense(2, activation="softmax")(x)

# Put together into a model
model = keras.Model(inputs, outputs)

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------------------------------------
# TRAIN MODEL (PHASE 1)
# -----------------------------------------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

# -----------------------------------------------------------
# OPTIONAL: FINE-TUNE THE MODEL
# -----------------------------------------------------------
# Unfreeze base model for fine-tuning
base_model.trainable = True

# Use a very small learning rate
model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

# -----------------------------------------------------------
# SAVE TRAINED MODEL
# -----------------------------------------------------------
model.save("model/plant_cancer_yesno.keras")
print("Model saved to model/plant_cancer_yesno.keras")


# -----------------------------------------------------------
# PLOT ACCURACY GRAPHS
# -----------------------------------------------------------
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.plot(history_fine.history["accuracy"], label="Fine-Tune Train Acc")
plt.plot(history_fine.history["val_accuracy"], label="Fine-Tune Val Acc")
plt.legend()
plt.title("Model Accuracy Over Time")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig("model/accuracy_graph.png")
plt.show()
