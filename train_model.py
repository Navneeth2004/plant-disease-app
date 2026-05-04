import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, Model
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

# =========================
# CONFIG
# =========================
data_dir = "PlantVillage"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20

# =========================
# CHECK STRUCTURE
# =========================
print("\n📊 Class Distribution:")
for cls in os.listdir(data_dir):
    path = os.path.join(data_dir, cls)
    if os.path.isdir(path):
        print(f"{cls} → {len(os.listdir(path))} images")

# =========================
# DATA AUGMENTATION (IMPROVED)
# =========================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.6, 1.4],   # 🔥 improved
    shear_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
)

train = datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset="training",
    class_mode="categorical"
)

val = datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset="validation",
    class_mode="categorical"
)

# =========================
# SAVE CLASS MAPPING
# =========================
os.makedirs("model", exist_ok=True)
np.save("model/class_names.npy", train.class_indices)
print("\n📌 Class mapping:", train.class_indices)

# =========================
# CLASS WEIGHTS
# =========================
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train.classes),
    y=train.classes
)
class_weights = dict(enumerate(class_weights))

# =========================
# MODEL (FUNCTIONAL API)
# =========================
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze most layers
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Functional head
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)   # 🔥 increased dropout
output = layers.Dense(train.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# =========================
# COMPILE
# =========================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

# =========================
# CALLBACKS (IMPROVED)
# =========================
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.3),
    tf.keras.callbacks.ModelCheckpoint(
        "model/best_model.h5",
        monitor="val_accuracy",
        save_best_only=True
    )
]

# =========================
# TRAIN
# =========================
history = model.fit(
    train,
    validation_data=val,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# =========================
# SAVE MODEL (IMPORTANT CHANGE)
# =========================
model.save("model/model.h5")   # 🔥 CHANGED from .keras

print("\n✅ Training complete!")
