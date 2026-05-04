import numpy as np
import tensorflow as tf
import os
from utils.preprocess import preprocess_image

# -------------------------
# LOAD MODEL (SAFE + COMPATIBLE)
# -------------------------
MODEL_PATH = "model/best_model.h5"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found at {MODEL_PATH}")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False  # 🔥 prevents version conflicts
)

# -------------------------
# LOAD CLASS MAPPING
# -------------------------
CLASS_PATH = "model/class_names.npy"

if not os.path.exists(CLASS_PATH):
    raise FileNotFoundError(f"❌ Class names file not found at {CLASS_PATH}")

class_indices = np.load(CLASS_PATH, allow_pickle=True).item()

# Convert to ordered class list
class_names = list(class_indices.keys())

# -------------------------
# PREDICTION FUNCTION
# -------------------------
def predict(image):
    # Ensure RGB
    image = image.convert("RGB")

    # Preprocess
    img = preprocess_image(image)

    # Predict
    preds = model.predict(img)[0]

    # Top prediction
    idx = np.argmax(preds)
    confidence = float(preds[idx])

    # Top 2 predictions
    top2_idx = preds.argsort()[-2:][::-1]

    return class_names[idx], confidence, preds, top2_idx
