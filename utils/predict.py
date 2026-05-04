import numpy as np
import tensorflow as tf
from utils.preprocess import preprocess_image
import os

# =========================
# LOAD MODEL SAFELY
# =========================
MODEL_PATH = "model/final_model.h5"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# =========================
# LOAD CLASS MAPPING
# =========================
CLASS_PATH = "model/class_names.npy"

if not os.path.exists(CLASS_PATH):
    raise FileNotFoundError(f"❌ class_names.npy not found at {CLASS_PATH}")

class_indices = np.load(CLASS_PATH, allow_pickle=True).item()

# Clean class names → "Potato___Late_blight" → "late blight"
class_names = [
    cls.split("___")[-1].replace("_", " ").lower()
    for cls in class_indices.keys()
]

# =========================
# PREDICTION FUNCTION
# =========================
def predict(image):
    try:
        image = image.convert("RGB")
        img = preprocess_image(image)

        preds = model.predict(img)[0]

        # Top prediction
        idx = np.argmax(preds)
        confidence = float(preds[idx])

        # Top 2 predictions (for insights)
        top2_idx = preds.argsort()[-2:][::-1]

        return class_names[idx], confidence, preds, top2_idx

    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        return "error", 0.0, None, None
