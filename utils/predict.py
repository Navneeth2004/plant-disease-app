import numpy as np
import tensorflow as tf
from utils.preprocess import preprocess_image

# 🔥 FIX: load correct model file
model = tf.keras.models.load_model("model/best_model.h5")

# Load class mapping
class_indices = np.load("model/class_names.npy", allow_pickle=True).item()

# 🔥 IMPORTANT: keep correct class order
class_names = list(class_indices.keys())

def predict(image):
    image = image.convert("RGB")
    img = preprocess_image(image)

    preds = model.predict(img)[0]

    # Top 1 prediction
    idx = np.argmax(preds)
    confidence = float(preds[idx])

    # Top 2 predictions
    top2_idx = preds.argsort()[-2:][::-1]

    return class_names[idx], confidence, preds, top2_idx
