import numpy as np
import tensorflow as tf
from utils.preprocess import preprocess_image

# Load model safely (no compile issues)
model = tf.keras.models.load_model("model/best_model.h5", compile=False)

# Load class mapping
class_indices = np.load("model/class_names.npy", allow_pickle=True).item()

# Clean class names properly
class_names = [
    cls.split("___")[-1].replace("_", " ").lower()
    for cls in class_indices.keys()
]

def predict(image):
    image = image.convert("RGB")
    img = preprocess_image(image)

    preds = model.predict(img)[0]

    idx = np.argmax(preds)
    confidence = float(preds[idx])

    top2_idx = preds.argsort()[-2:][::-1]

    return class_names[idx], confidence, preds, top2_idx
