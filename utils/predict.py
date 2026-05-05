import numpy as np
import tensorflow as tf
import os
import streamlit as st
from utils.preprocess import preprocess_image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.keras")
CLASS_PATH = os.path.join(BASE_DIR, "model", "class_names.npy")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource
def load_classes():
    class_indices = np.load(CLASS_PATH, allow_pickle=True).item()
    return [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]

model = load_model()
class_names = load_classes()

def predict(image):
    image = image.convert("RGB")
    img = preprocess_image(image)

    preds = model.predict(img)[0]

    idx = np.argmax(preds)
    confidence = float(preds[idx])
    top2_idx = preds.argsort()[-2:][::-1]

    return class_names[idx], confidence, preds, top2_idx
