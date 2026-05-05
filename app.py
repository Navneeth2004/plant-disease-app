import streamlit as st
from PIL import Image
from utils.predict import predict, class_names
from utils.disease_info import disease_info
import matplotlib.pyplot as plt
import os

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Plant Intelligence Dashboard",
    layout="wide",
    page_icon="🌿"
)

# -------------------------
# LOAD CSS
# -------------------------
def load_css():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    css_path = os.path.join(base_dir, "assets", "style.css")

    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# -------------------------
# HEADER
# -------------------------
st.title("🌿 Plant Intelligence Dashboard")
st.caption("AI-powered crop disease detection with insights & analytics")

# -------------------------
# FILE UPLOAD
# -------------------------
uploaded_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        try:
            prediction, confidence, preds, top2_idx = predict(image)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

        display_label = prediction.split("___")[-1].replace("_", " ").title()
        clean_label = prediction.split("___")[-1].replace("_", " ").lower()

        info = disease_info.get(clean_label, {
            "cause": "Unknown",
            "solution": "N/A",
            "risk": "Unknown"
        })

        st.subheader("🧠 Prediction Result")
        st.success(display_label)

        st.metric("Confidence", f"{confidence*100:.2f}%")

        # Risk
        risk = info.get("risk", "unknown").lower()
        risk_map = {
            "low": "🟢 Low",
            "medium": "🟡 Medium",
            "high": "🔴 High"
        }
        st.markdown(f"**Risk Level:** {risk_map.get(risk, 'N/A')}")

        # Info
        st.markdown("### 🌿 Disease Insights")
        st.info(f"**Cause:** {info.get('cause')}")
        st.info(f"**Solution:** {info.get('solution')}")

        # Chart
        st.markdown("### 📊 Model Confidence Distribution")

        labels = [name.split("___")[-1].replace("_", " ") for name in class_names]

        fig, ax = plt.subplots()
        ax.bar(labels, preds[:len(labels)])
        ax.set_ylim([0, 1])
        st.pyplot(fig)
        plt.close(fig)

        # Top predictions
        st.markdown("### 🔍 Top Predictions")
        for i in top2_idx:
            st.write(f"{labels[i]} : {preds[i]*100:.2f}%")

        # Smart insights
        st.markdown("### 💡 Smart Insights")

        if "late" in clean_label:
            st.warning("Severe infection detected. Immediate action required.")
        elif "early" in clean_label:
            st.warning("Moderate infection. Monitor closely and treat early.")
        else:
            st.success("Plant is healthy. Maintain current care.")
