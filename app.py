import streamlit as st
from PIL import Image
from utils.predict import predict
from utils.disease_info import disease_info
import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Plant Intelligence Dashboard",
    layout="wide",
    page_icon="🌿"
)

# -------------------------
# CUSTOM CSS (🔥 UI UPGRADE)
# -------------------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
    color: white;
}
h1, h2, h3 {
    color: #4CAF50;
}
.block-container {
    padding-top: 2rem;
}
.metric-container {
    background: #1E222A;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

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

    # -------------------------
    # LEFT: IMAGE
    # -------------------------
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # -------------------------
    # RIGHT: ANALYTICS PANEL
    # -------------------------
    with col2:
        prediction, confidence, preds, top2_idx = predict(image)

        # Clean labels
        display_label = prediction.split("___")[-1].replace("_", " ").title()
        clean_label = prediction.split("___")[-1].replace("_", " ").lower()

        info = disease_info.get(clean_label, {
            "cause": "Unknown",
            "solution": "N/A",
            "risk": "N/A"
        })

        # -------------------------
        # MAIN PREDICTION
        # -------------------------
        st.subheader("🧠 Prediction Result")
        st.success(display_label)

        st.metric("Confidence", f"{confidence*100:.2f}%")

        # -------------------------
        # RISK BADGE
        # -------------------------
        risk_color = {
            "low": "🟢 Low",
            "medium": "🟡 Medium",
            "high": "🔴 High"
        }

        st.markdown(f"**Risk Level:** {risk_color.get(info['risk'].lower(), 'N/A')}")

        # -------------------------
        # DISEASE INFO
        # -------------------------
        st.markdown("### 🌿 Disease Insights")
        st.info(f"**Cause:** {info['cause']}")
        st.info(f"**Solution:** {info['solution']}")

        # -------------------------
        # CONFIDENCE CHART
        # -------------------------
        st.markdown("### 📊 Model Confidence Distribution")

        labels = ["Early Blight", "Late Blight", "Healthy"]
        values = preds

        fig, ax = plt.subplots()
        ax.bar(labels, values)
        ax.set_ylabel("Confidence")
        ax.set_ylim([0, 1])

        st.pyplot(fig)

        # -------------------------
        # TOP PREDICTIONS
        # -------------------------
        st.markdown("### 🔍 Top Predictions")

        class_names = ["Early Blight", "Late Blight", "Healthy"]

        for i in top2_idx:
            st.write(f"{class_names[i]} : {preds[i]*100:.2f}%")

        # -------------------------
        # SMART INSIGHTS (🔥 UNIQUE FEATURE)
        # -------------------------
        st.markdown("### 💡 Smart Insights")

        if "late" in clean_label:
            st.warning("Severe infection detected. Immediate action required.")
        elif "early" in clean_label:
            st.warning("Moderate infection. Monitor closely and treat early.")
        else:
            st.success("Plant is healthy. Maintain current care.")
