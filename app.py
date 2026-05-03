import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from utils.predict import predict
from utils.disease_info import disease_info

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Plant AI", page_icon="🌱", layout="wide")

# -------------------------
# LOAD CSS
# -------------------------
try:
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except:
    pass

# -------------------------
# SESSION STATE
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------
# HEADER
# -------------------------
st.markdown('<div class="header-title">🌱 Plant Intelligence Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">AI-powered plant disease detection & insights</div>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.title("⚙️ Controls")

theme = st.sidebar.toggle("🌗 Dark Mode", value=True)

st.sidebar.markdown("### 📌 About")
st.sidebar.info("Detect plant diseases with AI and get actionable insights.")

st.sidebar.markdown("### 📜 History (Last 5)")
for item in st.session_state.history[-5:][::-1]:
    st.sidebar.write(f"{item}")

# -------------------------
# FILE UPLOAD
# -------------------------
uploaded = st.file_uploader("📤 Upload a leaf image", type=["jpg", "png", "jpeg"])

if uploaded:
    image = Image.open(uploaded)

    # Tabs
    tab1, tab2 = st.tabs(["🔍 Analysis", "📊 Insights"])

    with tab1:
        col1, col2 = st.columns([1.2, 1])

        # IMAGE
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # RESULT
        with col2:
            with st.spinner("Analyzing image..."):
                label, confidence, probs, top2_idx = predict(image)

            # Save history
            st.session_state.history.append(f"{label.title()} ({confidence*100:.1f}%)")

            info = disease_info.get(label, {"cause":"Unknown","solution":"N/A","risk":"N/A"})

            # Risk badge
            risk = info["risk"].lower()
            badge_class = "badge-low" if risk=="low" else "badge-med" if risk=="medium" else "badge-high"

            st.markdown('<div class="card">', unsafe_allow_html=True)

            st.markdown("### 🧠 Prediction")
            st.markdown(f"## {label.title()}")

            st.markdown("### 📈 Confidence")
            st.markdown(f"## {confidence*100:.2f}%")

            st.markdown(f'<span class="badge {badge_class}">{info["risk"]} Risk</span>', unsafe_allow_html=True)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            st.markdown("### 🌿 Disease Info")
            st.write(f"**Cause:** {info['cause']}")
            st.write(f"**Solution:** {info['solution']}")

            st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------
    # INSIGHTS TAB
    # -------------------------
    with tab2:
        st.markdown("### 📊 Model Confidence")

        class_labels = ["Early Blight", "Late Blight", "Healthy"]

        fig, ax = plt.subplots()
        ax.bar(class_labels, probs)
        ax.set_title("Prediction Confidence")
        st.pyplot(fig)

        st.markdown("### 🔝 Top Predictions")
        for i in top2_idx:
            st.write(f"{class_labels[i]} → {probs[i]*100:.2f}%")

# -------------------------
# FOOTER
# -------------------------
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.caption("🚀 Built with Streamlit • TensorFlow • Deep Learning")