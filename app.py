import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image
import pandas as pd
import plotly.express as px

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🚧",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.hero {
    background: linear-gradient(135deg,#1e3c72,#2a5298);
    padding: 2rem;
    border-radius: 20px;
    text-align:center;
    color:white;
    margin-bottom:20px;
}

.metric-card{
    background:#161b22;
    padding:15px;
    border-radius:15px;
    border:1px solid #30363d;
    text-align:center;
}

.high{
    background:#ff4b4b;
    color:white;
    padding:10px;
    border-radius:10px;
    text-align:center;
    font-weight:bold;
}

.medium{
    background:#ffa500;
    color:white;
    padding:10px;
    border-radius:10px;
    text-align:center;
    font-weight:bold;
}

.low{
    background:#00c853;
    color:white;
    padding:10px;
    border-radius:10px;
    text-align:center;
    font-weight:bold;
}

.about-box{
    background:#161b22;
    padding:20px;
    border-radius:15px;
    border:1px solid #30363d;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("road_damage_cnn.h5")

@st.cache_resource
def load_encoder():
    with open("label_encoder.pkl","rb") as f:
        return pickle.load(f)

model = load_model()
encoder = load_encoder()

# --------------------------------------------------
# AUTO IMAGE SIZE
# --------------------------------------------------
INPUT_SIZE = (
    model.input_shape[1],
    model.input_shape[2]
)

# --------------------------------------------------
# PREPROCESS
# --------------------------------------------------
def preprocess(img):

    img = img.convert("RGB")
    img = img.resize(INPUT_SIZE)

    arr = np.array(img)/255.0
    arr = np.expand_dims(arr,axis=0)

    return arr

# --------------------------------------------------
# SEVERITY
# --------------------------------------------------
def get_severity(label):

    label = label.lower()

    if "pothole" in label:
        return "High"

    elif "crack" in label:
        return "Medium"

    else:
        return "Low"

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("""
<div class="hero">
<h1>🚧 AI-Based Road Damage Detection System</h1>
<h4>Smart City Infrastructure Monitoring using CNN</h4>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# ABOUT
# --------------------------------------------------
with st.expander("📘 About Project", expanded=True):

    st.markdown("""
### Why Road Monitoring Matters

Road damage affects public safety, transportation efficiency,
vehicle maintenance costs, and smart-city development.

### CNN in Computer Vision

Convolutional Neural Networks automatically learn visual
patterns from road images and identify:

- Potholes
- Cracks
- Surface Deterioration
- Structural Damage

### Industry Applications

✅ Smart Cities

✅ Highway Monitoring

✅ Municipal Road Inspection

✅ Autonomous Vehicles

✅ Infrastructure Asset Management
""")

# --------------------------------------------------
# UPLOAD
# --------------------------------------------------
st.subheader("📤 Upload Road Image")

uploaded = st.file_uploader(
    "Drag & Drop or Browse Image",
    type=["jpg","jpeg","png"]
)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if uploaded:

    image = Image.open(uploaded)

    col1,col2 = st.columns([1.2,1])

    # IMAGE PREVIEW
    with col1:

        st.subheader("🖼 Uploaded Road Image")

        st.image(
            image,
            use_container_width=True
        )

    # PREDICT
    pred = model.predict(
        preprocess(image),
        verbose=0
    )

    idx = np.argmax(pred)

    label = encoder.inverse_transform([idx])[0]

    confidence = float(np.max(pred)*100)

    severity = get_severity(label)

    # RESULTS
    with col2:

        st.subheader("🔍 Prediction Results")

        c1,c2 = st.columns(2)

        with c1:
            st.metric(
                "Damage Type",
                label
            )

        with c2:
            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

        st.markdown("### Severity")

        if severity=="High":
            st.markdown(
                '<div class="high">HIGH RISK</div>',
                unsafe_allow_html=True
            )

        elif severity=="Medium":
            st.markdown(
                '<div class="medium">MEDIUM RISK</div>',
                unsafe_allow_html=True
            )

        else:
            st.markdown(
                '<div class="low">LOW RISK</div>',
                unsafe_allow_html=True
            )

    st.divider()

    # --------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------
    st.subheader("📊 Prediction Analytics")

    classes = list(encoder.classes_)

    df = pd.DataFrame({
        "Class": classes,
        "Confidence": pred[0]*100
    })

    col3,col4 = st.columns(2)

    with col3:

        fig = px.bar(
            df,
            x="Class",
            y="Confidence",
            title="Class Confidence Graph",
            text_auto=".2f"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col4:

        fig2 = px.pie(
            df,
            names="Class",
            values="Confidence",
            title="Probability Distribution"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    # --------------------------------------------------
    # RECOMMENDATIONS
    # --------------------------------------------------
    st.subheader("🛠 Maintenance Recommendation")

    if severity=="High":

        st.error("""
🚨 Immediate maintenance recommended.

High-risk road condition detected.

Road repair should be prioritized immediately.
""")

    elif severity=="Medium":

        st.warning("""
⚠ Schedule maintenance soon.

Moderate damage may worsen if ignored.
""")

    else:

        st.success("""
✅ Road condition appears stable.

Routine monitoring recommended.
""")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")

st.caption(
    "AI-Based Road Damage Detection System • Smart City Infrastructure Monitoring using CNN"
)
