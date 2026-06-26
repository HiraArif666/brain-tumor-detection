import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from model import get_model, CLASS_NAMES

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
        .stApp {
            background-color: #0f1117;
            color: #ffffff;
        }
        [data-testid="stSidebar"] {
            background-color: #1a1d27;
            border-right: 1px solid #2e3250;
        }
        [data-testid="stAlert"] {
            border-radius: 10px;
        }
        h1 {
            color: #00b4d8;
            font-size: 2rem;
            font-weight: 700;
        }
        hr {
            border-color: #2e3250;
        }
        .result-box {
            background-color: #1e2130;
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #2e3250;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = get_model(num_classes=4)
    model.load_state_dict(torch.load(
        "brain_tumor_model.pth",
        map_location=torch.device("cpu")
    ))
    model.eval()
    return model

model = load_model()

# ── Image Transform ───────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Brain Tumor Detection")
    st.markdown("---")
    st.markdown("### 📋 Tumor Types")
    st.markdown("""
    - **Glioma** — tumor in glial cells
    - **Meningioma** — tumor in meninges
    - **Pituitary** — tumor in pituitary gland
    - **No Tumor** — healthy brain scan
    """)
    st.markdown("---")
    st.markdown("### ℹ️ How to use")
    st.markdown("""
    1. Upload an MRI scan image
    2. Click Analyze
    3. Get instant AI prediction
    """)
    st.markdown("---")
    st.markdown("Built with PyTorch + ResNet18")

# ── Main Area ─────────────────────────────────────────────
st.markdown("# 🧠 Brain Tumor Detection")
st.markdown("Upload an MRI scan to detect the type of brain tumor using AI.")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader(
        "Upload MRI Scan",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded MRI Scan", use_column_width=True)

        if st.button("🔍 Analyze", use_container_width=True):
            with st.spinner("Analyzing MRI scan..."):
                # Preprocess
                input_tensor = transform(image).unsqueeze(0)

                # Predict
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = F.softmax(outputs, dim=1)[0]
                    confidence, predicted = torch.max(probabilities, 0)

                predicted_class = CLASS_NAMES[predicted.item()]
                confidence_score = confidence.item() * 100

                # Store in session
                st.session_state.result = predicted_class
                st.session_state.confidence = confidence_score
                st.session_state.probabilities = probabilities

with col2:
    if "result" in st.session_state:
        result = st.session_state.result
        confidence = st.session_state.confidence
        probs = st.session_state.probabilities

        # Color based on result
        color = "#ff4757" if result != "No Tumor" else "#2ed573"

        st.markdown(f"""
            <div class='result-box'>
                <h2 style='color: {color}; font-size: 1.5rem;'>
                    {"⚠️" if result != "No Tumor" else "✅"} {result}
                </h2>
                <p style='font-size: 1.1rem; color: #aaaaaa;'>
                    Confidence: <strong style='color: white;'>{confidence:.2f}%</strong>
                </p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 📊 All Probabilities")
        for i, name in enumerate(CLASS_NAMES):
            prob = probs[i].item() * 100
            st.progress(int(prob), text=f"{name}: {prob:.2f}%")

        st.markdown("---")
        st.warning("⚠️ This is an AI prediction only. Always consult a medical professional for diagnosis.")