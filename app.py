import streamlit as st
st.set_page_config(
    page_title="RPS/RPC Prediction App",
    layout="wide",
    initial_sidebar_state="expanded",
)

import torch
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_PATH = "model/pytorch_model.pt"
SCALER_PATH = "model/scaler_y.pkl"
SENTENCE_TRANSFORMER_MODEL = "all-mpnet-base-v2"
EMBEDDING_DIM = 768  # must match model training

TARGET_COLS = [
    "total_search_count", "total_click_count", "total_new_rpk", "RPS", "RPC"
]

class SimpleMLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, out_dim)
        )
    def forward(self, x):
        return self.seq(x)

@st.cache_resource
def load_all():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL, device=device)
    model = SimpleMLP(EMBEDDING_DIM, 3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    try:
        scaler_y = joblib.load(SCALER_PATH)
    except Exception:
        scaler_y = None
    return embedder, model, scaler_y, device

embedder, model, scaler_y, device = load_all()

# --- Logo and welcome ---
_, center_col, _ = st.columns([0.1, 0.5, 0.1])
with center_col:
    st.image("assets/logo.png")

st.markdown(
    """
    <div style="text-align: center;">
        Welcome! This app uses advanced AI models to predict Revenue Per Search (RPS) and Revenue Per Click (RPC) metrics for a given keyword.<br>
        Simply enter your keyword and see the estimated revenue performance.
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("About the Metrics")
st.sidebar.markdown(
    """
- **RPS (Revenue Per Search):** An estimate of the average revenue generated each time this keyword is searched.
- **RPC (Revenue Per Click):** An estimate of the average revenue generated each time an ad for this keyword is clicked.

These metrics help evaluate the monetization potential and value of keywords in advertising campaigns.
"""
)
st.sidebar.markdown("---")
st.sidebar.info(f"Using device: {device.upper()}")
st.sidebar.success("All models loaded successfully.")

keyword_input = st.text_input(
    "🔑 Enter your keyword below:",
    placeholder="e.g., sustainable energy solutions, best travel cameras 2024",
    help="Type the keyword you want to analyze for RPS and RPC predictions.",
)

def predict_metrics_from_keyword(keyword_text: str):
    if not model or not embedder:
        st.error("Models are not fully loaded. Cannot perform prediction.")
        return None
    if not keyword_text.strip():
        return None
    embedding = embedder.encode([keyword_text], device=device, show_progress_bar=False)
    embedding_torch = torch.tensor(embedding, dtype=torch.float32).to(device)
    with torch.no_grad():
        pred_scaled = model(embedding_torch).cpu().numpy()
    if scaler_y is not None:
        pred = scaler_y.inverse_transform(pred_scaled)
    else:
        pred = pred_scaled
    tsc, tcc, trpk = pred[0]
    rps = trpk / tsc if tsc != 0 else np.nan
    rpc = trpk / tcc if tcc != 0 else np.nan
    result_dict = {
        "total_search_count": tsc,
        "total_click_count": tcc,
        "total_new_rpk": trpk,
        "RPS": rps,
        "RPC": rpc
    }
    return result_dict

if keyword_input is not None:
    if st.button("🚀 Predict Revenue Metrics", type="primary", use_container_width=True):
        if keyword_input.strip():
            with st.spinner("🧠 Analyzing keyword and calculating revenue metrics..."):
                predicted_values = predict_metrics_from_keyword(keyword_input)
            if predicted_values:
                st.subheader("💰 Predicted Revenue Metrics")
                display_names = [
                    ("Total Searches", "total_search_count"),
                    ("Total Clicks", "total_click_count"),
                    ("Total RPK", "total_new_rpk"),
                    ("RPS", "RPS"),
                    ("RPC", "RPC")
                ]
                cols = st.columns(len(display_names))
                for i, (label, key) in enumerate(display_names):
                    val = predicted_values[key]
                    if key in ["RPS", "RPC"]:
                        formatted_val = f"${float(val):.4f}"
                    else:
                        formatted_val = f"{float(val):.4f}"
                    cols[i].metric(label=label, value=formatted_val)
                st.markdown("---")
                with st.expander("📝 View Raw Prediction Output", expanded=False):
                    raw_output_str = (
                        "{\n"
                        + "\n".join(
                            [f"    '{k}': {v}," for k, v in predicted_values.items()]
                        )
                        + "\n}"
                    )
                    st.code(raw_output_str, language="text")
        else:
            st.warning("🔔 Please enter a keyword to get revenue predictions.")
