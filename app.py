import io
import tempfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from backend import (
    load_image, binarize, extract_features, train_pipeline,
    analyze_features, get_radar_values
)

st.set_page_config(
    page_title="Handwriting Personality Analyzer",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown("""
<style>
    .block-container { max-width: 1250px; padding-top: 2rem; }
    .hero {
        padding: 2rem 2.2rem;
        border-radius: 22px;
        background: linear-gradient(135deg, #111827 0%, #312e81 55%, #0f766e 100%);
        color: white;
        margin-bottom: 1.5rem;
    }
    .hero h1 { margin: 0 0 .5rem 0; font-size: 2.5rem; }
    .hero p { margin: 0; opacity: .9; font-size: 1.05rem; }
    .trait-card {
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        height: 100%;
        background: rgba(128,128,128,.06);
    }
    .trait-label { font-size: .85rem; opacity: .7; }
    .trait-value { font-size: 1.35rem; font-weight: 700; margin: .25rem 0 .5rem; }
    .small-note { font-size: .88rem; opacity: .75; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_pipeline():
    return train_pipeline()

def overlay_lines(img, lines):
    overlay = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    for s, e in lines:
        cv2.line(overlay, (0, s), (overlay.shape[1], s), (255, 0, 0), 2)
        cv2.line(overlay, (0, e), (overlay.shape[1], e), (0, 200, 0), 2)
    return overlay

def make_preprocessing_figure(img, binary):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(binary, cmap="gray")
    axes[1].set_title("Binarized (ink mask)")
    axes[1].axis("off")
    fig.tight_layout()
    return fig

def make_cluster_figure(analysis):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    scatter = ax.scatter(
        analysis["X_pca"][:, 0], analysis["X_pca"][:, 1],
        c=analysis["cluster_labels"], cmap="tab10", s=8, alpha=.35
    )
    ax.scatter(
        analysis["sample_pca"][:, 0], analysis["sample_pca"][:, 1],
        c="black", s=250, marker="*", edgecolors="white",
        linewidths=1.5, label="Your sample", zorder=5
    )
    pca = analysis["pca"]
    ax.set_title("K-Means clusters projected via PCA")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    ax.legend()
    fig.colorbar(scatter, ax=ax, label="Cluster ID")
    fig.tight_layout()
    return fig

def make_radar_figure(features):
    labels = [
        "Expressiveness\n(slant)",
        "Independence\n(letter spacing)",
        "Organization\n(line spacing)",
        "Intensity\n(pressure)",
    ]
    values = get_radar_values(features)
    values += values[:1]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 5.5), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    ax.set_title("Handwriting Trait Profile (ML-predicted, for fun)", pad=20)
    fig.tight_layout()
    return fig

# ---------- Header ----------
st.markdown("""
<div class="hero">
  <h1>✍️ Handwriting Personality Analyzer</h1>
  <p>Upload a handwriting sample and explore the same OpenCV + ML pipeline through a polished cloud interface.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Analysis")
    uploaded = st.file_uploader(
        "Upload handwriting image",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
        help="Use a clear image with visible handwritten text.",
    )
    st.divider()
    st.caption("Pipeline preserved")
    st.write("• OpenCV preprocessing")
    st.write("• 5 extracted handwriting features")
    st.write("• Synthetic weak supervision")
    st.write("• Random Forest vs Logistic Regression")
    st.write("• K-Means + PCA")
    st.write("• Trait radar profile")
    st.divider()
    st.warning(
        "The personality interpretation is folklore/entertainment, not a scientific "
        "assessment. The ML pipeline itself is computationally real."
    )

if not uploaded:
    st.info("Upload a handwriting image from the sidebar to start the analysis.")
    st.markdown("### What this software does")
    c1, c2, c3 = st.columns(3)
    c1.metric("Features", "5")
    c2.metric("Trait groups", "4")
    c3.metric("Clusters", "4")
    st.markdown(
        "The cloud UI does not ask the user to edit filenames or run notebook cells. "
        "Everything is executed after the image is uploaded."
    )
    st.stop()

# ---------- Run backend ----------
with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix or ".png") as tmp:
    tmp.write(uploaded.getbuffer())
    temp_path = tmp.name

try:
    with st.spinner("Running handwriting analysis..."):
        img, gray = load_image(temp_path)
        binary = binarize(gray)
        features, detected_lines = extract_features(img, gray, binary)
        pipeline = get_pipeline()
        analysis = analyze_features(features, pipeline)
finally:
    try:
        Path(temp_path).unlink(missing_ok=True)
    except Exception:
        pass

# ---------- Overview ----------
st.success("Analysis complete.")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Results", "🔬 Processing", "🧠 ML Details", "📈 Visualizations"
])

with tab1:
    st.subheader("Your handwriting report")
    cols = st.columns(4)
    for col, (trait, (label, desc, algo)) in zip(
        cols, analysis["predicted_traits"].items()
    ):
        with col:
            st.markdown(
                f'<div class="trait-card">'
                f'<div class="trait-label">{trait}</div>'
                f'<div class="trait-value">{label}</div>'
                f'<div class="small-note">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
            st.caption(f"Model: {algo}")

    st.markdown("### Unsupervised archetype")
    st.info(
        f'Cluster {analysis["sample_cluster"]} — {analysis["cluster_blurb"]}.'
    )

    st.markdown("### Extracted measurements")
    feature_rows = {
        "Lines detected": features["lines_detected"],
        "Line spacing ratio": round(features["line_spacing_ratio"], 4),
        "Letter spacing (px)": round(features["letter_spacing_px"], 4),
        "Slant angle (°)": round(features["slant_angle_deg"], 4),
        "Ink darkness": round(features["ink_darkness"], 4),
        "Stroke width (px)": round(features["stroke_width_px"], 4),
    }
    st.dataframe(
        [{"Feature": k, "Value": v} for k, v in feature_rows.items()],
        use_container_width=True,
        hide_index=True,
    )

with tab2:
    st.subheader("Image preprocessing")
    st.pyplot(make_preprocessing_figure(img, binary), use_container_width=True)
    st.subheader("Detected text lines")
    st.image(
        overlay_lines(img, detected_lines),
        caption=f"Detected {len(detected_lines)} text line(s) — red = top, green = bottom",
        use_container_width=True,
    )

with tab3:
    st.subheader("Synthetic training-data comparison")
    table = []
    for trait, rf_acc, lr_acc, chosen in pipeline["results_table"]:
        table.append({
            "Trait": trait,
            "Random Forest": f"{rf_acc:.3f}",
            "Logistic Regression": f"{lr_acc:.3f}",
            "Chosen model": chosen,
        })
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption(
        "The UI reports the same model-selection logic from the supplied project: "
        "the higher holdout accuracy is selected per trait."
    )

    st.subheader("Pipeline")
    st.code(
        "Image → grayscale → Gaussian blur → Otsu threshold → morphology → "
        "5 handwriting features → StandardScaler → RF/LR comparison → "
        "K-Means + PCA → trait report",
        language="text",
    )

with tab4:
    left, right = st.columns(2)
    with left:
        st.pyplot(make_cluster_figure(analysis), use_container_width=True)
    with right:
        st.pyplot(make_radar_figure(features), use_container_width=True)

st.divider()
st.caption(
    "For entertainment only: handwriting-based personality interpretation is not a "
    "scientifically validated psychological assessment."
)
