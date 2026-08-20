import sys, json, time
import numpy as np
import torch
import torch.nn as nn
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
import streamlit as st

sys.path.insert(0, ".")
from src.model import build_model
from src.xai import GradCAMPlusPlus, overlay_heatmap, mc_dropout_uncertainty
from src.deploy import ClinicalInferenceEngine, to_fhir_report, export_torchscript

# ── PAGE CONFIG
st.set_page_config(
    page_title="Lumbar MRI Diagnostic AI",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.4rem; font-weight: 800; text-align: center;
    padding: 1.25rem 0 0.5rem 0;
    background: linear-gradient(90deg, #6366f1, #8b5cf6 45%, #ec4899);
    -webkit-background-clip: text; background-clip: text;
    color: transparent;
}
.sub-header {
    font-size: 1.05rem; color: #6b7280;
    text-align: center; margin-bottom: 1.25rem;
}
.disclaimer {
    background: linear-gradient(135deg, #fff7ed, #fef3c7);
    border: 1px solid #fbbf24; border-left: 5px solid #f59e0b;
    border-radius: 10px; padding: 0.75rem 1.1rem;
    font-size: 0.85rem; color: #92400e; margin-top: 1rem;
}

/* Colored section headers */
.section-header {
    display: flex; align-items: center; gap: 0.6rem;
    font-size: 1.3rem; font-weight: 700; color: #ffffff;
    padding: 0.55rem 1rem; border-radius: 10px;
    margin: 1.1rem 0 0.9rem 0;
}

/* Stat cards */
.stat-card {
    border-radius: 12px; padding: 0.9rem 0.6rem; text-align: center;
    color: #ffffff; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
.stat-card .stat-value { font-size: 1.6rem; font-weight: 800; line-height: 1.2; }
.stat-card .stat-label { font-size: 0.78rem; font-weight: 600; opacity: 0.92; margin-top: 0.15rem; }

/* Step badges in Quick Guide */
.step-badge {
    display: flex; align-items: center; gap: 0.6rem;
    margin: 0.35rem 0; font-size: 0.92rem; color: #374151;
}
.step-num {
    flex: 0 0 auto; width: 1.5rem; height: 1.5rem; border-radius: 50%;
    background: linear-gradient(135deg, #6366f1, #ec4899);
    color: #fff; font-size: 0.78rem; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
}

/* Upload dropzone accent */
[data-testid="stFileUploaderDropzone"] {
    border: 2px dashed #a5b4fc !important;
    background: #f5f6ff !important; border-radius: 12px !important;
}

/* Sidebar accent */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f5f3ff 0%, #eef2ff 100%);
}
</style>
""", unsafe_allow_html=True)


# ── COLOR HELPERS
SECTION_COLORS = {
    "indigo": "linear-gradient(90deg,#6366f1,#818cf8)",
    "violet": "linear-gradient(90deg,#8b5cf6,#a78bfa)",
    "pink":   "linear-gradient(90deg,#ec4899,#f472b6)",
    "red":    "linear-gradient(90deg,#dc2626,#f87171)",
    "amber":  "linear-gradient(90deg,#f59e0b,#fbbf24)",
    "teal":   "linear-gradient(90deg,#0d9488,#2dd4bf)",
    "green":  "linear-gradient(90deg,#16a34a,#4ade80)",
    "blue":   "linear-gradient(90deg,#2563eb,#60a5fa)",
}


def section_header(icon, title, color="indigo"):
    st.markdown(
        '<div class="section-header" style="background:{bg}">{icon} {title}</div>'.format(
            bg=SECTION_COLORS.get(color, SECTION_COLORS["indigo"]), icon=icon, title=title),
        unsafe_allow_html=True)


def stat_card(col, label, value, color="blue"):
    col.markdown(
        '<div class="stat-card" style="background:{bg}">'
        '<div class="stat-value">{value}</div>'
        '<div class="stat-label">{label}</div></div>'.format(
            bg=SECTION_COLORS.get(color, SECTION_COLORS["blue"]), value=value, label=label),
        unsafe_allow_html=True)


# ── MODEL LOADING
@st.cache_resource(show_spinner="Loading diagnostic model...")
def load_engine():
    with open("configs/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_pt = Path("outputs/exports/lumbar_model.pt")
    if not model_pt.exists():
        model = build_model(cfg)
        state_dict_path = Path("outputs/exports/lumbar_state_dict.pt")
        if state_dict_path.exists():
            state = torch.load(state_dict_path, map_location="cpu")
            model.load_state_dict(state, strict=False)
        else:
            st.warning(
                "No trained weights found for inference "
                "(outputs/exports/lumbar_state_dict.pt) -- using untrained model.")
        model.eval()
        export_torchscript(model, str(model_pt), cfg["data"]["image_size"])
    engine = ClinicalInferenceEngine(str(model_pt))
    return engine, cfg


@st.cache_resource(show_spinner="Loading XAI model...")
def load_xai_model():
    with open("configs/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model = build_model(cfg)
    state_dict_path = Path("outputs/exports/lumbar_state_dict.pt")
    if state_dict_path.exists():
        state = torch.load(state_dict_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
    else:
        st.warning(
            "No trained weights found for GradCAM/uncertainty "
            "(outputs/exports/lumbar_state_dict.pt) -- using untrained model.")
    model.eval()
    return model


# ── IMAGE HELPERS
def load_uploaded_image(uploaded_file):
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (512, 512))
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def img_to_tensor(img_rgb):
    img = img_rgb.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    return torch.tensor(img.transpose(2, 0, 1)[None], dtype=torch.float32)


# ── PLOT HELPERS
def plot_gradcam(img_tensor, model, task="classification", class_idx=1, title=""):
    target_layer = list(model.encoder.backbone.encoder.children())[-1]
    gcam = GradCAMPlusPlus(model, target_layer)
    cam = gcam.generate(img_tensor, task=task, class_idx=class_idx)
    overlay = overlay_heatmap(img_tensor, cam)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    img_np = (img_tensor[0, 0].detach().cpu().numpy() * 0.5 + 0.5).clip(0, 1)
    axes[0].imshow(img_np, cmap="gray")
    axes[0].set_title("Input MRI", fontsize=11)
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title("GradCAM++ - {}".format(title), fontsize=11)
    axes[1].axis("off")
    sm = cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
    plt.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04, label="Activation")
    plt.tight_layout()
    return fig


def plot_uncertainty(unc_result):
    conditions = ["SCS", "LFN", "RFN", "LSS", "RSS"]
    levels = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
    unc_map = unc_result["std"][0].mean(-1).cpu().numpy().reshape(5, 5)
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(unc_map, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=0.3)
    ax.set_xticks(range(5))
    ax.set_xticklabels(levels, fontsize=9)
    ax.set_yticks(range(5))
    ax.set_yticklabels(conditions, fontsize=9)
    ax.set_title("Epistemic Uncertainty (MC-Dropout)", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Std deviation")
    for i in range(5):
        for j in range(5):
            color = "white" if unc_map[i, j] > 0.15 else "black"
            ax.text(j, i, "{:.2f}".format(unc_map[i, j]),
                    ha="center", va="center", fontsize=8, color=color)
    plt.tight_layout()
    return fig


def color_grade(val):
    s = str(val)
    if "Severe" in s:
        return "background-color: #ffebee; color: #c62828; font-weight: bold"
    if "Moderate" in s:
        return "background-color: #fff8e1; color: #e65100; font-weight: bold"
    if "Normal" in s:
        return "background-color: #e8f5e9; color: #2e7d32"
    return ""


def color_morph(val):
    s = str(val)
    if "Normal" in s:
        return "background-color: #e8f5e9; color: #2e7d32"
    if "Herniated" in s or "Osteophyte" in s:
        return "background-color: #ffebee; color: #c62828; font-weight: bold"
    return "background-color: #fff8e1; color: #e65100; font-weight: bold"


# ── SIDEBAR
with st.sidebar:
    st.title("Settings")
    st.subheader("Analysis Options")
    run_morphology = st.checkbox("Disc morphology", value=True,
                                  help="Normal/Degenerated/Bulging/Herniated/Thinning/Osteophyte "
                                       "for L3/L4, L4/L5 (lumbar) and L5/S1 (sacral).")
    run_gradcam = st.checkbox("GradCAM++ heatmap", value=True)
    run_uncertainty = st.checkbox("Uncertainty analysis", value=True)
    mc_passes = st.slider("MC-Dropout passes", 5, 30, 5,
                           help="More passes = smoother uncertainty estimate but slower (roughly 1s/pass on CPU).")
    st.subheader("GradCAM Target")
    gcam_task = "classification"
    grade_names = ["Normal/Mild", "Moderate", "Severe"]
    gcam_class = st.selectbox(
        "Grade to explain", options=[0, 1, 2],
        format_func=lambda i: grade_names[i], index=1)
    st.subheader("Export")
    export_fhir = st.checkbox("Export FHIR report", value=True)
    st.divider()
    st.caption("Model: MobileViT-v2 + FPN")
    st.caption("Params: 10.59M")
    st.caption("Device: CPU")


# ── HEADER
st.markdown('<div class="main-header">Lumbar MRI Diagnostic AI</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Multi-disease detection and explainable '
    'grading for lumbosacral MRI</div>',
    unsafe_allow_html=True)
st.markdown(
    '<div class="disclaimer">Research prototype only. Not validated for '
    'clinical use. All findings must be reviewed by a qualified radiologist '
    'before any clinical decision is made.</div>',
    unsafe_allow_html=True)
st.divider()


# ── UPLOAD
col_upload, col_info = st.columns([2, 1])
with col_upload:
    section_header("📤", "Upload MRI Image", "indigo")
    uploaded = st.file_uploader(
        "Upload a lumbar MRI slice (JPG or PNG)",
        type=["jpg", "jpeg", "png"],
    )
with col_info:
    section_header("🧭", "Quick Guide", "pink")
    steps = [
        "Upload a lumbar MRI image",
        "Click **Run Analysis**",
        "View findings, heatmaps and uncertainty",
        "Download the FHIR report",
    ]
    for i, step in enumerate(steps, 1):
        st.markdown(
            '<div class="step-badge"><span class="step-num">{}</span>{}</div>'.format(i, step),
            unsafe_allow_html=True)


# ── MAIN ANALYSIS
if uploaded is not None:
    img_rgb = load_uploaded_image(uploaded)
    if img_rgb is None:
        st.error("Could not read image. Please upload a valid JPG or PNG.")
        st.stop()

    st.image(img_rgb[:, :, 0], caption="Uploaded MRI",
             width=300, clamp=True)

    run_btn = st.button("Run Analysis", type="primary", use_container_width=True)

    if run_btn:
        engine, cfg = load_engine()
        img_tensor = img_to_tensor(img_rgb)

        with st.spinner("Running inference..."):
            result = engine.predict(img_rgb)

        st.success("Analysis complete in {} ms".format(result["latency_ms"]))
        st.divider()

        # FINDINGS
        section_header("🩺", "Clinical Findings", "blue")
        findings = result["findings"]

        conditions = [
            "Spinal canal stenosis",
            "Left neural foraminal narrowing",
            "Right neural foraminal narrowing",
            "Left subarticular stenosis",
            "Right subarticular stenosis",
        ]
        LUMBAR_LEVELS = ["L1/L2", "L2/L3", "L3/L4", "L4/L5"]
        SACRAL_LEVELS = ["L5/S1"]
        morphology = result.get("morphology", {}) if run_morphology else {}
        LUMBAR_MORPH_LEVELS = ["L3/L4", "L4/L5"]
        SACRAL_MORPH_LEVELS = ["L5/S1 (Sacral)"]

        def build_findings_df(levels):
            rows = []
            for cond in conditions:
                row = {"Condition": cond}
                for level in levels:
                    key = "{}_{}".format(cond, level)
                    info = findings.get(key, {"grade": "N/A", "confidence": 0})
                    row[level] = "{} ({:.0f}%)".format(
                        info["grade"], info["confidence"] * 100)
                rows.append(row)
            return pd.DataFrame(rows).set_index("Condition")

        def region_counts(levels):
            keys = ["{}_{}".format(c, l) for c in conditions for l in levels]
            sev = sum(1 for k in keys if findings.get(k, {}).get("grade") == "Severe")
            mod = sum(1 for k in keys if findings.get(k, {}).get("grade") == "Moderate")
            nor = sum(1 for k in keys if findings.get(k, {}).get("grade") == "Normal/Mild")
            return sev, mod, nor, len(keys)

        def render_morphology(levels):
            region_morph = {lvl: morphology[lvl] for lvl in levels if lvl in morphology}
            if not region_morph:
                return
            st.markdown("**Disc Morphology**")
            morph_cols = st.columns(len(region_morph))
            morph_palette = ["indigo", "violet", "pink"]
            for i, (level, info) in enumerate(region_morph.items()):
                stat_card(
                    morph_cols[i],
                    level,
                    "{} ({:.0f}%)".format(info["type"], info["confidence"] * 100),
                    morph_palette[i % len(morph_palette)],
                )
            morph_df = pd.DataFrame([
                {"Level": level, "Type": info["type"],
                 "Confidence": "{:.0f}%".format(info["confidence"] * 100)}
                for level, info in region_morph.items()
            ]).set_index("Level")
            st.dataframe(morph_df.style.map(color_morph, subset=["Type"]),
                         use_container_width=True)

        tab_lumbar, tab_sacral = st.tabs(
            ["🦵 Lumbar (L1/L2 - L4/L5)", "🦴 Sacral (L5/S1)"])

        with tab_lumbar:
            sev, mod, nor, total = region_counts(LUMBAR_LEVELS)
            m1, m2, m3, m4 = st.columns(4)
            stat_card(m1, "Total findings", total, "indigo")
            stat_card(m2, "Severe",   sev, "red")
            stat_card(m3, "Moderate", mod, "amber")
            stat_card(m4, "Normal",   nor, "green")
            st.dataframe(
                build_findings_df(LUMBAR_LEVELS).style.map(color_grade),
                use_container_width=True)
            if run_morphology:
                render_morphology(LUMBAR_MORPH_LEVELS)

        with tab_sacral:
            sev, mod, nor, total = region_counts(SACRAL_LEVELS)
            m1, m2, m3, m4 = st.columns(4)
            stat_card(m1, "Total findings", total, "indigo")
            stat_card(m2, "Severe",   sev, "red")
            stat_card(m3, "Moderate", mod, "amber")
            stat_card(m4, "Normal",   nor, "green")
            st.dataframe(
                build_findings_df(SACRAL_LEVELS).style.map(color_grade),
                use_container_width=True)
            if run_morphology:
                render_morphology(SACRAL_MORPH_LEVELS)
            st.caption("L5/S1 is the \"sacral\" level per the project brief -- "
                       "the sacrum is a single fused bone with no discs of its "
                       "own beyond the L5/S1 junction.")

        # GRADCAM
        if run_gradcam:
            st.divider()
            section_header("🔥", "GradCAM++ Explainability", "amber")
            xai_model = load_xai_model()
            with st.spinner("Computing GradCAM++..."):
                gcam_fig = plot_gradcam(
                    img_tensor, xai_model,
                    task=gcam_task, class_idx=gcam_class,
                    title="Grade: {}".format(grade_names[gcam_class]),
                )
            st.pyplot(gcam_fig)
            plt.close()
            st.caption("Warmer colours show where the model focused most.")

        # UNCERTAINTY
        if run_uncertainty:
            st.divider()
            section_header("📊", "Uncertainty Analysis", "teal")
            xai_model = load_xai_model()
            with st.spinner("Running MC-Dropout..."):
                unc = mc_dropout_uncertainty(
                    xai_model, img_tensor, n_passes=mc_passes)
            unc_fig = plot_uncertainty(unc)
            st.pyplot(unc_fig)
            plt.close()
            avg_unc = unc["std"].mean().item()
            n_unc   = int(unc["uncertain_mask"].sum().item())
            u1, u2  = st.columns(2)
            stat_card(u1, "Avg uncertainty", "{:.4f}".format(avg_unc), "teal")
            stat_card(u2, "Uncertain tasks", "{}/25".format(n_unc), "violet")
            if avg_unc > 0.15:
                st.warning("High uncertainty. Manual review strongly recommended.")
            else:
                st.success("Model confidence acceptable for preliminary screening.")

        # FHIR
        if export_fhir:
            st.divider()
            section_header("📄", "FHIR DiagnosticReport", "green")
            fhir = to_fhir_report(result, patient_id="P001", study_id="S001")
            fhir_json = json.dumps(fhir, indent=2)
            st.download_button(
                label="Download FHIR Report (JSON)",
                data=fhir_json,
                file_name="lumbar_diagnostic_report.json",
                mime="application/json",
            )
            with st.expander("Preview FHIR report"):
                st.json(fhir)

else:
    st.info("Upload a lumbar MRI image to begin analysis.")
    feature_cards = [
        ("🎯", "Detection", "8 pathology classes across 4 FPN scales.", "indigo"),
        ("🏷️", "Classification", "25 conditions graded Normal/Mild, Moderate, or Severe.", "pink"),
        ("🦴", "Sacral / Disc Morphology", "Normal/Degenerated/Bulging/Herniated/Thinning/Osteophyte for L3-L4 through L5/S1.", "violet"),
    ]
    for col, (icon, title, desc, color) in zip(st.columns(3), feature_cards):
        with col:
            st.markdown(
                '<div class="stat-card" style="background:{bg}; text-align:left; padding:1rem;">'
                '<div style="font-size:1.6rem;">{icon}</div>'
                '<div style="font-size:1.05rem; font-weight:700; margin-top:0.3rem;">{title}</div>'
                '<div style="font-size:0.85rem; font-weight:400; opacity:0.95; margin-top:0.3rem;">{desc}</div>'
                '</div>'.format(bg=SECTION_COLORS[color], icon=icon, title=title, desc=desc),
                unsafe_allow_html=True)