"""
CPV301 Drone Obstacle Avoidance — Web Prototype
================================================
Streamlit app for project showcase:
  1. Project overview & architecture
  2. Training results dashboard (model comparison)
  3. Live demo (upload image/video → inference if weights available)

Run (from repo root):
    streamlit run prototype/web_app.py
"""

import io
import sys
import tempfile
from pathlib import Path

# Ensure the repo root is on sys.path so `from src.*` imports work
# regardless of which directory Streamlit uses as CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CPV301 — Drone Obstacle Avoidance",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark, premium look
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
/* ---- Import Google Font ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ---- Global ---- */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #e0e0ff;
}

/* ---- Hero gradient text ---- */
.hero-title {
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
    line-height: 1.15;
}
.hero-sub {
    font-size: 1.25rem;
    color: #b0b0cc;
    font-weight: 300;
    margin-bottom: 2rem;
}

/* ---- Metric cards ---- */
.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid rgba(102, 126, 234, 0.25);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.metric-card:hover {
    border-color: rgba(102, 126, 234, 0.6);
    transform: translateY(-4px);
    box-shadow: 0 8px 30px rgba(102, 126, 234, 0.15);
}
.metric-value {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.metric-label {
    font-size: 0.85rem;
    color: #8888aa;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 0.4rem;
}

/* ---- Pipeline step cards ---- */
.pipeline-step {
    background: linear-gradient(135deg, #1e1e3a 0%, #252550 100%);
    border: 1px solid rgba(118, 75, 162, 0.3);
    border-radius: 14px;
    padding: 1.3rem;
    margin-bottom: 0.8rem;
    transition: all 0.3s ease;
}
.pipeline-step:hover {
    border-color: rgba(118, 75, 162, 0.7);
    box-shadow: 0 4px 20px rgba(118, 75, 162, 0.15);
}
.pipeline-step h4 {
    color: #c9b1ff;
    margin: 0 0 0.4rem 0;
    font-weight: 600;
}
.pipeline-step p {
    color: #9999bb;
    margin: 0;
    font-size: 0.9rem;
    line-height: 1.5;
}

/* ---- Section dividers ---- */
.section-header {
    font-size: 1.8rem;
    font-weight: 700;
    color: #e0e0ff;
    margin-top: 2.5rem;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid rgba(102, 126, 234, 0.3);
}

/* ---- Status badge ---- */
.badge-done {
    background: linear-gradient(135deg, #00b09b, #96c93d);
    color: white;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-progress {
    background: linear-gradient(135deg, #f5af19, #f12711);
    color: white;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-pending {
    background: linear-gradient(135deg, #4b6cb7, #182848);
    color: #ccc;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
}

/* ---- Comparison table ---- */
div[data-testid="stDataFrame"] table {
    border-radius: 12px;
    overflow: hidden;
}

/* ---- Tabs ---- */
button[data-baseweb="tab"] {
    font-weight: 600;
    font-size: 1rem;
}

/* ---- Smooth animations ---- */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
.stMarkdown, .stMetric, .stDataFrame {
    animation: fadeIn 0.5s ease-out;
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Training results data (from CLAUDE.md — Phase 6)
# ---------------------------------------------------------------------------
TRAINING_RESULTS = {
    "Model": ["YOLOv8n", "YOLOv8m", "RT-DETR-L"],
    "mAP@0.5": [0.474, 0.592, None],
    "mAP@0.5:0.95": [0.244, 0.323, None],
    "Vehicle AP": [0.745, 0.822, None],
    "Person AP": [0.329, 0.463, None],
    "Other AP": [0.349, 0.491, None],
    "Inference (ms)": [0.8, 6.0, None],
    "~FPS": [1250, 150, None],
    "Params (M)": [3.2, 25.9, 32.0],
    "Status": ["✅ Done", "✅ Done", "⏸ Paused"],
}

CLASS_NAMES = ["vehicle", "person", "static", "flying", "other"]


# ---------------------------------------------------------------------------
# Helper: try to load YOLO model (if weights exist)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model(weights_path: str):
    """Load a YOLO model if the weights file exists."""
    p = Path(weights_path)
    if not p.exists():
        return None
    try:
        from ultralytics import YOLO

        return YOLO(str(p))
    except Exception:
        return None


def find_available_weights():
    """Scan for any .pt files in models/ or project root."""
    candidates = []
    models_dir = Path("models")
    if models_dir.exists():
        candidates.extend(models_dir.glob("*.pt"))
    # Also check for default ultralytics weights in root
    for p in Path(".").glob("*.pt"):
        candidates.append(p)
    return [str(p) for p in candidates]


def run_inference_on_frame(model, frame: np.ndarray):
    """Run detection + tracking + avoidance on a single frame."""
    from src.detection.detector import Detection

    results = model(frame, verbose=False)[0]
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        name = results.names.get(cls, str(cls))
        detections.append(
            Detection(
                bbox=[x1, y1, x2, y2], confidence=conf, class_id=cls, class_name=name
            )
        )
    return detections, results


def draw_detections_on_frame(frame: np.ndarray, detections, avoidance_cmd=None):
    """Draw bounding boxes and labels on frame (web-friendly colors)."""
    out = frame.copy()
    # Color palette for classes
    colors = [
        (78, 126, 234),  # vehicle - blue
        (162, 75, 118),  # person - purple
        (45, 183, 147),  # static - teal
        (234, 166, 78),  # flying - orange
        (150, 150, 180),  # other - grey
    ]
    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d.bbox]
        cls_idx = d.class_id % len(colors)
        color = colors[cls_idx]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{d.class_name} {d.confidence:.2f}"
        # Background for text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            out,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    if avoidance_cmd and isinstance(avoidance_cmd, list) and avoidance_cmd:
        danger_n = sum(1 for r in avoidance_cmd if getattr(r, "risk", None) == "DANGER")
        cmd_text = f"Risk tracks: {len(avoidance_cmd)}  DANGER: {danger_n}"
        cv2.putText(
            out,
            cmd_text,
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 100, 255),
            2,
            cv2.LINE_AA,
        )
    return out


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("# 🛸 CPV301")
    st.markdown("### Drone Obstacle Avoidance")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Overview", "📊 Training Results", "🎯 Live Demo", "📋 Pipeline Status"],
        index=0,
    )

    st.markdown("---")
    st.markdown(
        """
    <div style="color: #666; font-size: 0.8rem; text-align: center;">
        <b>RT-DR-003</b><br>
        FPT University — CPV301<br>
        Computer Vision Course Project
    </div>
    """,
        unsafe_allow_html=True,
    )


# ===================================================================
# PAGE: Overview
# ===================================================================
if page == "🏠 Overview":
    # Hero
    st.markdown(
        """
    <div class="hero-title">Drone Obstacle Avoidance</div>
    <div class="hero-sub">
        How can drones avoid dynamic obstacles during flight?<br>
        A three-stage Computer Vision pipeline: <b>Detection → Tracking → Avoidance Planning</b>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            """<div class="metric-card">
            <div class="metric-value">59.2%</div>
            <div class="metric-label">Best mAP@0.5</div>
        </div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """<div class="metric-card">
            <div class="metric-value">150+</div>
            <div class="metric-label">FPS (YOLOv8m)</div>
        </div>""",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """<div class="metric-card">
            <div class="metric-value">5</div>
            <div class="metric-label">Object Classes</div>
        </div>""",
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            """<div class="metric-card">
            <div class="metric-value">8,629</div>
            <div class="metric-label">Training Images</div>
        </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("")

    # Architecture diagram
    st.markdown(
        '<div class="section-header">🔗 Pipeline Architecture</div>',
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown(
            """
        <div class="pipeline-step">
            <h4>📹 1. Input Frame</h4>
            <p>Video source (camera / file / image directory) → raw frame (numpy array H×W×3)</p>
        </div>
        <div class="pipeline-step">
            <h4>🔍 2. Detection — YoloDetector</h4>
            <p>Ultralytics YOLO model → List[Detection] with bbox [x1,y1,x2,y2], confidence, class_id.<br>
            Models: YOLOv8n (speed) · YOLOv8m (balanced) · RT-DETR-L (accuracy)</p>
        </div>
        <div class="pipeline-step">
            <h4>📡 3. Tracking — KalmanTracker</h4>
            <p>SORT-style multi-object tracker: 7-state Kalman filter (cx, cy, area, aspect_ratio + velocities)
            with Hungarian/IoU data association → List[Track] with track_id, bbox, velocity, age</p>
        </div>
        <div class="pipeline-step">
            <h4>🎯 4. Avoidance — GeometricPlanner</h4>
            <p>Projects each track's future position by time_horizon using velocity. Applies repulsive force
            if within safe_distance of frame center → (yaw_delta, altitude_delta) command</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown("##### Data Flow")
        st.code(
            """frame (np.ndarray)
  → BaseDetector.detect()
      → List[Detection]
        (bbox, confidence, class_id)

  → BaseTracker.update()
      → List[Track]
        (track_id, bbox, velocity, age)

  → BaseAvoidancePlanner.plan()
      → (yaw_delta, altitude_delta)""",
            language="text",
        )

        st.markdown("##### Class Scheme (5 coarse classes)")
        class_data = pd.DataFrame(
            {
                "Class": ["vehicle", "person", "static", "flying", "other"],
                "Source": [
                    "car, van, truck, bus",
                    "pedestrian, people",
                    "reserved (AirSim)",
                    "reserved (Bird-vs-Drone)",
                    "bicycle, tricycle, motor",
                ],
            }
        )
        st.dataframe(class_data, use_container_width=True, hide_index=True)

    # Tech stack
    st.markdown(
        '<div class="section-header">🛠 Technology Stack</div>', unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            """<div class="metric-card">
            <div class="metric-value" style="font-size:1.6rem;">🐍</div>
            <div class="metric-label">Python + PyTorch</div>
        </div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """<div class="metric-card">
            <div class="metric-value" style="font-size:1.6rem;">🔮</div>
            <div class="metric-label">Ultralytics YOLO</div>
        </div>""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """<div class="metric-card">
            <div class="metric-value" style="font-size:1.6rem;">📐</div>
            <div class="metric-label">Kalman + Hungarian</div>
        </div>""",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            """<div class="metric-card">
            <div class="metric-value" style="font-size:1.6rem;">📊</div>
            <div class="metric-label">VisDrone Dataset</div>
        </div>""",
            unsafe_allow_html=True,
        )


# ===================================================================
# PAGE: Training Results
# ===================================================================
elif page == "📊 Training Results":
    st.markdown(
        '<div class="hero-title" style="font-size:2.4rem;">Training Results Dashboard</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-sub">Phase 6 — YOLOv8n & YOLOv8m trained on VisDrone-DET (5 classes) · 50 epochs · Nvidia L4 GPU</div>',
        unsafe_allow_html=True,
    )

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            """<div class="metric-card">
            <div class="metric-value">59.2%</div>
            <div class="metric-label">Best mAP@0.5 (YOLOv8m)</div>
        </div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """<div class="metric-card">
            <div class="metric-value">32.3%</div>
            <div class="metric-label">Best mAP@0.5:0.95</div>
        </div>""",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """<div class="metric-card">
            <div class="metric-value">+25%</div>
            <div class="metric-label">mAP Gain (n→m)</div>
        </div>""",
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            """<div class="metric-card">
            <div class="metric-value">2/3</div>
            <div class="metric-label">Models Trained</div>
        </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("")

    # Model comparison table
    st.markdown(
        '<div class="section-header">📊 Model Comparison</div>', unsafe_allow_html=True
    )

    df = pd.DataFrame(TRAINING_RESULTS)
    # Style the dataframe
    st.dataframe(
        df.style.format(
            {
                "mAP@0.5": lambda x: f"{x:.1%}" if x is not None else "—",
                "mAP@0.5:0.95": lambda x: f"{x:.1%}" if x is not None else "—",
                "Vehicle AP": lambda x: f"{x:.1%}" if x is not None else "—",
                "Person AP": lambda x: f"{x:.1%}" if x is not None else "—",
                "Other AP": lambda x: f"{x:.1%}" if x is not None else "—",
                "Inference (ms)": lambda x: f"{x:.1f}" if x is not None else "—",
                "~FPS": lambda x: f"{x:,.0f}" if x is not None else "—",
                "Params (M)": lambda x: f"{x:.1f}" if x is not None else "—",
            }
        ).apply(
            lambda row: [
                (
                    "background-color: rgba(102,126,234,0.1)"
                    if row["Status"] == "✅ Done"
                    else ""
                )
            ]
            * len(row),
            axis=1,
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("")

    # Charts
    st.markdown(
        '<div class="section-header">📈 Detailed Analysis</div>', unsafe_allow_html=True
    )

    tab1, tab2, tab3 = st.tabs(["mAP Comparison", "Per-Class AP", "Speed vs Accuracy"])

    with tab1:
        chart_data = pd.DataFrame(
            {
                "Model": ["YOLOv8n", "YOLOv8m"],
                "mAP@0.5": [47.4, 59.2],
                "mAP@0.5:0.95": [24.4, 32.3],
            }
        )
        st.bar_chart(chart_data.set_index("Model"), color=["#667eea", "#764ba2"])

        st.markdown("""
        > **Key Finding:** Scaling from YOLOv8n (3.2M params) to YOLOv8m (25.9M params) yields
        > a **+11.8 pp** gain in mAP@0.5 — demonstrating that model capacity significantly
        > helps on the VisDrone drone-perspective data with its small, dense objects.
        """)

    with tab2:
        per_class = pd.DataFrame(
            {
                "Class": ["Vehicle", "Person", "Other"],
                "YOLOv8n": [74.5, 32.9, 34.9],
                "YOLOv8m": [82.2, 46.3, 49.1],
            }
        )
        st.bar_chart(per_class.set_index("Class"), color=["#667eea", "#764ba2"])

        st.markdown("""
        > **Per-class insights:**
        > - **Vehicle** is the easiest class (74-82% AP) — large, frequent, and distinctive
        > - **Person** benefits most from scale (+13.4 pp) — small objects need deeper features
        > - **Other** (bicycle, tricycle, motor) also improves significantly (+14.2 pp)
        """)

    with tab3:
        speed_data = pd.DataFrame(
            {
                "Model": ["YOLOv8n", "YOLOv8m", "RT-DETR-L (est.)"],
                "mAP@0.5 (%)": [47.4, 59.2, 65.0],
                "FPS": [1250, 150, 30],
                "Params (M)": [3.2, 25.9, 32.0],
            }
        )
        st.scatter_chart(
            speed_data,
            x="FPS",
            y="mAP@0.5 (%)",
            size="Params (M)",
            color="Model",
        )

        st.markdown("""
        > **Speed vs Accuracy tradeoff:**
        > - YOLOv8n is extremely fast but sacrifices accuracy
        > - YOLOv8m hits the sweet spot for our FPS≥30 requirement
        > - RT-DETR-L (estimated) pushes accuracy further but at ~30 FPS limit
        >
        > **Selection rule:** highest mAP@0.5 subject to FPS ≥ 30 → **YOLOv8m wins**
        """)

    # Training details
    st.markdown(
        '<div class="section-header">⚙️ Training Configuration</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Hyperparameters")
        config_data = pd.DataFrame(
            {
                "Parameter": [
                    "Dataset",
                    "Classes",
                    "Image Size",
                    "Epochs",
                    "Optimizer",
                    "Seed",
                    "Split",
                    "GPU",
                ],
                "Value": [
                    "VisDrone-DET",
                    "5 (vehicle, person, static, flying, other)",
                    "640 × 640",
                    "50",
                    "SGD (Ultralytics default)",
                    "42",
                    "70 / 15 / 15 stratified",
                    "Nvidia L4 (24 GB)",
                ],
            }
        )
        st.dataframe(config_data, use_container_width=True, hide_index=True)

    with c2:
        st.markdown("##### Batch Sizes (tuned for L4 24 GB)")
        batch_data = pd.DataFrame(
            {
                "Model": ["YOLOv8n", "YOLOv8m", "RT-DETR-L"],
                "Batch Size": [128, 64, 20],
                "~VRAM Usage": ["~8 GB", "~16 GB", "~22 GB"],
            }
        )
        st.dataframe(batch_data, use_container_width=True, hide_index=True)


# ===================================================================
# PAGE: Live Demo
# ===================================================================
elif page == "🎯 Live Demo":
    st.markdown(
        '<div class="hero-title" style="font-size:2.4rem;">Live Detection Demo</div>',
        unsafe_allow_html=True,
    )

    # Check for available weights
    available_weights = find_available_weights()

    if not available_weights:
        st.markdown(
            '<div class="hero-sub">Upload an image or video to run detection — or provide model weights.</div>',
            unsafe_allow_html=True,
        )

        st.warning(
            "⚠️ **No model weights found** in `models/` directory.\n\n"
            "To enable live inference, either:\n"
            "1. Download trained weights: `modal run modal_train.py::fetch --model yolov8m`\n"
            "2. Place any `.pt` weights file in the `models/` folder\n"
            "3. The app will use the default pretrained YOLOv8m (COCO classes) as fallback\n\n"
            "For now, you can still upload images/videos and the app will attempt to use "
            "the default `yolov8m.pt` pretrained model."
        )

        use_default = st.checkbox(
            "🔄 Use default pretrained YOLOv8m (COCO, 80 classes)", value=True
        )
        if use_default:
            weights_choice = "yolov8m.pt"
        else:
            weights_choice = None
    else:
        st.markdown(
            '<div class="hero-sub">Upload an image or video to run the full pipeline (detect → track → avoid).</div>',
            unsafe_allow_html=True,
        )
        weights_choice = st.selectbox("Select model weights:", available_weights)

    st.markdown("---")

    # Upload section
    upload_tab1, upload_tab2 = st.tabs(["🖼️ Image Upload", "🎬 Video Upload"])

    with upload_tab1:
        uploaded_img = st.file_uploader(
            "Upload an image", type=["jpg", "jpeg", "png", "bmp"], key="img_upload"
        )

        if uploaded_img and weights_choice:
            with st.spinner("Loading model & running inference..."):
                model = load_model(weights_choice)
                if model is None:
                    # Try downloading default
                    try:
                        from ultralytics import YOLO

                        model = YOLO(weights_choice)
                    except Exception as e:
                        st.error(f"Failed to load model: {e}")
                        model = None

                if model is not None:
                    # Read image
                    file_bytes = np.asarray(
                        bytearray(uploaded_img.read()), dtype=np.uint8
                    )
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                    if img is not None:
                        # Run detection
                        detections, raw_results = run_inference_on_frame(model, img)

                        # Try running full pipeline
                        try:
                            from src.tracking.kalman_tracker import KalmanTracker
                            from src.risk.zone_assessor import RiskZoneAssessor

                            h, w = img.shape[:2]
                            tracker = KalmanTracker(min_hits=1)
                            assessor = RiskZoneAssessor()
                            tracks = tracker.update(detections)
                            risked = assessor.assess(tracks, frame_shape=(h, w))
                            avoidance_cmd = risked
                        except Exception:
                            avoidance_cmd = None

                        # Draw results
                        annotated = draw_detections_on_frame(
                            img, detections, avoidance_cmd
                        )
                        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                        # Display
                        col_orig, col_det = st.columns(2)
                        with col_orig:
                            st.markdown("##### Original")
                            st.image(
                                cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                use_container_width=True,
                            )
                        with col_det:
                            st.markdown("##### Detection + Tracking + Avoidance")
                            st.image(annotated_rgb, use_container_width=True)

                        # Stats
                        st.markdown("---")
                        sc1, sc2, sc3 = st.columns(3)
                        with sc1:
                            st.metric("Objects Detected", len(detections))
                        with sc2:
                            n_tracks = len(avoidance_cmd) if avoidance_cmd else 0
                            st.metric("Tracks", n_tracks)
                        with sc3:
                            if avoidance_cmd:
                                danger_count = sum(
                                    1
                                    for r in avoidance_cmd
                                    if getattr(r, "risk", None) == "DANGER"
                                )
                                st.metric("DANGER tracks", danger_count)
                            else:
                                st.metric("DANGER tracks", "N/A")

                        # Detection details
                        if detections:
                            with st.expander("📋 Detection Details", expanded=False):
                                det_data = pd.DataFrame(
                                    [
                                        {
                                            "Class": d.class_name,
                                            "Confidence": f"{d.confidence:.2%}",
                                            "BBox": f"[{d.bbox[0]:.0f}, {d.bbox[1]:.0f}, {d.bbox[2]:.0f}, {d.bbox[3]:.0f}]",
                                        }
                                        for d in detections
                                    ]
                                )
                                st.dataframe(
                                    det_data, use_container_width=True, hide_index=True
                                )

    with upload_tab2:
        uploaded_vid = st.file_uploader(
            "Upload a video", type=["mp4", "avi", "mov", "mkv"], key="vid_upload"
        )

        if uploaded_vid and weights_choice:
            with st.spinner("Processing video... This may take a while."):
                model_v = load_model(weights_choice)
                if model_v is None:
                    try:
                        from ultralytics import YOLO

                        model_v = YOLO(weights_choice)
                    except Exception as e:
                        st.error(f"Failed to load model: {e}")
                        model_v = None

                if model_v is not None:
                    # Save uploaded video to temp file
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    tfile.write(uploaded_vid.read())
                    tfile.flush()

                    cap = cv2.VideoCapture(tfile.name)
                    if not cap.isOpened():
                        st.error("Could not open video file.")
                    else:
                        fps = cap.get(cv2.CAP_PROP_FPS) or 30
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        w_vid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                        # Guard: CAP_PROP_FRAME_COUNT may return 0 for
                        # streamed / short videos, which would crash the
                        # slider (max_value < min_value).
                        if total_frames < 10:
                            total_frames = 10

                        st.info(
                            f"📹 Video: {w_vid}×{h_vid} @ {fps:.0f}fps · {total_frames} frames"
                        )

                        max_frames = st.slider(
                            "Max frames to process",
                            min_value=10,
                            max_value=min(total_frames, 500),
                            value=min(100, total_frames),
                            step=10,
                        )

                        if st.button("▶️ Run Detection", type="primary"):
                            try:
                                from src.tracking.kalman_tracker import KalmanTracker
                                from src.risk.zone_assessor import RiskZoneAssessor

                                tracker_v = KalmanTracker()
                                assessor_v = RiskZoneAssessor()
                                use_full_pipeline = True
                            except Exception:
                                use_full_pipeline = False

                            # Process frames
                            out_path = tempfile.NamedTemporaryFile(
                                delete=False, suffix=".mp4"
                            ).name
                            # Use mp4v — avc1 (H.264) is unavailable in
                            # the stock opencv-python wheel and silently
                            # produces empty files.
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            out_writer = cv2.VideoWriter(
                                out_path, fourcc, fps, (w_vid, h_vid)
                            )

                            progress = st.progress(0, text="Processing frames...")
                            frame_count = 0

                            while frame_count < max_frames:
                                ret, frame = cap.read()
                                if not ret:
                                    break

                                dets, _ = run_inference_on_frame(model_v, frame)

                                avoidance_c = None
                                if use_full_pipeline:
                                    tracks_v = tracker_v.update(dets)
                                    avoidance_c = assessor_v.assess(
                                        tracks_v,
                                        frame_shape=(h_vid, w_vid),
                                    )

                                annotated_v = draw_detections_on_frame(
                                    frame, dets, avoidance_c
                                )
                                out_writer.write(annotated_v)
                                frame_count += 1
                                progress.progress(
                                    frame_count / max_frames,
                                    text=f"Processing frame {frame_count}/{max_frames}...",
                                )

                            out_writer.release()
                            cap.release()
                            progress.empty()

                            st.success(f"✅ Processed {frame_count} frames!")

                            # Display video
                            with open(out_path, "rb") as vf:
                                st.video(vf.read())


# ===================================================================
# PAGE: Pipeline Status
# ===================================================================
elif page == "📋 Pipeline Status":
    st.markdown(
        '<div class="hero-title" style="font-size:2.4rem;">Pipeline Status</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-sub">Project progress tracker — RT-DR-003 Drone Obstacle Avoidance</div>',
        unsafe_allow_html=True,
    )

    # Status overview
    phases = [
        ("Phase 0 — Environment", "✅", "done", "venv + requirements.txt configured"),
        (
            "Phase 1 — Raw Data",
            "✅",
            "done",
            "VisDrone-DET: 8,629 labeled images (YOLO-format)",
        ),
        (
            "Phase 2 — Validate Raw",
            "✅",
            "done",
            "All splits PASS; fixed 1 zero-height bbox in train",
        ),
        (
            "Phase 3 — Preprocess",
            "✅",
            "done",
            "10→5 class remap; 6040/1294/1295 stratified split",
        ),
        (
            "Phase 4 — Configs",
            "✅",
            "done",
            "visdrone5.yaml + yolov8n/m/rtdetr.yaml; tuned for L4 24 GB",
        ),
        (
            "Phase 5 — Sanity Run",
            "✅",
            "done",
            "YOLOv8n 100 epochs on Modal L4; mAP@0.5 = 47.4% — pipeline healthy",
        ),
        (
            "Phase 6 — Full Training",
            "🔄",
            "progress",
            "YOLOv8n ✅ (47.4%) · YOLOv8m ✅ (59.2%) · RT-DETR-L ⏸ paused (VRAM constraints)",
        ),
        (
            "Phase 7 — Evaluation",
            "⬜",
            "pending",
            "scripts/evaluate.py ready; awaiting Phase 6 completion",
        ),
        (
            "Phase 8 — Integration",
            "⬜",
            "pending",
            "Wire best.pt into main.py + AirSim closed-loop demo for R4",
        ),
    ]

    for name, icon, status, desc in phases:
        badge_class = f"badge-{status}"
        st.markdown(
            f"""
        <div class="pipeline-step">
            <h4>{icon} {name} <span class="{badge_class}">{status.upper()}</span></h4>
            <p>{desc}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # R-Round mapping
    st.markdown("")
    st.markdown(
        '<div class="section-header">📅 Submission Round Mapping</div>',
        unsafe_allow_html=True,
    )

    round_data = pd.DataFrame(
        {
            "Round": ["R1", "R2", "R3", "R4"],
            "Content": [
                "Phases 0-4 designed, data validation report",
                "Phases 0-5 executed, one model trained (YOLOv8n)",
                "Phases 6-7, full 3-model comparison table",
                "Phase 8, live demo with best model",
            ],
            "Status": ["✅ Submitted", "✅ Submitted", "🔄 In Progress", "⬜ Upcoming"],
        }
    )
    st.dataframe(round_data, use_container_width=True, hide_index=True)

    # Open issues
    st.markdown(
        '<div class="section-header">⚠️ Open Issues</div>', unsafe_allow_html=True
    )

    st.markdown(
        """
    <div class="pipeline-step">
        <h4>🔴 RT-DETR-L Training Paused</h4>
        <p>VRAM constraints at batch=4 on L4 (24 GB). Needs L40S/A100 to run at batch=16.
        Options: reduce batch further, use gradient accumulation, or switch to A100 instance.</p>
    </div>
    <div class="pipeline-step">
        <h4>🟡 Weights Download</h4>
        <p>Trained weights are stored on Modal volume <code>cpv-data</code>. Need to fetch locally
        for web demo: <code>modal run modal_train.py::fetch --model yolov8m</code></p>
    </div>
    """,
        unsafe_allow_html=True,
    )
