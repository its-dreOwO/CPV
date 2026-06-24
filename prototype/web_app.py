"""
CPV301 Vehicle & Pedestrian Risk Advisory — Web Prototype
=========================================================
Streamlit showcase for the ADAS Forward-Collision-Warning perception system:
  1. Project overview & architecture
  2. Training results dashboard (model comparison)
  3. Live demo (upload image/video → detect → track → risk advisory overlay)

Runs a monocular dashcam perception pipeline (detection → tracking → risk
assessment). This is a perception + risk-reasoning system, NOT a vehicle
control loop — no steering or braking output is produced.

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

# Import RiskLevel constants for risk comparisons.  Guarded so the app can
# still load (and fall back gracefully) if src is not yet importable.
try:
    from src.risk.assessor import RiskLevel
except Exception:
    # Fallback shim keeps bare-string comparisons working without src on path.
    class RiskLevel:  # type: ignore[no-redef]
        SAFE = "SAFE"
        CAUTION = "CAUTION"
        DANGER = "DANGER"


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CPV301 — Vehicle & Pedestrian Risk Advisory (FCW)",
    page_icon="🚗",
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
# Training results — pending BDD100K training (Plans 2-3)
# ---------------------------------------------------------------------------
CLASS_NAMES = ["vehicle", "person", "two_wheeler"]


def find_available_weights():
    """Scan for project-trained *-best.pt weights in models/, then any .pt fallback."""
    candidates = []
    models_dir = Path("models")
    # Prefer our trained checkpoint names first
    preferred = ["yolov8n-best.pt", "yolov8m-best.pt", "yolov10n-best.pt"]
    if models_dir.exists():
        for name in preferred:
            p = models_dir / name
            if p.exists():
                candidates.append(str(p))
        # Catch any other .pt in models/ not already listed
        for p in sorted(models_dir.glob("*.pt")):
            if str(p) not in candidates:
                candidates.append(str(p))
    # Also check for weights in project root
    for p in sorted(Path(".").glob("*.pt")):
        if str(p) not in candidates:
            candidates.append(str(p))
    return candidates


@st.cache_resource
def load_yolo_detector(weights_path: str):
    """Load a YoloDetector wrapping the given weights file."""
    from src.detection.yolo_detector import YoloDetector

    return YoloDetector(model_path=weights_path)


def run_inference_on_frame(detector, frame: np.ndarray):
    """Run YoloDetector detection on a single frame.

    Returns List[Detection] using the real src pipeline interface.
    """
    detections = detector.detect(frame)
    return detections


def draw_pipeline_frame(
    frame: np.ndarray,
    detections,
    risked_tracks=None,
    ego_polygon=None,
):
    """Draw the full risk-advisory overlay on a frame.

    When risked_tracks and ego_polygon are supplied (full pipeline), uses
    src.utils.visualizer.draw_risk for the ego-path trapezoid + color-coded
    SAFE/CAUTION/DANGER boxes.  Falls back to plain detection boxes when the
    pipeline is unavailable (no weights / import failure).
    """
    from src.utils.visualizer import draw_risk

    if risked_tracks is not None and ego_polygon is not None:
        # Full pipeline overlay: ego-path trapezoid + risk-colored boxes
        out = draw_risk(frame, risked_tracks, ego_polygon=ego_polygon)
        # Overlay raw-detection confidence labels (thin, grey)
        colors = [
            (78, 126, 234),
            (162, 75, 118),
            (45, 183, 147),
        ]
        for d in detections:
            x1, y1, x2, y2 = [int(v) for v in d.bbox]
            color = colors[d.class_id % len(colors)]
            label = f"{d.class_name} {d.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1)
            cv2.putText(
                out,
                label,
                (x1 + 1, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        # HUD: risk summary
        if risked_tracks:
            danger_n = sum(
                1 for r in risked_tracks if getattr(r, "risk", None) == RiskLevel.DANGER
            )
            caution_n = sum(
                1
                for r in risked_tracks
                if getattr(r, "risk", None) == RiskLevel.CAUTION
            )
            hud = (
                f"DANGER:{danger_n}  CAUTION:{caution_n}"
                f"  TRACKS:{len(risked_tracks)}"
            )
            cv2.putText(
                out,
                hud,
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 100, 255),
                2,
                cv2.LINE_AA,
            )
        return out

    # Fallback: plain detection boxes (no risk pipeline)
    out = frame.copy()
    colors = [
        (78, 126, 234),
        (162, 75, 118),
        (45, 183, 147),
    ]
    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d.bbox]
        color = colors[d.class_id % len(colors)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{d.class_name} {d.confidence:.2f}"
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
    return out


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("# 🚗 CPV301")
    st.markdown("### Vehicle & Pedestrian Risk Advisory")
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
        <b>Risk-Advisory Perception</b><br>
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
    <div class="hero-title">Vehicle & Pedestrian Risk Advisory</div>
    <div class="hero-sub">
        Forward-Collision-Warning perception for dashcam footage (monocular, no control loop)<br>
        Three-stage Computer Vision pipeline: <b>Detection → Tracking → Risk Assessment</b>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            """<div class="metric-card">
            <div class="metric-value">3</div>
            <div class="metric-label">Risk Levels</div>
        </div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """<div class="metric-card">
            <div class="metric-value">BDD100K</div>
            <div class="metric-label">Primary Dataset</div>
        </div>""",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """<div class="metric-card">
            <div class="metric-value">3</div>
            <div class="metric-label">Object Classes</div>
        </div>""",
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            """<div class="metric-card">
            <div class="metric-value">KITTI</div>
            <div class="metric-label">Cross-Dataset Test</div>
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
            Models: YOLOv8n (speed) · YOLOv8m (balanced) · YOLOv10n (NMS-free)</p>
        </div>
        <div class="pipeline-step">
            <h4>📡 3. Tracking — KalmanTracker</h4>
            <p>SORT-style multi-object tracker: 7-state Kalman filter (cx, cy, area, aspect_ratio + velocities)
            with Hungarian/IoU data association → List[Track] with track_id, bbox, velocity, age</p>
        </div>
        <div class="pipeline-step">
            <h4>🎯 4. Risk — RiskZoneAssessor</h4>
            <p>Projects an ego-path trapezoid (narrow at horizon, wide at frame bottom). An object is
            in-path if its bbox bottom-center falls inside it and closing if its bbox area is large or
            growing → tags each track SAFE / CAUTION / DANGER</p>
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
        (track_id, bbox, velocity, scale_velocity)

  → BaseRiskAssessor.assess()
      → List[RiskedTrack]
        (SAFE / CAUTION / DANGER)""",
            language="text",
        )

        st.markdown("##### Class Scheme (3 coarse classes)")
        class_data = pd.DataFrame(
            {
                "Class": ["vehicle", "person", "two_wheeler"],
                "Source (BDD100K)": [
                    "car, truck, bus, train",
                    "pedestrian, rider",
                    "bicycle, motorcycle",
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
            <div class="metric-label">BDD100K Dataset</div>
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
        '<div class="hero-sub">Model comparison on BDD100K — pending training (Plans 2-3)</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "🔧 **Pending.** The project pivoted to a vehicle & pedestrian "
        "risk-advisory system on **BDD100K**. The 3-model comparison "
        "(YOLOv8n / YOLOv8m / YOLOv10n) trains on BDD100K in the data "
        "pipeline (Plan 2) and training/evaluation phase (Plan 3). Results — "
        "mAP@0.5, per-class AP, FPS, plus the **KITTI cross-dataset "
        "generalization** and **day/night robustness** findings — populate "
        "this dashboard once those runs complete."
    )

    st.markdown(
        '<div class="section-header">📊 Planned Model Lineup</div>',
        unsafe_allow_html=True,
    )
    lineup = pd.DataFrame(
        {
            "Model": ["YOLOv8n", "YOLOv8m", "YOLOv10n"],
            "Role": [
                "Speed baseline / embedded floor",
                "Primary R4 demo model",
                "NMS-free end-to-end / architecture contrast",
            ],
            "Axis": [
                "capacity (n→m)",
                "capacity / arch pivot",
                "version/paradigm (v8n→v10n)",
            ],
            "Status": ["⬜ Pending", "⬜ Pending", "⬜ Pending"],
        }
    )
    st.dataframe(lineup, use_container_width=True, hide_index=True)


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
            "To enable live inference, train the models (Plan 3) and place the\n"
            "checkpoint files in `models/` with the expected names:\n"
            "- `models/yolov8n-best.pt` — speed baseline\n"
            "- `models/yolov8m-best.pt` — primary R4 demo model\n"
            "- `models/yolov10n-best.pt` — NMS-free end-to-end contrast\n\n"
            "As a fallback, tick the box below to use a generic pretrained YOLOv8m "
            "(COCO 80-class) — note: class names will not match the 3-class scheme."
        )

        use_default = st.checkbox(
            "🔄 Use fallback pretrained YOLOv8m (COCO, 80 classes)", value=True
        )
        if use_default:
            weights_choice = "yolov8m.pt"
        else:
            weights_choice = None
    else:
        st.markdown(
            '<div class="hero-sub">Upload an image or video to run the full pipeline'
            " (detect → track → risk advisory).</div>",
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
                try:
                    detector = load_yolo_detector(weights_choice)
                except Exception as e:
                    st.error(f"Failed to load detector: {e}")
                    detector = None

                if detector is not None:
                    # Read image
                    file_bytes = np.asarray(
                        bytearray(uploaded_img.read()), dtype=np.uint8
                    )
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                    if img is not None:
                        # Run detection via src pipeline
                        detections = run_inference_on_frame(detector, img)

                        # Full pipeline: tracking + risk assessment + ego-path
                        try:
                            from src.tracking.kalman_tracker import KalmanTracker
                            from src.risk.zone_assessor import RiskZoneAssessor

                            h, w = img.shape[:2]
                            tracker = KalmanTracker(min_hits=1)
                            assessor = RiskZoneAssessor()
                            tracks = tracker.update(detections)
                            risked_tracks = assessor.assess(tracks, frame_shape=(h, w))
                            ego_poly = assessor.ego_path_polygon((h, w))
                        except Exception as e:
                            risked_tracks = None
                            ego_poly = None
                            st.warning(
                                f"Risk pipeline unavailable — showing detection only."
                                f" ({e})"
                            )

                        # Draw results with ego-path overlay + risk colors
                        annotated = draw_pipeline_frame(
                            img, detections, risked_tracks, ego_poly
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
                            st.markdown("##### Detection + Tracking + Risk Advisory")
                            st.image(annotated_rgb, use_container_width=True)

                        # Stats
                        st.markdown("---")
                        sc1, sc2, sc3 = st.columns(3)
                        with sc1:
                            st.metric("Objects Detected", len(detections))
                        with sc2:
                            n_tracks = len(risked_tracks) if risked_tracks else 0
                            st.metric("Tracks", n_tracks)
                        with sc3:
                            if risked_tracks:
                                danger_count = sum(
                                    1
                                    for r in risked_tracks
                                    if getattr(r, "risk", None) == RiskLevel.DANGER
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
                                            "BBox": (
                                                f"[{d.bbox[0]:.0f}, {d.bbox[1]:.0f},"
                                                f" {d.bbox[2]:.0f}, {d.bbox[3]:.0f}]"
                                            ),
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
                try:
                    detector_v = load_yolo_detector(weights_choice)
                except Exception as e:
                    st.error(f"Failed to load detector: {e}")
                    detector_v = None

                if detector_v is not None:
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
                            f"📹 Video: {w_vid}×{h_vid} @ {fps:.0f}fps"
                            f" · {total_frames} frames"
                        )

                        max_frames = st.slider(
                            "Max frames to process",
                            min_value=10,
                            max_value=min(total_frames, 500),
                            value=min(100, total_frames),
                            step=10,
                        )

                        if st.button("▶️ Run Risk Advisory Pipeline", type="primary"):
                            try:
                                from src.tracking.kalman_tracker import KalmanTracker
                                from src.risk.zone_assessor import RiskZoneAssessor

                                tracker_v = KalmanTracker()
                                assessor_v = RiskZoneAssessor()
                                use_full_pipeline = True
                            except Exception as e:
                                assessor_v = None
                                use_full_pipeline = False
                                st.warning(
                                    f"Risk pipeline unavailable — showing detection only."
                                    f" ({e})"
                                )

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

                                dets = run_inference_on_frame(detector_v, frame)

                                risked_v = None
                                ego_poly_v = None
                                if use_full_pipeline:
                                    tracks_v = tracker_v.update(dets)
                                    risked_v = assessor_v.assess(
                                        tracks_v,
                                        frame_shape=(h_vid, w_vid),
                                    )
                                    ego_poly_v = assessor_v.ego_path_polygon(
                                        (h_vid, w_vid)
                                    )

                                annotated_v = draw_pipeline_frame(
                                    frame, dets, risked_v, ego_poly_v
                                )
                                out_writer.write(annotated_v)
                                frame_count += 1
                                progress.progress(
                                    frame_count / max_frames,
                                    text=(
                                        f"Processing frame"
                                        f" {frame_count}/{max_frames}..."
                                    ),
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
        '<div class="hero-sub">Project progress — Vehicle & Pedestrian Risk Advisory pivot</div>',
        unsafe_allow_html=True,
    )

    phases = [
        (
            "Cleanup — drone artifacts removed",
            "✅",
            "done",
            "Removed VisDrone data, drone demo videos, drone configs; git gc reclaimed ~3.3 GB",
        ),
        (
            "Design spec",
            "✅",
            "done",
            "Dashcam risk-advisory on BDD100K (3 classes) + KITTI hold-out; approved & committed",
        ),
        (
            "Plan 1 — Risk-assessor code pivot",
            "✅",
            "done",
            "src/avoidance → src/risk (RiskZoneAssessor); detect→track→risk wired; 26/26 tests",
        ),
        (
            "Plan 2 — BDD100K data pipeline",
            "⬜",
            "pending",
            "Download, JSON→YOLO conversion, 3-class remap, stratified split, KITTI prep",
        ),
        (
            "Plan 3 — Training + eval + docs",
            "⬜",
            "pending",
            "3-model comparison on BDD100K, KITTI generalization, risk validation, Streamlit re-skin",
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
                "Problem statement + design + BDD100K data-validation report",
                "Preprocess + YOLOv8n trained on BDD100K + initial risk-overlay demo",
                "3-model comparison + KITTI generalization + risk validation + day/night robustness",
                "Best model wired into live dashcam demo + end-to-end FPS + Streamlit showcase",
            ],
            "Status": ["🔄 In Progress", "⬜ Upcoming", "⬜ Upcoming", "⬜ Upcoming"],
        }
    )
    st.dataframe(round_data, use_container_width=True, hide_index=True)
