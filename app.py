import io
import os
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from ultralytics import YOLO

from src.ui_helpers import (
    COLOR_BGR,
    annotate_frame,
    convert_video_to_h264,
    get_available_models,
    get_project_root,
)

st.set_page_config(
    page_title="HelmetGuard AI - Helmet Compliance Detection",
    page_icon="🪖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Global Styling */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.1rem;
        font-weight: 700;
        color: #f8fafc;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .main-header p {
        margin: 0.4rem 0 0 0;
        color: #94a3b8;
        font-size: 1rem;
    }
    
    /* Metrics Card Styling */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.3);
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 0.3rem;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f1f5f9;
    }
    .metric-val.green { color: #10b981; }
    .metric-val.red { color: #ef4444; }
    .metric-val.blue { color: #3b82f6; }
    .metric-val.amber { color: #f59e0b; }

    /* Badges */
    .status-badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .status-badge.success {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-badge.danger {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .status-badge.warning {
        background-color: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    /* St tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        border-radius: 8px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading YOLO Detection Model...")
def get_cached_model(model_path_str: str) -> YOLO:
    return YOLO(model_path_str)


def render_header():
    st.markdown(
        """
        <div class="main-header">
            <h1>🪖 HelmetGuard AI &bull; Compliance Vision</h1>
            <p>Real-time automated bicycle & motorcycle helmet detection powered by fine-tuned Ultralytics YOLOv8-nano.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    st.sidebar.title("⚙️ Model & Config")
    models_dict = get_available_models()

    if not models_dict:
        st.sidebar.error("No model checkpoints found in `models/` or `model/`.")
        model_path = None
    else:
        model_names = list(models_dict.keys())
        selected_model_name = st.sidebar.selectbox(
            "Select Model Checkpoint",
            options=model_names,
            index=0,
            help="Choose the model weights file to perform detection.",
        )
        model_path = models_dict[selected_model_name]
        st.sidebar.caption(f"Path: `{model_path.name}`")

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Detection Parameters")

    conf_thresh = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=1.00,
        value=0.25,
        step=0.05,
        help="Minimum confidence score required to detect a helmet / rider.",
    )

    iou_thresh = st.sidebar.slider(
        "IoU / NMS Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05,
        help="Non-maximum suppression threshold to remove overlapping bounding boxes.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎨 Visual Overlay")
    show_boxes = st.sidebar.checkbox("Show Bounding Boxes", value=True)
    show_labels = st.sidebar.checkbox("Show Class Labels & Scores", value=True)
    show_hud = st.sidebar.checkbox("Show HUD Metrics Banner", value=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🏷️ Target Classes")
    st.sidebar.markdown(
        """
        <div style="display:flex; flex-direction:column; gap:8px;">
            <div style="display:flex; align-items:center; gap:8px;">
                <span style="display:inline-block; width:14px; height:14px; background-color:#2ecc71; border-radius:3px;"></span>
                <span style="font-size:0.95rem; font-weight:500;">Class 0: <b>With Helmet</b> (Compliant)</span>
            </div>
            <div style="display:flex; align-items:center; gap:8px;">
                <span style="display:inline-block; width:14px; height:14px; background-color:#e74c3c; border-radius:3px;"></span>
                <span style="font-size:0.95rem; font-weight:500;">Class 1: <b>Without Helmet</b> (Violation)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return model_path, conf_thresh, iou_thresh, show_boxes, show_labels, show_hud


def render_metrics_cards(stats: dict):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Total Riders</div>
                <div class="metric-val blue">{stats['total']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">With Helmet</div>
                <div class="metric-val green">{stats['helmet_count']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Without Helmet</div>
                <div class="metric-val red">{stats['no_helmet_count']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        val_color = "green" if stats["compliance_rate"] >= 80 else ("amber" if stats["compliance_rate"] >= 50 else "red")
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Compliance Rate</div>
                <div class="metric-val {val_color}">{stats['compliance_rate']:.0f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def handle_photo_upload(model, conf, iou, show_boxes, show_labels, show_hud):
    st.markdown("### 📤 Upload Photo for Inspection")
    col_input, col_sample = st.columns([2, 1])

    with col_input:
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            help="Upload an image to detect helmet compliance.",
        )

    with col_sample:
        root = get_project_root()
        test_images_dir = root / "Data" / "test" / "images"
        sample_options = ["None (Upload Custom)"]
        sample_files = []
        if test_images_dir.is_dir():
            sample_files = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))
            sample_options += [f.name for f in sample_files[:15]]

        selected_sample = st.selectbox("Or choose a sample test image", sample_options)

    image_to_process = None
    image_name = "image.jpg"

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_to_process = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_name = uploaded_file.name
    elif selected_sample != "None (Upload Custom)" and sample_files:
        sample_path = next((f for f in sample_files if f.name == selected_sample), None)
        if sample_path:
            image_to_process = cv2.imread(str(sample_path))
            image_name = sample_path.name

    if image_to_process is not None:
        with st.spinner("Analyzing image..."):
            results = model(image_to_process, conf=conf, iou=iou, verbose=False)
            annotated_bgr, stats, detections = annotate_frame(
                image_to_process,
                results,
                show_boxes=show_boxes,
                show_labels=show_labels,
                show_hud=show_hud,
            )

        st.markdown("---")
        render_metrics_cards(stats)
        st.write("")

        col_orig, col_annot = st.columns(2)
        with col_orig:
            st.markdown("#### 📷 Original Image")
            orig_rgb = cv2.cvtColor(image_to_process, cv2.COLOR_BGR2RGB)
            st.image(orig_rgb, use_container_width=True)

        with col_annot:
            st.markdown("#### 🎯 Detection Results")
            annot_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            st.image(annot_rgb, use_container_width=True)

        # Download button
        _, buf = cv2.imencode(".jpg", annotated_bgr)
        st.download_button(
            label="💾 Download Annotated Image",
            data=buf.tobytes(),
            file_name=f"detected_{image_name}",
            mime="image/jpeg",
        )

        if detections:
            st.markdown("#### 📋 Detections Breakdown")
            df = pd.DataFrame(detections)[["Class", "Confidence", "BBox"]]
            st.dataframe(df, use_container_width=True)
    else:
        st.info("👆 Please upload an image or select a sample image above to begin.")


def handle_photo_live(model, conf, iou, show_boxes, show_labels, show_hud):
    st.markdown("### 📸 Live Camera Snapshot")
    st.caption("Capture a live photo using your webcam or phone camera to perform immediate helmet detection.")

    camera_image = st.camera_input("Take a snapshot")

    if camera_image is not None:
        bytes_data = camera_image.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        with st.spinner("Analyzing captured photo..."):
            results = model(cv2_img, conf=conf, iou=iou, verbose=False)
            annotated_bgr, stats, detections = annotate_frame(
                cv2_img,
                results,
                show_boxes=show_boxes,
                show_labels=show_labels,
                show_hud=show_hud,
            )

        st.markdown("---")
        render_metrics_cards(stats)
        st.write("")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📷 Snapshot")
            st.image(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        with col2:
            st.markdown("#### 🎯 Annotated Detection")
            annot_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            st.image(annot_rgb, use_container_width=True)

        _, buf = cv2.imencode(".jpg", annotated_bgr)
        st.download_button(
            label="💾 Download Snapshot Result",
            data=buf.tobytes(),
            file_name=f"live_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
            mime="image/jpeg",
        )

        if detections:
            st.markdown("#### 📋 Detections Breakdown")
            df = pd.DataFrame(detections)[["Class", "Confidence", "BBox"]]
            st.dataframe(df, use_container_width=True)


def handle_video_upload(model, conf, iou, show_boxes, show_labels, show_hud):
    st.markdown("### 🎬 Upload Video for Inspection")
    uploaded_video = st.file_uploader(
        "Upload a video file",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        help="Upload a video to analyze helmet compliance across all frames.",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        use_tracking = st.checkbox("Enable Object Tracking (ByteTrack)", value=True, help="Assigns unique IDs to riders.")
    with c2:
        frame_stride = st.slider("Frame Stride", min_value=1, max_value=5, value=1, help="Process every N-th frame for speed.")
    with c3:
        record_csv = st.checkbox("Log Violations to CSV", value=True)

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.flush()
        temp_input_path = tfile.name

        cap = cv2.VideoCapture(temp_input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        st.write(f"**Video Info:** {width}x{height} | {total_frames} frames | {orig_fps:.1f} FPS")

        if st.button("🚀 Process Video", type="primary"):
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            preview_slot = st.empty()
            metric_slot = st.empty()

            temp_output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_writer = cv2.VideoWriter(temp_output_path, fourcc, orig_fps / frame_stride, (width, height))

            frame_idx = 0
            processed_count = 0
            total_helmets = 0
            total_violations = 0
            violation_logs = []
            seen_tracks = set()

            start_time = time.time()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if frame_idx % frame_stride != 0:
                    continue

                loop_time = time.time()
                if use_tracking:
                    results = model.track(frame, conf=conf, iou=iou, persist=True, verbose=False)
                else:
                    results = model(frame, conf=conf, iou=iou, verbose=False)

                fps_inst = 1.0 / max(time.time() - loop_time, 1e-4)

                annotated_bgr, stats, detections = annotate_frame(
                    frame,
                    results,
                    show_boxes=show_boxes,
                    show_labels=show_labels,
                    show_hud=show_hud,
                    fps=fps_inst,
                )

                total_helmets += stats["helmet_count"]
                total_violations += stats["no_helmet_count"]

                # Log violations
                for det in detections:
                    if det["Class_ID"] == 1:
                        tid = det["Track ID"]
                        if tid not in seen_tracks or tid == "-":
                            seen_tracks.add(tid)
                            violation_logs.append({
                                "Timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
                                "Frame": frame_idx,
                                "Track ID": tid,
                                "Class": det["Class"],
                                "Confidence": det["Confidence"],
                                "BBox": det["BBox"],
                            })

                out_writer.write(annotated_bgr)
                processed_count += 1

                pct = min(frame_idx / max(total_frames, 1), 1.0)
                progress_bar.progress(pct)
                status_text.text(f"Processing Frame {frame_idx}/{total_frames} ({pct:.1%}) | Real-time FPS: {fps_inst:.1f}")

                if processed_count % 3 == 0:
                    preview_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                    preview_slot.image(preview_rgb, caption=f"Processing Preview (Frame {frame_idx})", use_container_width=True)

            cap.release()
            out_writer.release()
            total_time = time.time() - start_time

            status_text.success(f"Processing complete! Processed {processed_count} frames in {total_time:.1f}s ({processed_count / max(total_time, 1e-4):.1f} FPS).")

            # Transcode with ffmpeg for HTML5 browser playback
            h264_output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            with st.spinner("Preparing video for browser playback..."):
                transcoded = convert_video_to_h264(temp_output_path, h264_output_path)
                final_video_path = h264_output_path if transcoded else temp_output_path

            st.markdown("---")
            total_detections = total_helmets + total_violations
            comp_rate = (total_helmets / total_detections * 100) if total_detections > 0 else 0.0

            render_metrics_cards({
                "total": total_detections,
                "helmet_count": total_helmets,
                "no_helmet_count": total_violations,
                "compliance_rate": comp_rate,
            })

            st.markdown("#### 🎬 Processed Video Output")
            try:
                with open(final_video_path, "rb") as vf:
                    video_bytes = vf.read()
                st.video(video_bytes)
                st.download_button(
                    label="💾 Download Processed Video",
                    data=video_bytes,
                    file_name=f"detected_{uploaded_video.name}",
                    mime="video/mp4",
                )
            except Exception as e:
                st.warning(f"Could not load in-browser player: {e}")

            if violation_logs and record_csv:
                st.markdown("#### 🚨 Violation Events Log")
                df_violations = pd.DataFrame(violation_logs)
                st.dataframe(df_violations, use_container_width=True)
                csv_data = df_violations.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Violations Log (CSV)",
                    data=csv_data,
                    file_name=f"violations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )


def handle_video_live(model, conf, iou, show_boxes, show_labels, show_hud):
    st.markdown("### 📹 Live Webcam Stream")
    st.caption("Stream live video from your connected camera with real-time helmet detection HUD.")

    col_cam, col_ctrl = st.columns([1, 2])
    with col_cam:
        camera_idx = st.number_input("Camera Index", min_value=0, max_value=5, value=0, step=1, help="Usually 0 for built-in webcam.")
        use_tracking = st.checkbox("Enable Tracking in Live Stream", value=True)

    with col_ctrl:
        st.write("")
        st.write("")
        run_stream = st.toggle("🔴 Start Real-Time Stream", value=False)

    if run_stream:
        cap = cv2.VideoCapture(int(camera_idx))

        if not cap.isOpened():
            st.error(f"Could not open camera {camera_idx}. Make sure your camera is connected and not currently used by another program.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        stream_slot = st.empty()
        stats_slot = st.empty()
        stop_notice = st.info("Stream running. Toggle the switch off above to stop.")

        prev_time = time.time()

        try:
            while run_stream:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to grab camera frame.")
                    break

                curr_time = time.time()
                fps = 1.0 / max(curr_time - prev_time, 1e-4)
                prev_time = curr_time

                if use_tracking:
                    results = model.track(frame, conf=conf, iou=iou, persist=True, verbose=False)
                else:
                    results = model(frame, conf=conf, iou=iou, verbose=False)

                annotated_bgr, stats, _ = annotate_frame(
                    frame,
                    results,
                    show_boxes=show_boxes,
                    show_labels=show_labels,
                    show_hud=show_hud,
                    fps=fps,
                )

                stream_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                stream_slot.image(stream_rgb, use_container_width=True)

                with stats_slot.container():
                    render_metrics_cards(stats)

                time.sleep(0.01)

        finally:
            cap.release()
            stream_slot.empty()


def main():
    render_header()
    model_path, conf, iou, show_boxes, show_labels, show_hud = render_sidebar()

    if not model_path:
        st.stop()

    model = get_cached_model(str(model_path))

    tab_photo_up, tab_photo_live, tab_vid_up, tab_vid_live = st.tabs([
        "🖼️ Upload Photo",
        "📸 Live Photo (Camera)",
        "🎞️ Upload Video",
        "📹 Live Video (Webcam)",
    ])

    with tab_photo_up:
        handle_photo_upload(model, conf, iou, show_boxes, show_labels, show_hud)

    with tab_photo_live:
        handle_photo_live(model, conf, iou, show_boxes, show_labels, show_hud)

    with tab_vid_up:
        handle_video_upload(model, conf, iou, show_boxes, show_labels, show_hud)

    with tab_vid_live:
        handle_video_live(model, conf, iou, show_boxes, show_labels, show_hud)


if __name__ == "__main__":
    main()
