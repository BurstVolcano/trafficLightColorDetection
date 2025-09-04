import streamlit as st
import cv2
import tempfile
import time
from traffic_detector import TrafficLightDetector

st.set_page_config(page_title="Traffic Light Detection", layout="wide")

class StreamlitUI:
    def __init__(self):
        self.detector = self._load_detector()

    @st.cache_resource
    def _load_detector(_self):
        return TrafficLightDetector(confidence_threshold=0.3)

    def render_sidebar(self):
        confidence = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.3, 0.05)
        frame_skip = st.sidebar.selectbox("Process Every N Frames", [1, 2, 3, 5, 10], 2)
        return confidence, frame_skip

    def process_video(self, video_file, confidence: float, frame_skip: int):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_file.read())
            tmp_file_path = tmp_file.name

        self.detector.confidence_threshold = confidence
        cap = cv2.VideoCapture(tmp_file_path)
        if not cap.isOpened():
            st.error("Could not open video file")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_placeholder = st.empty()
        stats_placeholder = st.empty()
        progress_bar = st.progress(0)

        state_counts = {"Red": 0, "Yellow": 0, "Green": 0, "Unknown": 0}
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_skip == 0:
                detections = self.detector.detect_in_frame(frame)
                annotated_frame = self.detector.draw_detections(frame, detections)
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                for d in detections:
                    state_counts[d.state] += 1
                stats_placeholder.markdown(f"Frame {frame_count+1}/{total_frames} | {state_counts}")
            frame_count += 1
            progress_bar.progress(frame_count / total_frames)
            time.sleep(1 / max(fps, 10))
        cap.release()
        st.success("Processing completed!")

    def run(self):
        st.title("Traffic Light Detection")
        confidence, frame_skip = self.render_sidebar()
        uploaded_file = st.file_uploader("Upload a video", type=['mp4','avi','mov','mkv','webm'])
        if uploaded_file and st.button("Process Video", type="primary"):
            with st.spinner("Processing video..."):
                self.process_video(uploaded_file, confidence, frame_skip)

if __name__ == "__main__":
    app = StreamlitUI()
    app.run()
