import streamlit as st
import cv2
import tempfile
import time
from traffic_detector import TrafficLightDetector, Detection
from typing import List
import numpy as np

# Page config
st.set_page_config(
    page_title="Traffic Light Detection",
    layout="wide"
)


class StreamlitUI:
    """Streamlit user interface for traffic light detection"""

    def __init__(self):
        self.detector = self._load_detector()

    @st.cache_resource
    def _load_detector(_self):
        """Load and cache the detection model"""
        return TrafficLightDetector(confidence_threshold=0.3)

    def render_header(self):
        """Render app header"""
        st.title("Traffic Light Detection System")
        st.markdown("Upload a video to detect traffic light states in real-time!")

    def render_sidebar(self):
        """Render sidebar with options"""
        st.sidebar.header("Settings")

        confidence = st.sidebar.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.3,
            step=0.05
        )

        frame_skip = st.sidebar.selectbox(
            "Process Every N Frames",
            options=[1, 2, 3, 5, 10],
            index=2
        )

        show_details = st.sidebar.checkbox("Show Detection Details", value=True)

        return confidence, frame_skip, show_details

    def render_file_uploader(self):
        """Render file upload widget"""
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
            help="Upload a video containing traffic lights"
        )
        return uploaded_file

    def process_video(self, video_file, confidence: float, frame_skip: int, show_details: bool):
        """Process uploaded video"""
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_file.read())
            tmp_file_path = tmp_file.name

        # Update detector confidence
        self.detector.confidence_threshold = confidence

        # Open video
        cap = cv2.VideoCapture(tmp_file_path)

        if not cap.isOpened():
            st.error("Error: Could not open video file")
            return

        # Get video info
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        st.success(f"Video loaded: {total_frames} frames at {fps:.1f} FPS ({width}x{height})")

        # Create layout
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Video Processing")
            frame_placeholder = st.empty()

        with col2:
            st.subheader("Detection Results")
            if show_details:
                stats_placeholder = st.empty()
                details_placeholder = st.empty()
            else:
                results_placeholder = st.empty()

        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Processing stats
        total_detections = 0
        state_counts = {"Red": 0, "Yellow": 0, "Green": 0, "Unknown": 0}
        frame_count = 0

        # Process video
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process every nth frame
                if frame_count % frame_skip == 0:
                    # Detect traffic lights
                    detections = self.detector.detect_in_frame(frame)

                    # Draw detections
                    annotated_frame = self.detector.draw_detections(frame, detections)

                    # Convert BGR to RGB for display
                    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                    # Update display
                    frame_placeholder.image(
                        frame_rgb,
                        channels="RGB",
                        caption=f"Frame {frame_count + 1}/{total_frames}",
                        use_column_width=True
                    )

                    # Update statistics
                    for detection in detections:
                        total_detections += 1
                        state_counts[detection.state] += 1

                    # Update results display
                    self._update_results_display(
                        detections,
                        show_details,
                        stats_placeholder if show_details else None,
                        details_placeholder if show_details else results_placeholder,
                        total_detections,
                        state_counts
                    )

                # Update progress
                frame_count += 1
                progress = frame_count / total_frames
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {frame_count}/{total_frames} ({progress:.1%})")

                # Small delay for smooth playback
                time.sleep(0.03)

        except Exception as e:
            st.error(f"Error during processing: {e}")

        finally:
            cap.release()
            status_text.text("Processing completed!")
            self._show_final_stats(total_detections, state_counts)

    def _update_results_display(self,
                                detections: List[Detection],
                                show_details: bool,
                                stats_placeholder,
                                details_placeholder,
                                total_detections: int,
                                state_counts: dict):
        """Update the results display"""
        if show_details and stats_placeholder and details_placeholder:
            # Show detailed stats
            stats_markdown = f"""
            ** Processing Statistics:**
            - **Total Detections:** {total_detections}
            - **Red Lights:** {state_counts['Red']} 
            - **Yellow Lights:** {state_counts['Yellow']}  
            - **Green Lights:** {state_counts['Green']} 
            - **Unknown:** {state_counts['Unknown']} 
            """
            stats_placeholder.markdown(stats_markdown)

            # Show current frame detections
            if detections:
                details_text = "**Current Frame Detections:**\n"
                for i, det in enumerate(detections):
                    details_text += f"{det.state}** (conf: {det.confidence:.2f})\n"
                    if det.zone_scores:
                        details_text += f"  - Zone scores: {det.zone_scores}\n"
                details_placeholder.markdown(details_text)
            else:
                details_placeholder.markdown("No traffic lights detected in current frame")

        else:
            # Show simple results
            if detections:
                results_text = "**Detected Traffic Lights:**\n"
                for det in detections:
                    results_text += f"**{det.state}** ({det.confidence:.2f})\n"
                details_placeholder.markdown(results_text)
            else:
                details_placeholder.markdown("No traffic lights detected")

    def _show_final_stats(self, total_detections: int, state_counts: dict):
        """Show final processing statistics"""
        st.subheader("Final Results")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Red Lights", state_counts['Red'])
        with col2:
            st.metric("Yellow Lights", state_counts['Yellow'])
        with col3:
            st.metric("Green Lights", state_counts['Green'])
        with col4:
            st.metric("Total Detections", total_detections)

    def render_instructions(self):
        """Instructions and info"""
        with st.expander("How to Use"):
            st.markdown("""
            ### Instructions:
            1. **Upload Video**: Choose a video file containing traffic lights
            2. **Adjust Settings**: Use sidebar to tune confidence and processing speed  
            3. **Process**: Click "Process Video" to start detection
            4. **View Results**: Watch real-time detection with statistics

            ### Features:
            - **YOLO Detection**: Accurate traffic light localization
            - **Zone-Based Analysis**: Color detection by light position
            - **Real-time Stats**: Live processing statistics
            - **Customizable**: Adjustable confidence and processing speed

            ### Tips:
            - Higher confidence = fewer false positives
            - Skip more frames = faster processing but less detail
            - Works best with clear, well-lit traffic lights
            """)

    def run(self):
        """Run the Streamlit app"""
        self.render_header()

        # Sidebar settings
        confidence, frame_skip, show_details = self.render_sidebar()

        # File upload
        uploaded_file = self.render_file_uploader()

        # Process video if uploaded
        if uploaded_file is not None:
            st.info(f"File uploaded: {uploaded_file.name}")

            if st.button("Process Video", type="primary"):
                with st.spinner("Processing video..."):
                    self.process_video(uploaded_file, confidence, frame_skip, show_details)

        # Instructions
        self.render_instructions()

        # Footer
        st.markdown("---")
        st.markdown("Built using **Streamlit**, **OpenCV**, and **YOLOv8**")


# Run the app
if __name__ == "__main__":
    app = StreamlitUI()
    app.run()