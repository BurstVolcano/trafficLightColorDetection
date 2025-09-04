import cv2
import numpy as np
from typing import Optional, Generator, Callable
from traffic_detector import TrafficLightDetector, Detection
import time


class VideoProcessor:
    """Handles video input and processing"""

    def __init__(self, detector: TrafficLightDetector):
        """
        Initialize video processor

        Args:
            detector: TrafficLightDetector instance
        """
        self.detector = detector
        self.cap = None

    def open_source(self, source: Optional[str] = None) -> bool:
        """
        Open video source (file or camera)

        Args:
            source: Path to video file or None for webcam

        Returns:
            True if source opened successfully
        """
        if source is None:
            self.cap = cv2.VideoCapture(0)  # Default camera
        else:
            self.cap = cv2.VideoCapture(source)

        return self.cap.isOpened()

    def get_video_info(self) -> dict:
        """Get video information"""
        if not self.cap or not self.cap.isOpened():
            return {}

        return {
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'total_frames': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }

    def process_stream(self,
                       callback: Optional[Callable] = None,
                       frame_skip: int = 1) -> Generator:
        """
        Process video stream frame by frame

        Args:
            callback: Optional callback function for each frame
            frame_skip: Process every nth frame

        Yields:
            Tuple of (frame, detections, frame_number)
        """
        if not self.cap or not self.cap.isOpened():
            return

        frame_count = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Skip frames if needed
            if frame_count % frame_skip == 0:
                detections = self.detector.detect_in_frame(frame)

                if callback:
                    callback(frame, detections, frame_count)

                yield frame, detections, frame_count

            frame_count += 1

    def process_and_display(self,
                            window_name: str = "Traffic Light Detection",
                            frame_skip: int = 1):
        """
        Process video and display results in real-time

        Args:
            window_name: Name of display window
            frame_skip: Process every nth frame
        """
        print("Press 'q' to quit, 'space' to pause")

        for frame, detections, frame_num in self.process_stream(frame_skip=frame_skip):
            # Draw detections
            annotated_frame = self.detector.draw_detections(frame, detections)

            # Display frame
            cv2.imshow(window_name, annotated_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Exiting...")
                break
            elif key == ord(' '):
                print("Paused. Press any key to continue...")
                cv2.waitKey(0)

        self.cleanup()

    def save_processed_video(self,
                             output_path: str,
                             codec: str = 'mp4v',
                             frame_skip: int = 1) -> bool:
        """
        Process video and save results

        Args:
            output_path: Path to save processed video
            codec: Video codec to use
            frame_skip: Process every nth frame

        Returns:
            True if successful
        """
        if not self.cap or not self.cap.isOpened():
            return False

        info = self.get_video_info()
        fourcc = cv2.VideoWriter_fourcc(*codec)

        out = cv2.VideoWriter(output_path, fourcc, info['fps'],
                              (info['width'], info['height']))

        try:
            for frame, detections, _ in self.process_stream(frame_skip=frame_skip):
                annotated_frame = self.detector.draw_detections(frame, detections)
                out.write(annotated_frame)

            return True

        except Exception as e:
            print(f"Error saving video: {e}")
            return False

        finally:
            out.release()
            self.cleanup()

    def cleanup(self):
        """Release video capture resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()