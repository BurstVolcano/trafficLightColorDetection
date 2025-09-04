import cv2
from typing import Optional, Generator, Callable
from traffic_detector import TrafficLightDetector

class VideoProcessor:
    def __init__(self, detector: TrafficLightDetector):
        self.detector = detector
        self.cap = None

    def open_source(self, source: Optional[str] = None) -> bool:
        self.cap = cv2.VideoCapture(0) if source is None else cv2.VideoCapture(source)
        return self.cap.isOpened()

    def get_video_info(self) -> dict:
        if not self.cap or not self.cap.isOpened():
            return {}
        return {
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'total_frames': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }

    def process_stream(self, callback: Optional[Callable] = None, frame_skip: int = 1) -> Generator:
        if not self.cap or not self.cap.isOpened():
            return
        frame_count = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if frame_count % frame_skip == 0:
                detections = self.detector.detect_in_frame(frame)
                if callback:
                    callback(frame, detections, frame_count)
                yield frame, detections, frame_count
            frame_count += 1

    def process_and_display(self, window_name: str = "Traffic Light Detection", frame_skip: int = 1):
        for frame, detections, _ in self.process_stream(frame_skip=frame_skip):
            annotated_frame = self.detector.draw_detections(frame, detections)
            cv2.imshow(window_name, annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                cv2.waitKey(0)
        self.cleanup()

    def save_processed_video(self, output_path: str, codec: str = 'mp4v', frame_skip: int = 1) -> bool:
        if not self.cap or not self.cap.isOpened():
            return False
        info = self.get_video_info()
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*codec), info['fps'], (info['width'], info['height']))
        try:
            for frame, detections, _ in self.process_stream(frame_skip=frame_skip):
                out.write(self.detector.draw_detections(frame, detections))
            return True
        except Exception as e:
            print(f"Error saving video: {e}")
            return False
        finally:
            out.release()
            self.cleanup()

    def cleanup(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
