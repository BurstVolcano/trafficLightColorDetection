import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class Detection:
    state: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    zone_scores: Dict[str, int] = None

class ColorRanges:
    RED_LOWER1 = np.array([0, 100, 100])
    RED_UPPER1 = np.array([10, 255, 255])
    RED_LOWER2 = np.array([160, 100, 100])
    RED_UPPER2 = np.array([180, 255, 255])
    YELLOW_LOWER = np.array([20, 100, 100])
    YELLOW_UPPER = np.array([35, 255, 255])
    GREEN_LOWER = np.array([40, 100, 100])
    GREEN_UPPER = np.array([85, 255, 255])

class TrafficLightDetector:
    def __init__(self, model_path: str = 'yolov8n.pt', confidence_threshold: float = 0.3):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.traffic_light_class_id = 9

    def _count_color_pixels(self, region: np.ndarray) -> Dict[str, int]:
        if region.size == 0:
            return {"Red": 0, "Yellow": 0, "Green": 0}
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, ColorRanges.RED_LOWER1, ColorRanges.RED_UPPER1),
            cv2.inRange(hsv, ColorRanges.RED_LOWER2, ColorRanges.RED_UPPER2)
        )
        return {
            "Red": cv2.countNonZero(red_mask),
            "Yellow": cv2.countNonZero(cv2.inRange(hsv, ColorRanges.YELLOW_LOWER, ColorRanges.YELLOW_UPPER)),
            "Green": cv2.countNonZero(cv2.inRange(hsv, ColorRanges.GREEN_LOWER, ColorRanges.GREEN_UPPER))
        }

    def _classify_traffic_light_state(self, crop: np.ndarray, pixel_threshold: int = 50) -> Tuple[str, Dict[str, int]]:
        if crop.size == 0:
            return "Unknown", {}
        crop = cv2.resize(crop, (50, 150))
        h = crop.shape[0]
        top_counts = self._count_color_pixels(crop[0:h // 3, :])
        mid_counts = self._count_color_pixels(crop[h // 3:2 * h // 3, :])
        bot_counts = self._count_color_pixels(crop[2 * h // 3:, :])
        zone_scores = {'top_red': top_counts["Red"], 'mid_yellow': mid_counts["Yellow"], 'bot_green': bot_counts["Green"]}
        if top_counts["Red"] > pixel_threshold:
            return "Red", zone_scores
        elif mid_counts["Yellow"] > pixel_threshold:
            return "Yellow", zone_scores
        elif bot_counts["Green"] > pixel_threshold:
            return "Green", zone_scores
        return "Unknown", zone_scores

    def detect_in_frame(self, frame: np.ndarray) -> List[Detection]:
        detections = []
        results = self.model(frame)
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    if int(box.cls) == self.traffic_light_class_id and float(box.conf[0]) >= self.confidence_threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        roi = frame[y1:y2, x1:x2]
                        state, zone_scores = self._classify_traffic_light_state(roi)
                        detections.append(Detection(state, float(box.conf[0]), (x1, y1, x2, y2), zone_scores))
        return detections

    @staticmethod
    def get_state_color(state: str) -> Tuple[int, int, int]:
        return {
            "Red": (0, 0, 255),
            "Yellow": (0, 255, 255),
            "Green": (0, 255, 0),
            "Unknown": (128, 128, 128)
        }.get(state, (128, 128, 128))

    @staticmethod
    def draw_detections(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        annotated_frame = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            color = TrafficLightDetector.get_state_color(detection.state)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            label = f'{detection.state} ({detection.confidence:.2f})'
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return annotated_frame
