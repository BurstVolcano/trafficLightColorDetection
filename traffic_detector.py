# traffic_detector.py
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Detection:
    """Data class for traffic light detection results"""
    state: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    zone_scores: Dict[str, int] = None


class ColorRanges:
    """HSV color ranges for traffic light detection"""
    RED_LOWER1 = np.array([0, 100, 100])
    RED_UPPER1 = np.array([10, 255, 255])
    RED_LOWER2 = np.array([160, 100, 100])
    RED_UPPER2 = np.array([180, 255, 255])

    YELLOW_LOWER = np.array([20, 100, 100])
    YELLOW_UPPER = np.array([35, 255, 255])

    GREEN_LOWER = np.array([40, 100, 100])
    GREEN_UPPER = np.array([85, 255, 255])


class TrafficLightDetector:
    """Main traffic light detection class"""

    def __init__(self, model_path: str = 'yolov8n.pt', confidence_threshold: float = 0.3):
        """
        Initialize the traffic light detector

        Args:
            model_path: Path to YOLO model
            confidence_threshold: Minimum confidence for detections
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.traffic_light_class_id = 9  # COCO class ID for traffic light

    def _count_color_pixels(self, region: np.ndarray) -> Dict[str, int]:
        """
        Count colored pixels in a region

        Args:
            region: BGR image region

        Returns:
            Dictionary with color pixel counts
        """
        if region.size == 0:
            return {"Red": 0, "Yellow": 0, "Green": 0}

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        # Create color masks
        red_mask1 = cv2.inRange(hsv, ColorRanges.RED_LOWER1, ColorRanges.RED_UPPER1)
        red_mask2 = cv2.inRange(hsv, ColorRanges.RED_LOWER2, ColorRanges.RED_UPPER2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        yellow_mask = cv2.inRange(hsv, ColorRanges.YELLOW_LOWER, ColorRanges.YELLOW_UPPER)
        green_mask = cv2.inRange(hsv, ColorRanges.GREEN_LOWER, ColorRanges.GREEN_UPPER)

        return {
            "Red": cv2.countNonZero(red_mask),
            "Yellow": cv2.countNonZero(yellow_mask),
            "Green": cv2.countNonZero(green_mask)
        }

    def _classify_traffic_light_state(self, crop: np.ndarray, pixel_threshold: int = 50) -> Tuple[str, Dict[str, int]]:
        """
        Classify traffic light state using zone-based analysis

        Args:
            crop: Cropped traffic light region
            pixel_threshold: Minimum colored pixels to classify as active

        Returns:
            Tuple of (state, zone_scores)
        """
        if crop.size == 0:
            return "Unknown", {}

        # Normalize size
        crop = cv2.resize(crop, (50, 150))
        h = crop.shape[0]

        # Divide into zones
        top_zone = crop[0:h // 3, :]  # Red light zone
        middle_zone = crop[h // 3:2 * h // 3, :]  # Yellow light zone
        bottom_zone = crop[2 * h // 3:, :]  # Green light zone

        # Analyze each zone
        top_counts = self._count_color_pixels(top_zone)
        mid_counts = self._count_color_pixels(middle_zone)
        bot_counts = self._count_color_pixels(bottom_zone)

        zone_scores = {
            'top_red': top_counts["Red"],
            'mid_yellow': mid_counts["Yellow"],
            'bot_green': bot_counts["Green"]
        }

        # Classify based on zone analysis
        if top_counts["Red"] > pixel_threshold:
            return "Red", zone_scores
        elif mid_counts["Yellow"] > pixel_threshold:
            return "Yellow", zone_scores
        elif bot_counts["Green"] > pixel_threshold:
            return "Green", zone_scores
        else:
            return "Unknown", zone_scores

    def detect_in_frame(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect traffic lights and their states in a frame

        Args:
            frame: Input frame (BGR format)

        Returns:
            List of Detection objects
        """
        detections = []
        results = self.model(frame)

        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    # Filter for traffic lights with sufficient confidence
                    if (int(box.cls) == self.traffic_light_class_id and
                            float(box.conf[0]) >= self.confidence_threshold):
                        # Extract bounding box
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])

                        # Extract and classify region
                        roi = frame[y1:y2, x1:x2]
                        state, zone_scores = self._classify_traffic_light_state(roi)

                        detection = Detection(
                            state=state,
                            confidence=confidence,
                            bbox=(x1, y1, x2, y2),
                            zone_scores=zone_scores
                        )
                        detections.append(detection)

        return detections

    @staticmethod
    def get_state_color(state: str) -> Tuple[int, int, int]:
        """Get BGR color for visualization based on state"""
        color_map = {
            "Red": (0, 0, 255),
            "Yellow": (0, 255, 255),
            "Green": (0, 255, 0),
            "Unknown": (128, 128, 128)
        }
        return color_map.get(state, (128, 128, 128))

    @staticmethod
    def draw_detections(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """
        Draw detection results on frame

        Args:
            frame: Input frame
            detections: List of detections to draw

        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            color = TrafficLightDetector.get_state_color(detection.state)

            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f'{detection.state} ({detection.confidence:.2f})'
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return annotated_frame