from typing import List
import numpy as np
from ultralytics import YOLO
from src.detection.detector import BaseDetector, Detection

class YoloDetector(BaseDetector):
    """YOLO-based object detector using ultralytics."""

    def __init__(self, model_path: str = "yolov8m.pt"):
        """
        Initialize the detector.
        
        Args:
            model_path: Path to model weights (e.g., 'yolov8m.pt' or custom weights).
        """
        self.model = YOLO(model_path)
        
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Perform detection on a single frame.
        
        Args:
            frame: Input image as numpy array (H, W, C).
            
        Returns:
            List of Detection objects.
        """
        # Perform inference
        results = self.model(frame, verbose=False)[0]
        
        detections = []
        for box in results.boxes:
            # Get coordinates and metadata
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            name = results.names[cls]
            
            detections.append(Detection(
                bbox=[x1, y1, x2, y2],
                confidence=conf,
                class_id=cls,
                class_name=name
            ))
            
        return detections
