from dotenv import load_dotenv
import os
import torch
from ultralytics import YOLO

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
roboflow_key = os.getenv("ROBOFLOW_API_KEY")


class PlayerDetector:
    def __init__(self, model_path: str, device: str = "cpu", output_dir: str = None):
        self.device = device
        self.output_dir = output_dir

        # Load YOLO model
        self.model = YOLO(model_path)
        self.model.to(self.device)

    def process_frame(self, frame, frame_id):
        results = self.model(frame, device=self.device)[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf
            })

        return detections
