import os
import cv2
from ultralytics import YOLO
from ai_services.process_camera import CameraProcessor
from config.camera import CameraConfig


class CropSaver(CameraProcessor):
    """Extends CameraProcessor to save detection crops instead of tracking."""

    def __init__(self, config_camera: CameraConfig, output_folder: str):
        super().__init__(config_camera)
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        self.crop_index = 0
        self.frame_count = self.config.detection_interval - 1

    def process_and_save_crops(self, detector: YOLO):
        """Read frames, detect persons, save crops to folder."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.frame_count += 1
            if self.frame_count % self.config.detection_interval == 0:
                frame = self._preprocess_frame(frame)
                detections = self._detect_persons(detector, frame)

                for det in detections:
                    (x, y, w, h), conf, _ = det
                    crop = frame[y : y + h, x : x + w]
                    crop_filename = os.path.join(
                        self.output_folder, f"crop_{self.crop_index:05d}.jpg"
                    )
                    cv2.imwrite(crop_filename, crop)
                    self.crop_index += 1

        self.cleanup()
