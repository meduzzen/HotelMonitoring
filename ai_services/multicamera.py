import threading

import torch
from ultralytics import YOLO

from ai_services.process_camera import CameraProcessor
from ai_services.reid import ReIDModel
from config.camera import CameraConfig


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class MultiCameraTracker:
    """Main class for multi-camera person tracking system with threading."""

    def __init__(self, reid_model_path: str, camera_configs: list[CameraConfig]):
        # Initialize shared models
        self.reid_model = ReIDModel(reid_model_path)
        self.detector = YOLO("models/yolov8s.pt").to(device)

        # Initialize cameras with required arguments
        self.cameras = {
            config.camera_id: CameraProcessor(config, self.detector, self.reid_model)
            for config in camera_configs
        }

        self.threads = []

    def run(self) -> None:
        """Run multi-camera tracking in parallel threads."""
        try:
            for cam_name, camera in self.cameras.items():
                t = threading.Thread(
                    target=self._process_camera_loop, args=(cam_name, camera)
                )
                t.start()
                self.threads.append(t)

            # Wait for all threads to finish
            for t in self.threads:
                t.join()

        finally:
            self._cleanup()

    def _process_camera_loop(self, cam_name, camera):
        """Process one camera in a loop inside a thread."""
        while True:
            frame = camera.process_next_frame()
            if frame is None:
                break
            camera.write_frame(frame)
        camera.cleanup()

    def _cleanup(self):
        for camera in self.cameras.values():
            camera.cleanup()
        # cv2.destroyAllWindows()
