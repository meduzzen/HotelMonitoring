import cv2

from ultralytics import YOLO

from ai_services.process_camera import CameraProcessor
from ai_services.reid import ReIDModel
from config.camera import CameraConfig
import torch
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
class MultiCameraTracker:
    """Main class for multi-camera person tracking system."""

    def __init__(self, reid_model_path: str, camera_configs: list[CameraConfig]):
        self.reid_model = ReIDModel(reid_model_path)
        self.detector = YOLO("models/yolov8s.pt").to(device)

        # Initialize camera processors
        self.cameras = {
            config.name: CameraProcessor(config)
            for config in camera_configs
        }

    def run(self) -> None:
        """Run the multi-camera tracking system."""
        try:
            while self._process_cameras():
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self._cleanup()

    def _process_cameras(self) -> bool:
        """Process all active cameras. Returns False if no cameras are active."""
        active_cameras = []

        for cam_name, camera in list(self.cameras.items()):
            frame = camera.process_frame(self.detector)

            if frame is None:
                # Camera finished processing
                camera.cleanup()
                del self.cameras[cam_name]
                continue

            # Process tracks and add annotations
            annotated_frame = camera.process_tracks(
                frame, self.reid_model
            )

            # Display and save frame
            #cv2.imshow(cam_name, annotated_frame)
            camera.write_frame(annotated_frame)
            active_cameras.append(cam_name)

        return len(active_cameras) > 0

    def _cleanup(self) -> None:
        """Clean up resources."""
        for camera in self.cameras.values():
            camera.cleanup()
        cv2.destroyAllWindows()