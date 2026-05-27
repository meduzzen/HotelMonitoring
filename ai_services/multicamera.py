import os
import platform
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
        self.start_event = threading.Event()

        print("Initializing ReID Model...")
        self.reid_model = ReIDModel(reid_model_path)

        base_model_name = "yolo26n"
        pytorch_model_path = f"{base_model_name}.pt"

        is_mac = platform.system() == "Darwin"
        is_apple_silicon = is_mac and platform.machine() == "arm64"

        if is_apple_silicon:
            export_format = "coreml"
            exported_model_path = f"{base_model_name}.mlpackage"
        else:
            export_format = "openvino"
            exported_model_path = f"{base_model_name}_openvino_model"

        if not os.path.exists(exported_model_path):
            print(
                f"Exporting {pytorch_model_path} to {export_format.upper()} format..."
            )
            temp_model = YOLO(pytorch_model_path)

            if export_format == "coreml":
                temp_model.export(format=export_format, nms=True, imgsz=1088)
            else:
                temp_model.export(format=export_format, imgsz=1088)

            print("Export complete!")

        print(f"Loading object detection model from {exported_model_path}...")
        self.detector = YOLO(exported_model_path, task="detect")

        print("Initializing camera processors...")
        self.cameras = {
            config.camera_id: CameraProcessor(config, self.detector, self.reid_model)
            for config in camera_configs
        }

        self.threads = []
        print("All models and processors successfully loaded.")

    def run(self) -> None:
        """Run multi-camera tracking in parallel threads."""
        try:
            for cam_name, camera in self.cameras.items():
                t = threading.Thread(
                    target=self._process_camera_loop, args=(cam_name, camera)
                )
                t.start()
                self.threads.append(t)

            print("Starting frame processing across all cameras...")
            self.start_event.set()

            # Wait for all threads to finish
            for t in self.threads:
                t.join()

        finally:
            self._cleanup()

    def _process_camera_loop(self, cam_name, camera):
        """Process one camera in a loop inside a thread."""
        self.start_event.wait()

        while True:
            frame = camera.process_next_frame()
            if frame is None:
                break
            camera.write_frame(frame)
        camera.cleanup()

    def _cleanup(self):
        for camera in self.cameras.values():
            camera.cleanup()
