import numpy as np
import torch
from ultralytics import YOLO

from config.camera import CameraConfig
from ai_services.reid import ReIDModel
from ai_services.video_capture import VideoSource
from ai_services.video_writer import VideoOutput
from ai_services.frame_processor import FrameProcessor
from ai_services.person_detector import PersonDetector
from ai_services.tracker_manager import TrackerManager

device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)


class CameraProcessor:
    """High-level orchestrator for processing a camera feed."""

    def __init__(
        self, config_camera: CameraConfig, detector: YOLO, reid_model: ReIDModel
    ):
        self.config = config_camera
        self.detector = PersonDetector(detector)
        self.reid_model = reid_model

        self.source = VideoSource(config_camera.video_path, config_camera.stream_url)
        output_path = (
            self._generate_output_filename() if config_camera.video_path else None
        )
        self.output = VideoOutput(
            width=self.source.width,
            height=self.source.height,
            fps=self.source.fps,
            output_path=output_path,
            stream_url=config_camera.output_url,
        )

        self.processor = FrameProcessor()
        self.tracker = TrackerManager()
        self.frame_count = config_camera.detection_interval - 1
        self.max_frames = (
            float("inf")
            if config_camera.stream_url
            else int(self.source.fps * config_camera.max_duration_seconds)
        )
        self.last_frame: np.ndarray | None = None

    def _generate_output_filename(self) -> str:
        return f"output_osnet_x1_0_{self.config.camera_id}.mp4"

    def process_next_frame(self) -> np.ndarray | None:
        """Grab, process, track, annotate, and return the next frame."""
        if not self.source.grab() or self.frame_count >= self.max_frames:
            return None

        self.frame_count += 1
        if self.frame_count % self.config.detection_interval == 0:
            frame = self.source.retrieve()
            if frame is None:
                return None
            frame = self.processor.preprocess(frame)
            self.last_frame = frame.copy()

            detections = self.detector.detect(frame)
            self.tracker.update(
                frame,
                detections,
                self.reid_model,
                self.frame_count,
                self.config.detection_interval,
                self.config.camera_id,
            )
        else:
            frame = self.last_frame
            if frame is not None:
                self.tracker.update(
                    frame,
                    [],
                    self.reid_model,
                    self.frame_count,
                    self.config.detection_interval,
                    self.config.camera_id,
                )

        return frame

    def write_frame(self, frame: np.ndarray):
        self.output.write(frame)

    def cleanup(self):
        self.source.release()
        self.output.release()
