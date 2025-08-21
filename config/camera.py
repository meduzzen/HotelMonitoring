from config.base import BaseConfig


class CameraConfig(BaseConfig):
    camera_id: int
    video_path: str
    max_duration_seconds: int = 4500
    detection_interval: int = 10  # Process detection every N frames