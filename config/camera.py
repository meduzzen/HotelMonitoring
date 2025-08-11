from config.base import BaseConfig


class CameraConfig(BaseConfig):
    name: str
    video_path: str
    max_duration_seconds: int = 4500
    detection_interval: int = 24  # Process detection every N frames