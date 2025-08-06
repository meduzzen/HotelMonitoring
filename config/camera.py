from config.base import BaseConfig


class CameraConfig(BaseConfig):
    name: str
    video_path: str
    max_duration_seconds: int = 12
    detection_interval: int = 96  # Process detection every N frames