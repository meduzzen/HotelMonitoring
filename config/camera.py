from config.base import BaseConfig


class CameraConfig(BaseConfig):
    name: str
    video_path: str
    max_duration_seconds: int = 10
    detection_interval: int = 10  # Process detection every N frames