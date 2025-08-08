from config.base import BaseConfig
from typing import Optional

class CameraConfig(BaseConfig):
    name: str
    video_path: Optional[str] = None
    stream_url: str = None
    max_duration_seconds: int = 60
    detection_interval: int = 400  # Process detection every N frames