from config.base import BaseConfig
from typing import Optional

class CameraConfig(BaseConfig):
    camera_id: int
    video_path: Optional[str] = None
    stream_url: str = None
    output_url: str
    max_duration_seconds: int = 4500
    detection_interval: int = 10  # Process detection every N frames