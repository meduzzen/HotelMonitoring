from config.base import BaseConfig


class CameraConfig(BaseConfig):
    camera_id: int
    video_path: str | None = None
    stream_url: str | None = None
    output_url: str | None = None
    max_duration_seconds: int = 4500
    detection_interval: int = 8  # Process detection every N frames
    min_box_area: int = 5  # Minimum bounding box area in pixels to accept detection
