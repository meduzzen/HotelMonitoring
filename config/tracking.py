from config.base import BaseConfig


class TrackingConfig(BaseConfig):
    reid_threshold: float = 0.35
    max_age: int = 30
    embedding_buffer_size: int = 10
    min_crop_height: int = 50
    min_crop_width: int = 25
