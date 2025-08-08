from config.base import BaseConfig


class TrackingConfig(BaseConfig):
    reid_threshold: float = 0.2
    max_age: int = 15
    embedding_buffer_size: int = 10
    min_crop_height: int = 50
    min_crop_width: int = 25
