import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

from config.tracking import TrackingConfig


class TrackerManager:
    """
    Thin wrapper around DeepSort.

    Responsibility: tracking only — no ReID, no annotation.
    ReID and events are handled by CameraProcessor.
    """

    def __init__(self):
        config = TrackingConfig()
        self.tracker = DeepSort(
            max_age=config.max_age,
            max_iou_distance=0.5,
            n_init=5,
            max_cosine_distance=0.2,
        )

    def update(self, frame: np.ndarray, detections: list) -> list:
        """
        Feed detections into DeepSort and return updated tracks.

        Args:
            frame:      current frame (used by DeepSort for appearance features)
            detections: list of ([x, y, w, h], conf, "person") tuples

        Returns:
            List of DeepSort Track objects.
        """
        return self.tracker.update_tracks(detections, frame=frame)
