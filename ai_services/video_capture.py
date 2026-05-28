import cv2
import numpy as np


class VideoSource:
    """Handles video capture from file or stream."""

    def __init__(self, video_path: str | None, stream_url: str | None):
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
        elif stream_url:
            self.cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
        else:
            raise ValueError("Either video_path or stream_url must be provided.")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def grab(self) -> bool:
        """Grab next frame (without decoding)."""
        return self.cap.grab()

    def retrieve(self) -> np.ndarray | None:
        """Decode the grabbed frame."""
        ret, frame = self.cap.retrieve()
        if not ret:
            return None
        return frame

    def release(self):
        self.cap.release()
