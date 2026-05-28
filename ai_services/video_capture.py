import threading
import time
import cv2
import numpy as np


class VideoSource:
    """Continuously reads frames from a camera/stream in a dedicated background thread."""

    def __init__(self, video_path: str | None, stream_url: str | None):
        source = stream_url if stream_url else video_path
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video source: {source}")

        # Extract properties so your CameraProcessor can read them
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

        self.ret = False
        self.frame = None
        self.running = True

        self.lock = threading.Lock()
        self.ret, self.frame = self.cap.read()

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        """Constantly pull the latest frame from the stream."""
        while self.running:
            ret, frame = self.cap.read()

            with self.lock:
                self.ret = ret
                self.frame = frame

            if not ret:
                time.sleep(0.01)

    def grab(self) -> bool:
        """Check if the stream is still providing valid frames."""
        with self.lock:
            return self.ret

    def retrieve(self) -> np.ndarray | None:
        """Get a copy of the most recent frame."""
        with self.lock:
            if self.ret and self.frame is not None:
                return self.frame.copy()
            return None

    def release(self):
        """Stop the background reading thread and release the camera."""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()
