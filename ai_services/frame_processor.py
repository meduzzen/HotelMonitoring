import cv2
import numpy as np


class FrameProcessor:
    """Preprocesses frames and draws annotations."""

    MIN_BOX_AREA = 30000
    MAX_VERTICAL_RATIO = 1.6

    @staticmethod
    def preprocess(frame: np.ndarray) -> np.ndarray:
        """Apply noise reduction and enhance brightness/contrast."""
        frame = cv2.bilateralFilter(frame, 5, 75, 75)
        frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)
        return frame

    @staticmethod
    def annotate(frame: np.ndarray, l: int, t: int, r: int, b: int, global_id: str):
        """Draw bounding box and global ID."""
        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID {global_id}",
            (l, t - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )
