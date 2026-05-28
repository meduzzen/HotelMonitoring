import cv2
import numpy as np


class FrameProcessor:
    """Preprocesses frames and draws annotations."""

    MIN_BOX_AREA = 5
    MAX_VERTICAL_RATIO = 1.6

    @staticmethod
    def preprocess(frame: np.ndarray) -> np.ndarray:
        """Apply noise reduction and enhance brightness/contrast."""
        # frame = cv2.bilateralFilter(frame, 5, 75, 75)
        # frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)
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

    @staticmethod
    def draw_person_count(frame: np.ndarray, count: int):
        """Draw the total person count in the top right corner."""
        height, width = frame.shape[:2]
        text = f"People: {count}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2

        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        x = width - text_width - 20
        y = text_height + 20

        cv2.rectangle(
            frame,
            (x - 10, y - text_height - 10),
            (x + text_width + 10, y + 10),
            (0, 0, 0),
            -1,
        )

        cv2.putText(frame, text, (x, y), font, font_scale, (0, 255, 0), thickness)
