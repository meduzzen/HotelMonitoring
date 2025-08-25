import cv2
import numpy as np
from ultralytics import YOLO

from config.logging import setup_camera_logger
from config.tracking import TrackingConfig
from ai_services.reid import ReIDModel
from config.camera import CameraConfig
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
import os
from datetime import datetime

tracking_config = TrackingConfig()
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class CameraProcessor:
    """Handles processing for a single camera feed."""

    def __init__(self, config_camera: CameraConfig):
        self.config = config_camera
        if config_camera.video_path:
            self.cap = cv2.VideoCapture(config_camera.video_path)
        elif config_camera.stream_url is not None:
            self.cap = cv2.VideoCapture(config_camera.stream_url, cv2.CAP_FFMPEG)
        else:
            raise ValueError("Either video_path or stream_url must be provided")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.min_box_area = 30000
        output_filename = self._generate_output_filename()
        self.writer = cv2.VideoWriter(
            output_filename,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.width, self.height),
        )

        # Initialize tracker
        self.tracker = DeepSort(
            max_age=tracking_config.max_age,
            max_iou_distance=0.8,
            n_init=10,
            max_cosine_distance=0.2,
        )  # можливо треба ще доналаштувати ці параметри
        self.frame_count = self.config.detection_interval - 1
        self.max_frames = (
            float("inf")
            if config_camera.stream_url
            else int(self.fps * config_camera.max_duration_seconds)
        )
        self.last_tracks = []
        self.last_detection = []
        self.track_to_global = {}  # словник де track_id(який генерує deepsort) в парі з global_id (нашими айді)
        self.logger = setup_camera_logger(config_camera.camera_id)

        self.entry_line_y = 785
        self.prev_track_centers = {}  # track_id -> previous y
        self.entered_global_ids = set()  # prevent duplicate events
        self.exited_global_ids = set()

    def _generate_output_filename(self) -> str:
        """Generate output video filename."""
        return f"output_osnet_x1_0_{self.config.camera_id}.mp4"

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply preprocessing to the frame."""
        # Apply bilateral filter for noise reduction while preserving edges
        filtered_frame = cv2.bilateralFilter(frame, 5, 75, 75)
        # Enhance contrast and brightness
        enhanced_frame = cv2.convertScaleAbs(filtered_frame, alpha=1.1, beta=10)
        return enhanced_frame

    def _is_valid_crop(self, crop: np.ndarray) -> bool:
        """Check if the crop is valid for ReID processing."""
        return (
            crop.size > 0
            and crop.shape[0] >= TrackingConfig.min_crop_height
            and crop.shape[1] >= TrackingConfig.min_crop_width
        )

    def process_frame(self, detector: YOLO) -> np.ndarray | None:
        """Process a single frame and return it if successful."""
        ret, frame = self.cap.read()
        if not ret:
            return None

        self.frame_count += 1

        # Perform detection periodically
        if self.frame_count % self.config.detection_interval == 0:
            frame = self._preprocess_frame(frame)
            detections = self._detect_persons(detector, frame)
            self.last_detection = detections

        # Update tracker
        self.last_tracks = self.tracker.update_tracks(self.last_detection, frame=frame)

        for track in self.last_tracks:
            l, t, r, b = map(int, track.to_ltrb())  # noqa: E741
            if self.frame_count % self.config.detection_interval == 0:
                self.logger.info(
                    {
                        "event": "track_detected",
                        "frame": self.frame_count,
                        "local_id": track.track_id,
                        "bbox": [l, t, r, b],
                    }
                )
        return frame

    def _detect_persons(self, detector: YOLO, frame: np.ndarray) -> list[tuple]:
        """Detect persons in the frame."""
        detections = []
        results = detector(frame, device=device)
        CONF_THRESHOLD = 0.3
        for i, box in enumerate(results[0].boxes.xyxy):
            cls = int(results[0].boxes.cls[i])
            conf = float(results[0].boxes.conf[i])
            print(conf)
            # Only process person class (class 0)
            if cls != 0 or conf < CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            vertical = False if self.config.camera_id == 3 else w / h > 1.6
            if (
                w * h > self.min_box_area and not vertical
            ):  # валідація кропа за розміром та орієнтацією
                detections.append(([x1, y1, w, h], conf, "person"))
            else:
                continue
        return detections

    def process_tracks(self, frame: np.ndarray, reid_model: ReIDModel) -> np.ndarray:
        """Process tracks and add annotations to frame."""
        if self.frame_count % self.config.detection_interval == 0:
            print(
                f"[FRAME {self.frame_count}] Processing {len(self.last_tracks)} tracks..."
            )
        timestamp = None
        # active_ids_before = set(self.track_to_global.values())
        used_gids_this_frame: set[str] = set()

        for track in self.last_tracks:
            try:
                local_id = track.track_id
                l, t, r, b = map(int, track.to_ltrb())  # noqa: E741
                crop = frame[t:b, l:r]
                vertical = (
                    False if self.config.camera_id == 3 else (r - l) / (b - t) > 1.6
                )
                is_bottom_frame = (
                    False if self.config.camera_id == 3 else (self.height - b) < 100
                )  # якщо камера ліфта, правило що внизу або вгорі зображеня не робимо реідентифікацію  не працює
                is_top_frame = False if self.config.camera_id == 3 else t < 100
                if (r - l) * (b - t) > self.min_box_area and not vertical:
                    current_gid = self.track_to_global.get(
                        local_id
                    )  # дізнаємось який зараз айді у цього треку
                    # Extract ReID embedding and assign global ID
                    if (
                        self.frame_count % self.config.detection_interval == 0
                        and not is_bottom_frame
                        and not is_top_frame
                    ):
                        embedding = reid_model.extract_embedding(crop)
                        assigned_gid = reid_model.assign_global_id(
                            embedding,
                            self.config.camera_id,
                            current_gid,
                            used_gids_this_frame,
                            self.logger,
                        )
                        if assigned_gid:
                            if assigned_gid in used_gids_this_frame:
                                assigned_gid = reid_model._create_new_identity(
                                    embedding, self.config.camera_id
                                )

                            self.track_to_global[local_id] = assigned_gid
                            used_gids_this_frame.add(assigned_gid)
                    else:
                        assigned_gid = current_gid
                    timestamp = self.get_current_timestamp()
                    self.logger.info(
                        {
                            "event": "reid_assignment",
                            "frame": self.frame_count,
                            "local_id": local_id,
                            "global_id": assigned_gid,
                            "timestamp": timestamp,
                            "used_gids": list(used_gids_this_frame),
                        }
                    )
                    # Draw bounding box and ID
                    if assigned_gid:
                        self._draw_track_annotation(frame, l, t, r, b, assigned_gid)
                        self._check_entry_event(
                            track_id=local_id, global_id=assigned_gid, bbox=(l, t, r, b)
                        )

            except Exception as e:
                self.logger.error(
                    {"event": "error", "frame": self.frame_count, "error": str(e)}
                )
                continue

        return frame

    def _draw_track_annotation(
        self,
        frame: np.ndarray,
        l: int,  # noqa: E741
        t: int,
        r: int,
        b: int,
        global_id: str,
    ) -> None:
        """Draw bounding box and ID annotation on frame."""
        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)  # noqa: E741
        cv2.putText(
            frame,
            f"ID {global_id}, {l}, {t}, {r}, {b}",  # noqa: E741
            (l, t - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )
        if self.config.camera_id == 1:
            cv2.line(frame, (0, 785), (1920, 785), (255, 0, 0), 2)

    def write_frame(self, frame: np.ndarray) -> None:
        """Write frame to output video."""
        self.writer.write(frame)

    def get_current_timestamp(self) -> str:
        return datetime.now().strftime("%Y-%m-%d_%H:%M:%S:%f")[:-3]

    def save_body_crop(self, global_id: str, crop: np.ndarray, timestamp: str):
        os.makedirs(f"data_analysis/body_crop/{global_id}/", exist_ok=True)
        path = f"data_analysis/body_crop/{global_id}/{timestamp.replace('.', '_')}_id_{global_id}.jpg"
        cv2.imwrite(path, crop)
        return path

    def cleanup(self) -> None:
        """Release resources."""
        self.cap.release()
        self.writer.release()

    def _check_entry_event(
        self, track_id: int, global_id: str, bbox: tuple[int, int, int, int]
    ) -> None:
        """
        Detect entry / exit events by crossing y=785.
        Entry  : top -> bottom
        Exit   : bottom -> top
        Applies ONLY for camera_id == 1.
        """

        if (
            self.config.camera_id != 1 or not global_id
        ):  # детектимо вішла чи вишла людина з готелю на камері 1
            return

        l, t, r, b = bbox  # noqa: E741
        curr_y = b

        prev_y = self.prev_track_centers.get(track_id)

        self.prev_track_centers[track_id] = curr_y

        if prev_y is None:
            return

        if prev_y < self.entry_line_y <= curr_y:
            if global_id not in self.entered_global_ids:
                self.entered_global_ids.add(global_id)

                self.logger.info(
                    {
                        "event": "person_entered_building",
                        "camera_id": self.config.camera_id,
                        "global_id": global_id,
                        "frame": self.frame_count,
                        "entry_line_y": self.entry_line_y,
                        "direction": "top_to_bottom",
                        "timestamp": self.get_current_timestamp(),
                    }
                )

        elif prev_y > self.entry_line_y >= curr_y:
            if global_id not in self.exited_global_ids:
                self.exited_global_ids.add(global_id)

                self.logger.info(
                    {
                        "event": "person_exited_building",
                        "camera_id": self.config.camera_id,
                        "global_id": global_id,
                        "frame": self.frame_count,
                        "entry_line_y": self.entry_line_y,
                        "direction": "bottom_to_top",
                        "timestamp": self.get_current_timestamp(),
                    }
                )
