import cv2
import numpy as np
from ultralytics import YOLO

from config.tracking import TrackingConfig
from ai_services.reid import ReIDModel
from config.camera import CameraConfig
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
import subprocess
tracking_config= TrackingConfig()
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
        if config_camera.video_path:
            self.writer = cv2.VideoWriter(
                output_filename,
                cv2.VideoWriter_fourcc(*'mp4v'),
                self.fps,
                (self.width, self.height)
            )
        else:
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{self.width}x{self.height}",
                "-r", str(int(self.fps)),
                "-i", "-",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-f", "rtsp",  
                config_camera.output_url
            ]
            self.writer = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
            self.stream = True


        # Initialize tracker
        self.tracker = DeepSort(max_age=tracking_config.max_age, max_iou_distance=0.8, n_init=10, max_cosine_distance=0.2) #можливо треба ще доналаштувати ці параметри
        self.frame_count = self.config.detection_interval-1
        self.max_frames = float("inf") if config_camera.stream_url else int(self.fps * config_camera.max_duration_seconds)
        self.last_tracks = []
        self.last_detection = []
        self.track_to_global = {} #словник де track_id(який генерує deepsort) в парі з global_id (нашими айді)


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
        return (crop.size > 0 and
                crop.shape[0] >= TrackingConfig.min_crop_height and
                crop.shape[1] >= TrackingConfig.min_crop_width)

    def process_frame(self, detector: YOLO) -> np.ndarray | None:
        """Process a single frame and return it if successful."""
        ret, frame = self.cap.read()

        if not ret or self.frame_count >= self.max_frames:
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
            l,t,r,b = map(int, track.to_ltrb())
            if self.frame_count % self.config.detection_interval == 0:
                print(f"[FRAME {self.frame_count}] DEEPSORT tracks:")
                print(f"    LocalID={track.track_id} bbox=({l}, {t}, {r}, {b})")
        return frame

    def _detect_persons(self, detector: YOLO, frame: np.ndarray) -> list[tuple]:
        """Detect persons in the frame."""
        detections = []
        results = detector(frame, device= device)
        CONF_THRESHOLD = 0.5
        for i, box in enumerate(results[0].boxes.xyxy):
            cls = int(results[0].boxes.cls[i])
            conf = float(results[0].boxes.conf[i])
            # Only process person class (class 0)
            if cls != 0 or conf<CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            vertical = w / h > 1.6
            if w * h > self.min_box_area and not vertical: #валідація кропа за розміром та орієнтацією
                detections.append(([x1, y1, w, h], conf, 'person'))
            else:
                continue
        return detections

    def process_tracks(self, frame: np.ndarray, reid_model: ReIDModel) -> np.ndarray:
        """Process tracks and add annotations to frame."""
        if self.frame_count % self.config.detection_interval == 0:
            print(f"[FRAME {self.frame_count}] Processing {len(self.last_tracks)} tracks...")

        #active_ids_before = set(self.track_to_global.values())
        used_gids_this_frame: set[str] = set()

        for track in self.last_tracks:
            try:
                local_id = track.track_id
                l, t, r, b = map(int, track.to_ltrb())
                crop = frame[t:b, l:r]
                vertical = (r-l) / (b-t) > 1.6
                if (r-l) * (b-t)>self.min_box_area and not vertical:
                    print((r-l) * (b-t))
                    current_gid=self.track_to_global.get(local_id) #дізнаємось який зараз айді у цього треку
                    # Extract ReID embedding and assign global ID
                    if self.frame_count % self.config.detection_interval == 0:
                        embedding = reid_model.extract_embedding(crop)
                        assigned_gid = reid_model.assign_global_id(embedding, self.config.camera_id, current_gid, active_ids=used_gids_this_frame)
                        if assigned_gid:
                            if assigned_gid in used_gids_this_frame:
                                print(f"[ReID] {assigned_gid} already used in this frame, creating new one.")
                                assigned_gid = reid_model._create_new_identity(embedding, self.config.camera_id)

                            self.track_to_global[local_id] = assigned_gid
                            used_gids_this_frame.add(assigned_gid)
                        print(f"[DEBUG] used_gids_this_frame: {used_gids_this_frame}")
                    else:
                        assigned_gid = current_gid

                    # Draw bounding box and ID
                    if self.frame_count % self.config.detection_interval == 0:
                        print(f"    LocalID={local_id} -> GlobalID={assigned_gid}")

                    if assigned_gid:
                        self._draw_track_annotation(frame, l, t, r, b, assigned_gid)
                
            except Exception as e:
                print(e)
                # Log error in production code
                continue


        return frame

    def _draw_track_annotation(self, frame: np.ndarray, l: int, t: int,
                               r: int, b: int, global_id: str) -> None:
        """Draw bounding box and ID annotation on frame."""
        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {global_id}", (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    def write_frame(self, frame: np.ndarray) -> None:
        """Write frame to output video."""
        if self.stream:
             self.writer.stdin.write(frame.tobytes())
        else:
            self.writer.write(frame)

    def cleanup(self) -> None:
        """Release resources."""
        self.cap.release()
        if self.stream:
            self.writer.stdin.close()
            self.writer.wait()
        else:
            self.writer.release()