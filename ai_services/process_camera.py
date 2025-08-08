import cv2
import numpy as np
from ultralytics import YOLO

from config.tracking import TrackingConfig
from ai_services.reid import ReIDModel
from config.camera import CameraConfig
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
import json
import os
from datetime import datetime


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
            self.cap = cv2.VideoCapture(config_camera.stream_url)
        else:
            raise ValueError("Either video_path or stream_url must be provided")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.active_ids = {}
        self.entry_exit_log = {}


        # Initialize tracker
        self.tracker = DeepSort(max_age=tracking_config.max_age)
        self.frame_count = self.config.detection_interval-1
        self.max_frames = float("inf") if config_camera.stream_url else int(self.fps * config_camera.max_duration_seconds)
        self.last_tracks = []
        self.last_detection = []
        self.log_data = {}
        self.current_sessions = {}



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
            self.save_log_to_json()
            return None

        
        self.frame_count += 1


        # Perform detection periodically
        if self.frame_count % self.config.detection_interval == 0:
            frame = self._preprocess_frame(frame)
            detections = self._detect_persons(detector, frame)
            self.last_detection = detections
            self.detection_this_frame = True
        else:
            self.detection_this_frame = False

        # Update tracker
        self.last_tracks = self.tracker.update_tracks(self.last_detection, frame=frame)
        return frame



    def _detect_persons(self, detector: YOLO, frame: np.ndarray) -> list[tuple]:
        """Detect persons in the frame."""
        detections = []
        results = detector(frame, device= device)

        for i, box in enumerate(results[0].boxes.xyxy):
            cls = int(results[0].boxes.cls[i])
            conf = float(results[0].boxes.conf[i])
            # Only process person class (class 0)
            if cls != 0:
                continue

            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            detections.append(([x1, y1, w, h], conf, 'person'))
        return detections
    def _get_current_timestamp(self) -> str:
        return datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

    def _save_body_crop(self, global_id: str, crop: np.ndarray, timestamp: str):
        os.makedirs("data_analysis/people/body_crops", exist_ok=True)
        path = f"data_analysis/people/body_crops/{global_id}_at_{timestamp.replace('.', '_')}.jpg"
        cv2.imwrite(path, crop)
        print(f"[SAVE] Body crop saved to {path}")
        return path

    def process_tracks(self, frame: np.ndarray, reid_model: ReIDModel) -> np.ndarray:
        current_ids = set()
        """Process tracks and add annotations to frame."""
        for track in self.last_tracks:
            try:
                l, t, r, b = map(int, track.to_ltrb())
                crop = frame[t:b, l:r]
                if self.detection_this_frame:
                    # Extract ReID embedding and assign global ID
                    embedding = reid_model.extract_embedding(crop)
                    global_id, face_crop_path = reid_model.assign_global_id(embedding, camera_name=self.config.name, frame=frame, frame_number=self.frame_count)
                    track.global_id = global_id
                    current_ids.add(global_id)
                    # Draw bounding box and ID
                    #self._draw_track_annotation(frame, l, t, r, b, global_id)
                    timestamp = self._get_current_timestamp()

                    if global_id not in self.active_ids:
                        
                        self.active_ids[global_id] = timestamp
                        self.entry_exit_log[global_id] = {'entry': timestamp}
                        print(f"[ENTRY] Person {global_id} entered at {timestamp}")
                        body_crop_path = self._save_body_crop(global_id, crop, timestamp)
                        # Start session logging
                        self._on_entry(global_id, timestamp, face_detected=face_crop_path is not None, body_crop_path=body_crop_path, face_crop_path=face_crop_path)
                    else:
                        # Append crops for already active IDs
                        timestamp = self._get_current_timestamp()
                        body_crop_path = self._save_body_crop(global_id, crop, timestamp)
                        self._append_body_crop(global_id, body_crop_path)
                        self._append_face_crop(global_id, face_crop_path)
                else:
                        # On non-detection frames, retrieve the global_id from the track if it exists
                    if hasattr(track, 'global_id'):
                        current_ids.add(track.global_id)
                        #self._draw_track_annotation(frame, l, t, r, b, track.global_id)
            except Exception as e:
                # Log error in production code
                continue
        vanished_ids = set(self.active_ids.keys()) - current_ids
        for gid in vanished_ids:
            exit_time = self._get_current_timestamp()
            print(f"[EXIT] Person {gid} exited at {exit_time}")
            self.entry_exit_log[gid]['exit'] = exit_time
            del self.active_ids[gid]
            self._on_exit(gid, exit_time)

        return frame

    def _draw_track_annotation(self, frame: np.ndarray, l: int, t: int,
                               r: int, b: int, global_id: str) -> None:
        """Draw bounding box and ID annotation on frame."""
        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {global_id}", (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    '''def write_frame(self, frame: np.ndarray) -> None:
        """Write frame to output video."""
        self.writer.write(frame)'''

    def cleanup(self) -> None:
        """Release resources."""
        self.cap.release()
        '''self.writer.release()'''

    def _on_entry(self, global_id: str, timestamp: str, face_detected=False, body_crop_path=None, face_crop_path=None):
        session = {
            "entry": timestamp,
            "exit": None,
            "face_detected": face_detected,
            "body_crop_paths": [],
            "face_crop_paths": []
        }
        if body_crop_path:
            session["body_crop_paths"].append(body_crop_path)
        if face_crop_path:
            session["face_crop_paths"].append(face_crop_path)

        self.current_sessions[global_id] = session
        if global_id not in self.log_data:
            self.log_data[global_id] = []

    def _append_body_crop(self, global_id: str, path: str):
        if global_id in self.current_sessions:
            self.current_sessions[global_id]["body_crop_paths"].append(path)

    def _append_face_crop(self, global_id: str, path: str):
        if global_id in self.current_sessions:
            self.current_sessions[global_id]["face_crop_paths"].append(path)

    def _on_exit(self, global_id: str, timestamp: str):
        if global_id in self.current_sessions:
            self.current_sessions[global_id]["exit"] = timestamp

            if global_id not in self.log_data:
                self.log_data[global_id] = []
            self.log_data[global_id].append(self.current_sessions[global_id])

            #self.log_data[global_id].append(self.current_sessions[global_id])
            del self.current_sessions[global_id]
            
            self.save_log_to_json()
            self.log_data[global_id].clear()


    def save_log_to_json(self, file_path="data_analysis/entry_exit_log.json"):
        '''with open(file_path, "w") as f:
            json.dump(self.log_data, f, indent=4)
        print(f"[LOG] Entry/Exit log saved to {file_path}")'''
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = {}
        else:
            existing_data = {}
        for global_id, sessions in self.log_data.items():
            if global_id in existing_data:
                existing_data[global_id].extend(sessions)
            else:
                existing_data[global_id] = sessions

        with open(file_path, "w") as f:
            json.dump(existing_data, f, indent=4)

        print(f"[LOG] Entry/Exit log updated in {file_path}")