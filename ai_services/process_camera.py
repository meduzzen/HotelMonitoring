import os
from typing import Optional, List, Dict

import cv2
import numpy as np

from loguru import logger
from config.tracking import TrackingConfig
from config.camera import CameraConfig
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
from FairMOT.src.lib.tracker.multitracker import JDETracker, STrack
from FairMOT.src.lib.opts import opts  # or wherever your option parser is
from ai_services.face_recognition import FaceRecognition

tracking_config = TrackingConfig()


class SharedMultiCameraTracker:
    def __init__(self, opts, face_recog: FaceRecognition):
        self.global_tracks: List[STrack] = []
        self.trackers: Dict[int, JDETracker] = {}  # Changed to dict with cam_id as key
        self.opts = opts
        self.global_track_id = 0
        self.face_recognition = face_recog

    def register_camera(self, cam_id: int) -> JDETracker:
        """Register a camera and return its tracker"""
        if cam_id in self.trackers:
            logger.warning(f"Camera {cam_id} already registered, returning existing tracker")
            return self.trackers[cam_id]

        tracker = JDETracker(self.opts)
        self.trackers[cam_id] = tracker
        logger.info(f"Registered camera {cam_id}, total cameras: {len(self.trackers)}")
        return tracker

    def update_and_match(self, img0, frame, cam_id: int):
        """Update tracker for specific camera and match with global tracks"""
        if cam_id not in self.trackers:
            raise ValueError(f"Camera {cam_id} not registered. Call register_camera({cam_id}) first.")

        tracker = self.trackers[cam_id]
        online_targets = tracker.update(frame, img0)

        logger.info(f"[Cam {cam_id}] Detected {len(online_targets)} tracks this frame")

        is_elevator = cam_id == 0

        # Match new tracks with global tracks
        for track in online_targets:
            if not track.is_activated:
                logger.debug(f"[Cam {cam_id}] Skipping inactive track {getattr(track, 'track_id', 'N/A')}")
                continue
            best_match = None
            min_dist = float('inf')
            matched = False

            logger.debug(f"[Cam {cam_id}] Track ID {track.track_id} trying to match...")
            for gtrack in self.global_tracks:
                # Skip comparison if track is from the same camera
                if gtrack.track_id==track.track_id:
                    logger.debug(f"[Same track {track.track_id} and gtrack {gtrack.track_id}]")
                    if best_match is None:
                        best_match = gtrack
                    matched=True
                    continue
                if hasattr(track, 'curr_feat') and hasattr(gtrack, 'curr_feat'):
                    dist = np.linalg.norm(track.curr_feat - gtrack.curr_feat)
                    logger.debug(
                        f"   Comparing with global ID {gtrack.track_id} from cam {cam_id} | Distance: {dist:.4f}")
                    if 0.2 <dist < 0.8:
                        min_dist = dist
                        best_match = gtrack
                        matched = True

            if best_match:
                logger.info(
                    f"[Cam {cam_id}] Track {track.track_id} matched with global ID {best_match.track_id} (dist={min_dist:.4f})")
                track.track_id = best_match.track_id
                best_match.curr_feat = track.curr_feat  # update embedding
                if is_elevator and best_match.track_id not in self.face_recognition.known_faces:
                    face_emb = self.face_recognition.extract_face_embedding(frame)
                    if face_emb:
                        self.face_recognition.save_face_embedding(best_match.track_id, face_emb, frame)
                        logger.info(f"[Elevator] Face saved for existing ReID match: {best_match.track_id}")
                    else:
                        logger.warning("[Elevator] No face detected to link to existing ReID match.")

            else:
                if is_elevator:
                    face_emb = self.face_recognition.extract_face_embedding(frame)
                    if face_emb is not None:
                        matching_face_id = self.face_recognition.find_matching_face_id(face_emb)
                        if matching_face_id:
                            logger.info(f"[FaceRec] Face match found → Assigning existing face ID: {matching_face_id}")
                            track.track_id = matching_face_id

                            # Search global_tracks and update body embedding
                            for gtrack in self.global_tracks:
                                if gtrack.track_id == matching_face_id:
                                    logger.info(f"[Update] Updating body embedding for track ID {matching_face_id}")
                                    gtrack.curr_feat = track.curr_feat
                                    break
                        else:
                            # New face: assign new global track ID and save face embedding
                            self.global_track_id += 1
                            track.track_id = self.global_track_id
                            self.global_tracks.append(track)
                            self.face_recognition.save_face_embedding(track.track_id, face_emb, frame)
                            logger.info(f"[FaceRec] No face match → Assigned new ID: {track.track_id}")
                    else:
                        # No face embedding found
                        logger.warning(f"[FaceRec] No face detected for track {track.track_id}")
                        self.global_track_id += 1
                        track.track_id = self.global_track_id
                        self.global_tracks.append(track)
                else:
                    # Non-elevator logic: assign new id
                    self.global_track_id += 1
                    track.track_id = self.global_track_id
                    self.global_tracks.append(track)
                    logger.info(f"[Cam {cam_id}] Track {track.track_id} added as new global ID {self.global_track_id}")

        return online_targets

    def get_camera_count(self) -> int:
        """Get number of registered cameras"""
        return len(self.trackers)


device = torch.device("cpu")


def get_fairmot_opt(load_model_path, input_w, input_h, fps):
    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    reid_dim = 128
    opt = Struct(
        task='mot',
        dataset='jde',
        arch='dla_34',
        load_model=load_model_path,
        reid_dim=reid_dim,
        heads={'hm': 1, 'wh': 4, 'id': reid_dim, 'reg': 2},
        head_conv=256,
        input_w=input_w,
        input_h=input_h,
        output_w=input_w // 4,
        output_h=input_h // 4,
        down_ratio=4,
        mean=[0.408, 0.447, 0.470],
        std=[0.289, 0.274, 0.278],
        nID=14455,
        conf_thres=0.4,
        det_thres=0.3,
        nms_thres=0.4,
        track_buffer=30,
        min_box_area=100,
        fix_res=True,
        keep_res=False,
        not_reg_offset=False,
        reg_offset=True,
        pad=31,
        num_stacks=1,
        root_dir=os.getcwd(),
        vis_thresh=0.5,
        gpus=[0],
        gpus_str='0',
        K=500,
        ltrb=True,
        num_classes=1,
    )
    opt.frame_rate = fps
    return opt


class CameraProcessor:
    """Handles processing for a single camera feed."""

    def __init__(self, config_camera: CameraConfig, tracker: SharedMultiCameraTracker):
        self.config = config_camera
        self.cap = cv2.VideoCapture(config_camera.video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.tracker = tracker
        self.camera_id = self.config.camera_id
        self.tracker.register_camera(self.camera_id)
        output_filename = self._generate_output_filename()
        self.output_width, self.output_height = 1088, 608
        self.writer = cv2.VideoWriter(
            output_filename,
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.fps,
            (self.output_width, self.output_height)
        )

        self.frame_count = self.config.detection_interval - 1
        self.max_frames = int(self.fps * config_camera.max_duration_seconds)
        self.last_tracks = []
        self.last_detection = []

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

    def preprocess_image_for_model(self, image: np.ndarray, target_size=(1088, 608)) -> np.ndarray:
        """
        Preprocess an image for FairMOT-style inference:
        - Resize with letterboxing to target size
        - Convert BGR to RGB
        - Normalize to [0, 1] and apply channel-first layout

        Args:
            image (np.ndarray): Input BGR image (H, W, 3)
            target_size (tuple): Desired (width, height)

        Returns:
            np.ndarray: Preprocessed image of shape (3, H, W)
        """
        # Resize with padding
        img = cv2.resize(image, (self.output_width, self.output_height))
        img, _, _, _ = self.letterbox(img, height=target_size[1], width=target_size[0])

        # BGR → RGB and CHW format
        img = img[:, :, ::-1].transpose(2, 0, 1)

        # Convert to float and normalize
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0

        return img

    @staticmethod
    def letterbox(img, height=608, width=1088,
                  color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
        shape = img.shape[:2]  # shape = [height, width]
        ratio = min(float(height) / shape[0], float(width) / shape[1])
        new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
        dw = (width - new_shape[0]) / 2  # width padding
        dh = (height - new_shape[1]) / 2  # height padding
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
        return img, ratio, dw, dh

    def process_frame(self) -> Optional[np.ndarray]:
        ret, frame = self.cap.read()
        if not ret or self.frame_count >= self.max_frames:
            return None

        self.frame_count += 1
        frame=cv2.resize(frame, (self.output_width, self.output_height))
        if self.frame_count % self.config.detection_interval == 0:
            img = self.preprocess_image_for_model(frame)
            blob = torch.from_numpy(img).unsqueeze(0)

            try:
                online_targets = self.tracker.update_and_match(frame, blob, cam_id=self.camera_id)
                logger.info(f"[CAM {self.camera_id}] Frame {self.frame_count}: {len(online_targets)} targets")

                tlwhs = []
                ids = []
                for t in online_targets:
                    if hasattr(t, 'tlwh') and hasattr(t, 'track_id'):
                        tlwhs.append(t.tlwh)
                        ids.append(t.track_id)

                self.last_detection = [tlwhs, ids]
                self.last_tracks = self.last_detection
            except Exception as e:
                logger.error(f"Error processing frame for camera {self.camera_id}: {e}")
                # Keep previous tracks on error
                pass

        return frame

    def process_tracks(self, frame: np.ndarray) -> np.ndarray:
        """Process tracks and add annotations to frame."""
        if len(self.last_tracks) >= 2:
            tlwhs, ids = self.last_tracks[0], self.last_tracks[1]
            for tlwh, track_id in zip(tlwhs, ids):
                frame = self.plot_tracking(frame, tlwh, track_id, frame_id=self.frame_count)
        return frame

    def plot_tracking(self, image, tlwh, obj_id, scores=None, frame_id=0, fps=0., ids2=None):
        im = np.ascontiguousarray(np.copy(image))
        im_h, im_w = im.shape[:2]

        text_scale = max(1, image.shape[1] / 1600.)
        text_thickness = 2
        line_thickness = max(1, int(image.shape[1] / 500.))

        cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, 1),
                    (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2)

        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        id_text = '{}'.format(int(obj_id))
        color = self.get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)
        return im

    @staticmethod
    def get_color(idx):
        idx = idx * 3
        color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
        return color

    def write_frame(self, frame: np.ndarray) -> None:
        if frame is None or frame.size == 0:
            logger.warning(f"Empty frame received for camera {self.camera_id}, skipping.")
            return

        try:
            self.writer.write(frame)
        except Exception as e:
            logger.error(f"Exception while writing frame for camera {self.camera_id}: {e}")
            logger.info(
                f"Frame info - shape: {frame.shape}, dtype: {frame.dtype}, min: {frame.min()}, max: {frame.max()}")



    def cleanup(self) -> None:
        """Release resources."""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        if hasattr(self, 'writer') and self.writer is not None:
            self.writer.release()
            logger.info(f"Video writer released for camera {self.camera_id}")
        cv2.destroyAllWindows()

