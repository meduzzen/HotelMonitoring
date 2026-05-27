from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

from config.tracking import TrackingConfig
from ai_services.reid import ReIDModel
from ai_services.frame_processor import FrameProcessor


class TrackerManager:
    """Manages DeepSort tracking and ReID assignments."""

    def __init__(self):
        tracking_config = TrackingConfig()
        self.tracker = DeepSort(
            max_age=tracking_config.max_age,
            max_iou_distance=0.8,
            n_init=2,
            max_cosine_distance=0.2,
        )
        self.track_to_global: dict[int, str] = {}
        self.last_person_count: int = 0

    def update(
        self,
        frame: np.ndarray,
        detections: list,
        reid_model: ReIDModel,
        frame_count: int,
        detection_interval: int,
        camera_id: str,
    ) -> list:
        """
        Update tracker and assign global IDs using ReID.

        Returns:
            List of updated tracks.
        """
        # We pass detections (even if empty) so DeepSORT can predict motion
        tracks = self.tracker.update_tracks(detections, frame=frame)
        used_gids: set[str] = set()

        # Get frame boundaries to prevent negative crop indices
        frame_h, frame_w = frame.shape[:2]

        current_person_count = 0

        for track in tracks:
            try:
                local_id = track.track_id
                l, t, r, b = map(int, track.to_ltrb())

                # Clamp coordinates to the frame boundaries
                l, t = max(0, l), max(0, t)
                r, b = min(frame_w, r), min(frame_h, b)

                # Ensure the crop is valid before proceeding
                if r <= l or b <= t:
                    continue

                crop = frame[t:b, l:r]
                vertical = (r - l) / (b - t) > FrameProcessor.MAX_VERTICAL_RATIO

                if (r - l) * (b - t) <= FrameProcessor.MIN_BOX_AREA or vertical:
                    continue

                # Count valid tracks
                current_person_count += 1

                current_gid = self.track_to_global.get(local_id)
                assigned_gid = current_gid

                if frame_count % detection_interval == 0:
                    embedding = reid_model.extract_embedding(crop)
                    assigned_gid = reid_model.assign_global_id(
                        embedding,
                        camera_id,
                        current_gid,
                        active_ids=used_gids,
                    )
                    if assigned_gid in used_gids:
                        assigned_gid = reid_model._create_new_identity(
                            embedding, camera_id
                        )

                    if assigned_gid:
                        self.track_to_global[local_id] = assigned_gid
                        used_gids.add(assigned_gid)

                if assigned_gid:
                    FrameProcessor.annotate(frame, l, t, r, b, assigned_gid)

            except Exception as e:
                print(f"Tracking error: {e}")
                continue

        # Only overwrite the saved count if this was a YOLO detection frame
        # This prevents the number from dropping to 0 on in-between frames
        if frame_count % detection_interval == 0:
            self.last_person_count = current_person_count

        FrameProcessor.draw_person_count(frame, self.last_person_count)

        return tracks
