from datetime import datetime

from db.analytics import AnalyticsDB


class EventDetector:
    """
    Detects zone-crossing events per camera:
      - Camera 1: person entered / exited building (horizontal line crossing)
      - Camera 3: person entered / exited elevator (vertical line crossing)
    """

    def __init__(
        self, camera_id: int, db: AnalyticsDB, logger, width: int, height: int
    ):
        self.camera_id = camera_id
        self.db = db
        self.logger = logger
        self.width = width
        self.height = height

        # ---- camera 1: building entry/exit ----
        self.entry_line_y = 785
        self.prev_track_centers: dict[int, int] = {}  # track_id -> prev bottom-y
        self.person_inside: dict[
            str, bool | None
        ] = {}  # global_id -> state (allows re-entry)

        # ---- camera 3: elevator entry/exit ----
        self.elevator_left_line = 630
        self.elevator_right_line = 1230
        self.prev_track_lr: dict[
            int, tuple[int, int]
        ] = {}  # track_id -> (prev_l, prev_r)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def check_entry_event(
        self,
        track_id: int,
        global_id: str,
        bbox: tuple[int, int, int, int],
        frame_number: int,
    ) -> None:
        """
        Fire entered/exited building events when the bottom edge of the bbox
        crosses entry_line_y. Works only for camera 1.
        Uses per-global_id state so the same person can enter/exit multiple times.
        """
        if self.camera_id != 1 or not global_id:
            return

        _, _, _, b = bbox
        curr_y = b
        prev_y = self.prev_track_centers.get(track_id)
        self.prev_track_centers[track_id] = curr_y

        if prev_y is None:
            return

        timestamp = self._now()
        is_inside = self.person_inside.get(global_id)

        # Top -> Bottom: person entered building
        if prev_y < self.entry_line_y <= curr_y:
            if is_inside is not True:
                self.person_inside[global_id] = True
                self._log_and_save(
                    event_type="person_entered_building",
                    global_id=global_id,
                    frame_number=frame_number,
                    timestamp=timestamp,
                    extra={
                        "direction": "top_to_bottom",
                        "entry_line_y": self.entry_line_y,
                    },
                )

        # Bottom -> Top: person exited building
        elif prev_y > self.entry_line_y >= curr_y:
            if is_inside is not False:
                self.person_inside[global_id] = False
                self._log_and_save(
                    event_type="person_exited_building",
                    global_id=global_id,
                    frame_number=frame_number,
                    timestamp=timestamp,
                    extra={
                        "direction": "bottom_to_top",
                        "entry_line_y": self.entry_line_y,
                    },
                )

    def check_elevator_event(
        self,
        track_id: int,
        global_id: str,
        bbox: tuple[int, int, int, int],
        frame_number: int,
    ) -> bool | None:
        """
        Detect elevator entry/exit when bbox fully crosses the left (630)
        or right (1230) boundary lines. Works only for camera 3.

        Returns:
            True  — person entered elevator
            False — person exited elevator
            None  — no event this frame
        """
        if self.camera_id != 3 or not global_id:
            return None

        l, _, r, _ = bbox
        prev = self.prev_track_lr.get(track_id)
        self.prev_track_lr[track_id] = (l, r)

        if prev is None:
            return None

        prev_l, prev_r = prev
        entered = None
        direction = None

        if (
            prev_l < self.elevator_left_line
            and l >= self.elevator_left_line
            and r >= self.elevator_left_line
        ):
            entered, direction = True, "left_to_inside"
        elif (
            prev_r >= self.elevator_left_line
            and l < self.elevator_left_line
            and r < self.elevator_left_line
        ):
            entered, direction = False, "inside_to_left"
        elif (
            prev_r > self.elevator_right_line
            and l <= self.elevator_right_line
            and r <= self.elevator_right_line
        ):
            entered, direction = True, "right_to_inside"
        elif (
            prev_l <= self.elevator_right_line
            and l > self.elevator_right_line
            and r > self.elevator_right_line
        ):
            entered, direction = False, "inside_to_right"

        if entered is None:
            return None

        event_type = "person_entered_elevator" if entered else "person_exited_elevator"
        timestamp = self._now()

        self._log_and_save(
            event_type=event_type,
            global_id=global_id,
            frame_number=frame_number,
            timestamp=timestamp,
            extra={"direction": direction},
        )
        return entered

    def cleanup_track(self, track_id: int) -> None:
        """Remove state for a track that DeepSort has deleted."""
        self.prev_track_centers.pop(track_id, None)
        self.prev_track_lr.pop(track_id, None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_and_save(
        self,
        event_type: str,
        global_id: str,
        frame_number: int,
        timestamp: str,
        extra: dict,
    ) -> None:
        self.logger.info(
            {
                "event": event_type,
                "camera_id": self.camera_id,
                "global_id": global_id,
                "frame": frame_number,
                "timestamp": timestamp,
                **extra,
            }
        )
        self.db.record_event(
            global_id=global_id,
            camera_id=self.camera_id,
            event_type=event_type,
            timestamp=timestamp,
            frame_number=frame_number,
        )

    @staticmethod
    def _now() -> str:
        return datetime.now().strftime("%Y-%m-%d_%H:%M:%S:%f")[:-3]
