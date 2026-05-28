import sqlite3
import threading

import numpy as np


DB_PATH = "analytics.db"


class AnalyticsDB:
    """
    Thread-safe SQLite database for hotel monitoring analytics.

    Tables:
        persons — unique people seen by the system
        events  — entry/exit/elevator events per person
        sightings — every frame a person is confirmed on a camera
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_db()

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS persons (
                    global_id       TEXT PRIMARY KEY,
                    first_seen_at   TEXT NOT NULL,
                    last_seen_at    TEXT NOT NULL,
                    first_camera_id INTEGER NOT NULL,
                    role            TEXT NOT NULL DEFAULT 'client'
                );

                CREATE TABLE IF NOT EXISTS events (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    global_id   TEXT    NOT NULL,
                    camera_id   INTEGER NOT NULL,
                    event_type  TEXT    NOT NULL,
                    timestamp   TEXT    NOT NULL,
                    frame_number INTEGER,
                    FOREIGN KEY (global_id) REFERENCES persons(global_id)
                );

                CREATE TABLE IF NOT EXISTS sightings (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    global_id    TEXT    NOT NULL,
                    camera_id    INTEGER NOT NULL,
                    timestamp    TEXT    NOT NULL,
                    frame_number INTEGER,
                    bbox_l       INTEGER,
                    bbox_t       INTEGER,
                    bbox_r       INTEGER,
                    bbox_b       INTEGER,
                    FOREIGN KEY (global_id) REFERENCES persons(global_id)
                );

                CREATE TABLE IF NOT EXISTS embeddings (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    global_id   TEXT    NOT NULL,
                    camera_id   INTEGER NOT NULL,
                    timestamp   REAL    NOT NULL,
                    embedding   BLOB    NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_events_global_id    ON events(global_id);
                CREATE INDEX IF NOT EXISTS idx_events_camera_id    ON events(camera_id);
                CREATE INDEX IF NOT EXISTS idx_events_event_type   ON events(event_type);
                CREATE INDEX IF NOT EXISTS idx_sightings_global    ON sightings(global_id);
                CREATE INDEX IF NOT EXISTS idx_embeddings_global   ON embeddings(global_id);
                CREATE INDEX IF NOT EXISTS idx_embeddings_ts       ON embeddings(timestamp);
                CREATE INDEX IF NOT EXISTS idx_persons_role         ON persons(role);
            """)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upsert_person(
        self,
        global_id: str,
        camera_id: int,
        timestamp: str,
    ) -> None:
        """Create person record or update last_seen_at."""
        with self.lock:
            with self._conn() as conn:
                conn.execute(
                    """
                    INSERT INTO persons (global_id, first_seen_at, last_seen_at, first_camera_id)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(global_id) DO UPDATE SET last_seen_at = excluded.last_seen_at
                    """,
                    (global_id, timestamp, timestamp, camera_id),
                )

    def record_event(
        self,
        global_id: str,
        camera_id: int,
        event_type: str,
        timestamp: str,
        frame_number: int | None = None,
    ) -> None:
        """
        Record a named event for a person.

        event_type examples:
            person_entered_building
            person_exited_building
            person_entered_elevator
            person_exited_elevator
        """
        self.upsert_person(global_id, camera_id, timestamp)
        with self.lock:
            with self._conn() as conn:
                conn.execute(
                    """
                    INSERT INTO events (global_id, camera_id, event_type, timestamp, frame_number)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (global_id, camera_id, event_type, timestamp, frame_number),
                )

    def record_sighting(
        self,
        global_id: str,
        camera_id: int,
        timestamp: str,
        frame_number: int,
        bbox: tuple[int, int, int, int],
    ) -> None:
        """Record a single detection of a person on a camera."""
        self.upsert_person(global_id, camera_id, timestamp)
        l, t, r, b = bbox
        with self.lock:
            with self._conn() as conn:
                conn.execute(
                    """
                    INSERT INTO sightings
                        (global_id, camera_id, timestamp, frame_number, bbox_l, bbox_t, bbox_r, bbox_b)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (global_id, camera_id, timestamp, frame_number, l, t, r, b),
                )

    # ------------------------------------------------------------------
    # Embedding persistence
    # ------------------------------------------------------------------

    def save_embedding(
        self,
        global_id: str,
        embedding: np.ndarray,
        camera_id: int,
        timestamp: float,
    ) -> None:
        """Persist one embedding vector (float32 → raw bytes)."""
        blob = embedding.astype(np.float32).tobytes()
        with self.lock:
            with self._conn() as conn:
                conn.execute(
                    """
                    INSERT INTO embeddings (global_id, camera_id, timestamp, embedding)
                    VALUES (?, ?, ?, ?)
                    """,
                    (global_id, camera_id, timestamp, blob),
                )

    def load_embeddings(
        self, min_timestamp: float
    ) -> dict[str, list[tuple[np.ndarray, float, int]]]:
        """
        Load all embeddings newer than *min_timestamp*.
        Staff embeddings are always loaded regardless of TTL.

        Returns:
            {global_id: [(embedding, timestamp, camera_id), ...]}
            Each list is sorted by timestamp ASC.
        """
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT e.global_id, e.embedding, e.timestamp, e.camera_id
                FROM   embeddings e
                LEFT JOIN persons p ON e.global_id = p.global_id
                WHERE  e.timestamp >= ?
                   OR  p.role = 'staff'
                ORDER  BY e.global_id, e.timestamp ASC
                """,
                (min_timestamp,),
            ).fetchall()

        result: dict[str, list[tuple[np.ndarray, float, int]]] = {}
        for row in rows:
            gid = row["global_id"]
            emb = np.frombuffer(bytes(row["embedding"]), dtype=np.float32).copy()
            result.setdefault(gid, []).append(
                (emb, float(row["timestamp"]), int(row["camera_id"]))
            )
        return result

    def delete_identity_embeddings(self, global_id: str) -> None:
        """Remove all embedding rows for a stale / deleted identity."""
        with self.lock:
            with self._conn() as conn:
                conn.execute("DELETE FROM embeddings WHERE global_id = ?", (global_id,))

    def prune_old_embeddings(self, global_id: str, keep_count: int) -> None:
        """Keep only the *keep_count* newest rows for an identity."""
        with self.lock:
            with self._conn() as conn:
                conn.execute(
                    """
                    DELETE FROM embeddings
                    WHERE  global_id = ?
                      AND  id NOT IN (
                            SELECT id FROM embeddings
                            WHERE  global_id = ?
                            ORDER  BY timestamp DESC
                            LIMIT  ?
                           )
                    """,
                    (global_id, global_id, keep_count),
                )

    # ------------------------------------------------------------------
    # Staff management
    # ------------------------------------------------------------------

    def set_role(self, global_id: str, role: str) -> None:
        """Set the role for a person ('staff' or 'client')."""
        with self.lock:
            with self._conn() as conn:
                conn.execute(
                    "UPDATE persons SET role = ? WHERE global_id = ?",
                    (role, global_id),
                )

    def mark_as_staff(self, global_id: str) -> None:
        """Shortcut: mark a person as staff."""
        self.set_role(global_id, "staff")

    def get_staff_ids(self) -> set[str]:
        """Return the set of all staff global IDs."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT global_id FROM persons WHERE role = 'staff'"
            ).fetchall()
        return {row["global_id"] for row in rows}

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_all_persons(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM persons ORDER BY first_seen_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_events_for_person(self, global_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM events WHERE global_id = ? ORDER BY timestamp",
                (global_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_events_by_type(self, event_type: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM events WHERE event_type = ? ORDER BY timestamp DESC",
                (event_type,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_camera_activity(self, camera_id: int) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM events WHERE camera_id = ? ORDER BY timestamp DESC",
                (camera_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_entries_count(self) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM events WHERE event_type = 'person_entered_building'"
            ).fetchone()
        return row["cnt"]

    def get_exits_count(self) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM events WHERE event_type = 'person_exited_building'"
            ).fetchone()
        return row["cnt"]
