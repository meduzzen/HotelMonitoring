from pydantic import BaseModel
import numpy as np


class EmbeddingEntry(BaseModel):
    embedding: np.ndarray
    timestamp: float
    camera_id: int
    model_config = {"arbitrary_types_allowed": True}


class TrackState(BaseModel):
    """Stores state information for a tracked person."""

    local_id: int
    global_id: str | None
    last_seen_frame: int
    elevator: bool
    # entry_timestamp: Optional[str] = None
    # exit_timestamp: Optional[str] = None
