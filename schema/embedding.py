from pydantic import BaseModel
import numpy as np


class EmbeddingEntry(BaseModel):
    embedding: np.ndarray
    timestamp: float
    camera_id: int
    model_config = {"arbitrary_types_allowed": True}
