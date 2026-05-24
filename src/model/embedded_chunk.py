import numpy as np
from pydantic import BaseModel, ConfigDict


class EmbeddedChunk(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    doc_id: str
    text: str
    vector: np.ndarray
