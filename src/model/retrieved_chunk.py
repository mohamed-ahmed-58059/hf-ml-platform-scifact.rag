from pydantic import BaseModel


class RetrievedChunk(BaseModel):
    id: str
    doc_id: str
    text: str
    score: float
