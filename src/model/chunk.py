from pydantic import BaseModel


class Chunk(BaseModel):
    id: str
    doc_id: str
    text: str
