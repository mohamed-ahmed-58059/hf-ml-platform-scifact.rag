from pydantic import BaseModel


class Doc(BaseModel):
    id: str
    title: str
    text: str
    doc_type: str
