from typing import Optional
from pydantic import BaseModel


class Section(BaseModel):
    type: str
    count: int
    marks_each: int


class PaperRequest(BaseModel):
    class_level: str
    subject: str
    chapters: list[str]
    total_marks: int
    duration_minutes: Optional[int] = None
    difficulty: str
    sections: list[Section]