from uuid import uuid4

from pydantic import BaseModel


class Task(BaseModel):
    id: str = uuid4()
    result: dict = {}
