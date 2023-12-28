from typing import Optional, UUID
from uuid import uuid4
from pydantic import BaseModel, Field

from whisperlab.time import time_ms


class Task(BaseModel):
    # Task identification
    id: UUID = Field(default_factory=uuid4)
    batch: str = ""
    sequence: int = 0

    # Task status
    completed: bool = False
    result: dict = {}

    # Task timing
    created_time: float = Field(default_factory=time_ms)
    completed_time: float = None

    def complete(self, result: dict):
        """Complete the task with the given result."""
        self.completed = True
        self.result = result
        self.completed_time = time_ms()
