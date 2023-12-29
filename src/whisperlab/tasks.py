from uuid import uuid4, UUID
from pydantic import BaseModel, Field

from whisperlab.time import time_ms


class Task(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    # Task identification
    id: UUID = Field(default_factory=uuid4)
    batch_name: str = ""
    sequence_num: int = 0

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

    @property
    def duration(self) -> float | None:
        """Get the duration of the task in milliseconds."""
        if self.completed_time is None:
            return None
        return self.completed_time - self.created_time
