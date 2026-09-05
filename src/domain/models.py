from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import UUID, uuid4

@dataclass
class DocumentChunk:
    content: str
    metadata: dict[str, str] = field(default_factory=dict)
    embedding: list[float] | None = None
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class DomainException(Exception):
    """Base excpetion for the domain layer"""

class EmptyContentException(DomainException):
    def __init__(self):
        super().__init__("Fragment content cannot be empty")
