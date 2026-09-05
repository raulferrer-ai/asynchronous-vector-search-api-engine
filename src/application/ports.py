from abc import ABC, abstractmethod
from uuid import UUID
from src.domain.models import DocumentChunk

class DocumentRepositoryPort(ABC):
    @abstractmethod
    async def save(self, chunk: DocumentChunk) -> DocumentChunk:
        """Saves a vectorial chunk on the database."""
        pass

    @abstractmethod
    async def get_by_id(self, chunk_id: UUID) -> DocumentChunk | None:
        """Recovers a chunk by its ID UUID."""
        pass

    @abstractmethod
    async def search_similar(
        self, query_vector: list[float], limit: int = 5
    ) -> list[tuple[DocumentChunk, float]]:
        """Performs cearch by consinus similarity and returns tuples (chunk, similarity_score)."""
        pass

class EmbeddingGeneratorPort(ABC):
    @abstractmethod
    async def generate_embedding(self, text: str) -> list[float]:
        """Generates a vector embedding in an async way."""
        pass
