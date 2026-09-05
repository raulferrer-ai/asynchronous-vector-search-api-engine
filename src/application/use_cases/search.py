from src.application.ports import DocumentRepositoryPort, EmbeddingGeneratorPort
from src.domain.models import DocumentChunk

class SearchSimilarDocumentsUseCase:
    def __init__(
        self,
        repository: DocumentRepositoryPort,
        embedder: EmbeddingGeneratorPort,
    ):
        self._repository = repository
        self._embedder = embedder

    async def execute(self, query: str, limit: int = 5) -> list[tuple[DocumentChunk, float]]:
        if not query.strip():
            return []
        
        # 1. Generar embedding para la consulta
        query_vector = await self._embedder.generate_embedding(query)
        
        # 2. Buscar similitud vectorial en el repositorio
        return await self._repository.search_similar(query_vector, limit=limit)
