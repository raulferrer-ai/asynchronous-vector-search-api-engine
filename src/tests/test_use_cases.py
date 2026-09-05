import pytest
from src.domain.models import DocumentChunk
from src.application.use_cases.search import SearchSimilarDocumentsUseCase

@pytest.mark.asyncio
async def test_search_similar_documents_success(mocker):
    # Arrange
    mock_repo = mocker.AsyncMock()
    mock_embedder = mocker.AsyncMock()
    
    mock_embedder.generate_embedding.return_value = [0.1] * 1536
    sample_chunk = DocumentChunk(content="Test Clean Architecture")
    mock_repo.search_similar.return_value = [(sample_chunk, 0.96)]
    
    use_case = SearchSimilarDocumentsUseCase(mock_repo, mock_embedder)
    
    # Act
    results = await use_case.execute(query="Clean Code", limit=1)
    
    # Assert
    assert len(results) == 1
    assert results[0][0].content == "Test Clean Architecture"
    assert results[0][1] == 0.96
    mock_embedder.generate_embedding.assert_awaited_once_with("Clean Code")
    mock_repo.search_similar.assert_awaited_once()
