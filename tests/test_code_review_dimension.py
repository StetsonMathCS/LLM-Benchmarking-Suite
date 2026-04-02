"""
End-to-end tests for the CodeReviewDimension.

Note: These tests mock the Ollama embedding service since it requires
a running Ollama server. Tests verify the logic, not the Ollama connection.
"""
import pytest
from unittest.mock import patch, MagicMock
from benchmarks.dimensions.code_review import CodeReviewDimension
from core.base import DimensionResult


# ── Cosine similarity ─────────────────────────────────────────────────────

class TestCosineSimilarity:
    def setup_method(self):
        # Create instance with mocked Ollama (won't try to connect)
        with patch.object(CodeReviewDimension, '_connect_ollama', return_value=False):
            self.dim = CodeReviewDimension()

    def test_identical_vectors(self):
        vec = [1.0, 0.0, 0.0]
        assert self.dim._cosine_similarity(vec, vec) == pytest.approx(1.0, abs=0.01)

    def test_orthogonal_vectors(self):
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        assert self.dim._cosine_similarity(v1, v2) == pytest.approx(0.0, abs=0.01)

    def test_empty_vectors(self):
        assert self.dim._cosine_similarity([], []) == 0.0

    def test_none_vectors(self):
        assert self.dim._cosine_similarity(None, [1.0]) == 0.0
        assert self.dim._cosine_similarity([1.0], None) == 0.0

    def test_zero_vector(self):
        assert self.dim._cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


# ── Evaluation without Ollama ─────────────────────────────────────────────

class TestEvaluationWithoutOllama:
    def setup_method(self):
        with patch.object(CodeReviewDimension, '_connect_ollama', return_value=False):
            self.dim = CodeReviewDimension()

    def test_missing_expected_output(self):
        result = self.dim.evaluate(generated_review="good code")
        assert result.score == 0.0
        assert not result.passed

    def test_missing_generated_review(self):
        result = self.dim.evaluate(
            generated_review="",
            expected_output="should find bugs",
        )
        assert result.score == 0.0

    def test_ollama_unavailable_returns_zero(self):
        result = self.dim.evaluate(
            generated_review="looks good",
            expected_output="the code has issues",
        )
        assert result.score == 0.0
        assert "Failed to generate embeddings" in result.details.get("error", "")


# ── Evaluation with mocked Ollama ─────────────────────────────────────────

class TestEvaluationWithMockedOllama:
    def setup_method(self):
        with patch.object(CodeReviewDimension, '_connect_ollama', return_value=True):
            self.dim = CodeReviewDimension()
            self.dim._ollama_available = True

    def test_similar_reviews_score_high(self):
        embedding = [0.5, 0.3, 0.8, 0.1]
        with patch.object(self.dim, '_get_embedding', return_value=embedding):
            result = self.dim.evaluate(
                generated_review="The code has a bug in the loop",
                expected_output="There is a bug in the loop logic",
            )
        assert result.score == pytest.approx(1.0, abs=0.01)
        assert result.passed is True

    def test_different_reviews_score_lower(self):
        def mock_embedding(text):
            if "bug" in text:
                return [1.0, 0.0, 0.0, 0.0]
            return [0.0, 0.0, 0.0, 1.0]

        with patch.object(self.dim, '_get_embedding', side_effect=mock_embedding):
            result = self.dim.evaluate(
                generated_review="The code looks fine",
                expected_output="There is a critical bug in the loop",
            )
        assert result.score < 0.5

    def test_details_contain_metadata(self):
        embedding = [0.5, 0.3, 0.8]
        with patch.object(self.dim, '_get_embedding', return_value=embedding):
            result = self.dim.evaluate(
                generated_review="review text",
                expected_output="expected text",
            )
        assert "similarity_method" in result.details
        assert "embedding_model" in result.details
        assert result.details["similarity_method"] == "cosine"


# ── Embedding cache ───────────────────────────────────────────────────────

class TestEmbeddingCache:
    def setup_method(self):
        with patch.object(CodeReviewDimension, '_connect_ollama', return_value=True):
            self.dim = CodeReviewDimension()
            self.dim._ollama_available = True

    def test_cache_populated(self):
        self.dim._embedding_cache["cached text"] = [1.0, 2.0]
        # _get_embedding checks _ollama_available first; set it True to hit cache
        self.dim._ollama_available = True
        self.dim._lib = MagicMock()  # needs a non-None _lib
        result = self.dim._get_embedding("cached text")
        assert result == [1.0, 2.0]

    def test_empty_text_returns_none(self):
        assert self.dim._get_embedding("") is None
        assert self.dim._get_embedding(None) is None
