"""
benchmarks/dimensions/code_review.py

Evaluates code review quality using Ollama embeddings for semantic similarity.
Compares generated review against expected review using cosine similarity.
"""
from core.base import (
    BaseDimension,
    DimensionResult
)
from typing import Optional, List
from time import perf_counter


class ReferenceReviewSimilarityDimension(BaseDimension):
    dimension_id = "reference_review_similarity"
    name = "Reference Review Similarity (RRS)"
    description = "Embedding similarity to a reference review; it does not establish expert review accuracy."
# Default Ollama embedding model and host
    EMBEDDING_MODEL = "nomic-embed-text:latest"
    DEFAULT_BASE_URL = "http://localhost:11434"
    USAGE_SINK = None

    def __init__(self):
        """Initialize the dimension and create Ollama client."""
        super().__init__()
        self._embedding_cache = {}
        self._base_url = self.DEFAULT_BASE_URL
        self._lib = None
        self._ollama_available = self._connect_ollama()

    def _report_usage(self, raw_usage: dict | None, latency_ms: float | None, status: str) -> None:
        """Report embedding token usage when the run has accounting enabled.

        Embedding calls are billable model work, so they are recorded separately
        from generation. Accounting never issues its own requests.
        """
        sink = self.USAGE_SINK
        if sink is None:
            return
        sink({
            "purpose": "embedding",
            "provider": "ollama",
            "model": self.EMBEDDING_MODEL,
            "raw_usage": raw_usage,
            "usage": None,
            "status": status,
            "latency_ms": latency_ms,
        })
    
    def _connect_ollama(self) -> bool:
        """Create Ollama client and check availability."""
        try:
            import ollama
            self._lib = ollama.Client(host=self._base_url)
            
            # Test connection by listing models
            models = self._lib.list()
            model_names = [m.model.split(':')[0] for m in models.models]
            
            if self.EMBEDDING_MODEL.split(':')[0] not in model_names:
                print(f"Warning: {self.EMBEDDING_MODEL} not found in Ollama. Available: {model_names}")
                return False
            return True
        except Exception as e:
            print(f"Warning: Ollama not available at {self._base_url}: {e}")
            return False
    
    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text using Ollama."""
        if not text or not self._ollama_available or not self._lib:
            return None
        
        # Check cache first
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        
        try:
            started = perf_counter()
            response = self._lib.embed(
                model=self.EMBEDDING_MODEL,
                input=text
            )
            latency_ms = (perf_counter() - started) * 1000
            embedding = response['embeddings'][0] if response.get('embeddings') else None
            self._report_usage(
                {"prompt_eval_count": response.get("prompt_eval_count")},
                latency_ms,
                "success" if embedding else "failed",
            )

            if embedding:
                self._embedding_cache[text] = embedding
            return embedding

        except Exception as e:
            self._report_usage(None, None, "failed")
            print(f"Error getting embedding: {e}")
            return None
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not vec1 or not vec2:
            return 0.0
        
        import numpy as np
        arr1 = np.array(vec1, dtype=np.float32)
        arr2 = np.array(vec2, dtype=np.float32)
        
        dot_product = np.dot(arr1, arr2)
        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def evaluate(
        self, 
        generated_review: str,
        **kwargs
    ) -> DimensionResult:
        """
        Evaluate code review by comparing semantic similarity using embeddings.
        
        Args:
            language: Programming language (unused here)
            code_input: Original code being reviewed
            expected_output: Expected/reference review from dataset
            generated_review: LLM-generated review to evaluate
            **kwargs: Additional arguments (ignored)
        
        Returns:
            DimensionResult with semantic similarity score
        """
        expected_output = kwargs.get('expected_output')
        # Validate inputs
        if not expected_output or not generated_review:
            status = "infrastructure_error" if not expected_output else "candidate_failure"
            return DimensionResult(
                dimension_name=self.name,
                score=0.0,
                passed=False,
                details={
                    "error": "Missing expected_output or generated_review",
                    "expected_empty": not expected_output,
                    "generated_empty": not generated_review,
                },
                status=status,
            )
        
        # Get embeddings
        expected_embedding = self._get_embedding(expected_output)
        generated_embedding = self._get_embedding(generated_review)
        
        # Require both embeddings to succeed
        if expected_embedding is None or generated_embedding is None:
            return DimensionResult(
                dimension_name=self.name,
                score=0.0,
                passed=False,
                details={
                    "error": "Failed to generate embeddings",
                    "ollama_available": self._ollama_available,
                },
                status="infrastructure_error",
            )
        
        # Calculate cosine similarity
        similarity_score = self._cosine_similarity(expected_embedding, generated_embedding)
        
        return DimensionResult(
            dimension_name=self.name,
            score=similarity_score,
            passed=True, 
            details={
                "similarity_method": "cosine",
                "embedding_model": self.EMBEDDING_MODEL,
                "expected_length": len(expected_output),
                "generated_length": len(generated_review),
                "embedding_dim": len(expected_embedding) if expected_embedding else 0,
            }
        )


CodeReviewDimension = ReferenceReviewSimilarityDimension
