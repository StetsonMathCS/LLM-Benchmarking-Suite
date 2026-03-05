"""
benchmarks/dimensions/code_review_dimension.py

Checks if the review covers expected issues/topics
"""
from core.base import(
    BaseDimension,
    DimensionResult
)
import re

class CodeReviewDimension(BaseDimension):
    name="Code Review"
    description=""

    def evaluate(self, language: str, original_code: str, generated_reviews: str, **kwargs) -> DimensionResult:
        expected_keywords = kwargs.get("expected_reviews", []) # expects a list of expected keywords.
        hits = 0
        total = len(expected_keywords)
        for word in expected_keywords:
            total_hits = len(re.findall(word, generated_reviews))
            if total_hits>0:
                hits+=1
        return DimensionResult(
            dimension_name=self.name,
            score=hits/total,
            passed=True,
        )