"""
benchmarks/dimensions/partial_transform.py

Evaluate the code on transformations occured vs. transformations expected.
"""
from core.base import (
    BaseDimension,
    DimensionResult
)

import re


class PartialTransformationDimension(BaseDimension):
    name = "Partial Transformation"
    description = "Evaluates the quality of code transformation by measuring how many instances were successfully transformed from one pattern to another versus how many instances of the original pattern remain untransformed."

    @staticmethod
    def count_occurrences(code, word):
        # Case insensitive
        matches = [(m.start(), m.group()) for m in re.finditer(word, code, re.IGNORECASE)]
        return len(matches)

    def evaluate(self, language: str, original_code: str, generated_code: str, **kwargs) -> DimensionResult:
        transform_from = kwargs["transform_from"]
        transform_to = kwargs["transform_to"]

        try:
            orig_count_to = PartialTransformationDimension.count_occurrences(original_code, transform_to)

            generated_count_from = PartialTransformationDimension.count_occurrences(generated_code, transform_from)
            generated_count_to = PartialTransformationDimension.count_occurrences(generated_code, transform_to)

            to_occurrences = (generated_count_to - orig_count_to) # Good
            from_occurrences = (generated_count_from) # Bad

            # Good transformations / Total issues
            if to_occurrences + from_occurrences == 0:
                score = 1.0  # Perfect (nothing to transform)
            else:
                score = to_occurrences / (to_occurrences + from_occurrences)
            
            return DimensionResult(
                dimension_name=self.name,
                score=score,
                passed=True,
                details={
                    "to_occurrences" : to_occurrences,
                    "form_occurrences" : from_occurrences,
                }
            )
        except Exception as e:
            return DimensionResult(
                dimension_name=self.name,
                score=0.0,
                passed=False,
                details={
                    "error": str(e),
                }
            )