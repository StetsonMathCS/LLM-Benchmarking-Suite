"""Compatibility import for the pre-revised Code Generation Tests name.

Historical CGT values are not RTS values and migration code must retain their
legacy label. This alias exists only for Python import compatibility.
"""

from benchmarks.dimensions.reference_test_success import ReferenceTestSuccessDimension


CodeGenerationTestsDimension = ReferenceTestSuccessDimension

__all__ = ["ReferenceTestSuccessDimension", "CodeGenerationTestsDimension"]

