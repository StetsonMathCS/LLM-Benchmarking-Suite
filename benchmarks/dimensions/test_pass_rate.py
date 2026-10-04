"""Compatibility import for the legacy Test Pass Rate class name.

Legacy TPR scores used aggregate pass-count differences and are not GTE.
"""

from benchmarks.dimensions.generated_test_effectiveness import GeneratedTestEffectivenessDimension


TestPassRateDimension = GeneratedTestEffectivenessDimension

__all__ = ["GeneratedTestEffectivenessDimension", "TestPassRateDimension"]

