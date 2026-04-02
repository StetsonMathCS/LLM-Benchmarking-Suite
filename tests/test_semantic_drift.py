"""
End-to-end tests for the SemanticDriftDimension.
"""
import pytest
from benchmarks.dimensions.semantic_drift import SemanticDriftDimension
from core.base import DimensionResult


@pytest.fixture
def dimension():
    return SemanticDriftDimension()


# ── Python similarity ─────────────────────────────────────────────────────

class TestPythonSemanticDrift:
    def test_identical_code_scores_one(self, dimension):
        code = "def add(a, b):\n    return a + b\n"
        result = dimension.evaluate(
            language="python",
            original_code=code,
            generated_code=code,
        )
        assert isinstance(result, DimensionResult)
        assert result.score == 1.0

    def test_similar_code_scores_high(self, dimension):
        orig = "def add(a, b):\n    return a + b\n"
        gen = "def add(x, y):\n    return x + y\n"
        result = dimension.evaluate(
            language="python",
            original_code=orig,
            generated_code=gen,
        )
        assert result.score > 0.5

    def test_completely_different_code_scores_low(self, dimension):
        orig = "x = 1\n"
        gen = """\
import os
import sys
class Foo:
    def __init__(self):
        self.bar = []
    def baz(self, x, y, z):
        for i in range(x):
            if i % 2 == 0:
                self.bar.append(i)
        return self.bar
"""
        result = dimension.evaluate(
            language="python",
            original_code=orig,
            generated_code=gen,
        )
        assert result.score < 0.5

    def test_empty_code_does_not_crash(self, dimension):
        result = dimension.evaluate(
            language="python",
            original_code="",
            generated_code="",
        )
        assert isinstance(result, DimensionResult)


# ── Safe parsing ──────────────────────────────────────────────────────────

class TestSafeParsePython:
    def test_valid_code_parses(self):
        tree = SemanticDriftDimension.safe_parse_python("x = 1\n")
        import ast
        assert isinstance(tree, ast.Module)

    def test_indentation_error_falls_back(self):
        bad = "    x = 1\n  y = 2\n"
        tree = SemanticDriftDimension.safe_parse_python(bad)
        import ast
        assert isinstance(tree, ast.Module)

    def test_syntax_error_returns_pass_ast(self):
        bad = "def foo(\n"
        tree = SemanticDriftDimension.safe_parse_python(bad)
        import ast
        assert isinstance(tree, ast.Module)

    def test_markdown_block_extracted(self):
        md = "```python\nx = 1\n```\n"
        tree = SemanticDriftDimension.safe_parse_python(md)
        import ast
        assert isinstance(tree, ast.Module)


# ── Extract code block ────────────────────────────────────────────────────

class TestExtractCodeBlock:
    def test_no_markdown_returns_original(self):
        code = "x = 1"
        assert SemanticDriftDimension.extract_code_block(code) == code

    def test_python_block_extracted(self):
        md = "```python\nx = 1\n```"
        assert SemanticDriftDimension.extract_code_block(md, "python") == "x = 1"

    def test_generic_block_extracted(self):
        md = "```\nx = 1\n```"
        assert SemanticDriftDimension.extract_code_block(md) == "x = 1"


# ── C++ similarity ────────────────────────────────────────────────────────

class TestCppSemanticDrift:
    def test_identical_cpp_scores_one(self, dimension):
        code = "int main() { return 0; }\n"
        result = dimension.evaluate(
            language="cpp",
            original_code=code,
            generated_code=code,
        )
        assert result.score == 1.0

    def test_similar_cpp_scores_high(self, dimension):
        orig = "int main() { int x = 1; return x; }\n"
        gen = "int main() { int y = 1; return y; }\n"
        result = dimension.evaluate(
            language="cpp",
            original_code=orig,
            generated_code=gen,
        )
        assert result.score > 0.5


# ── Unsupported language ─────────────────────────────────────────────────

class TestSemanticDriftEdgeCases:
    def test_unsupported_language(self, dimension):
        result = dimension.evaluate(
            language="rust",
            original_code="fn main() {}",
            generated_code="fn main() {}",
        )
        assert result.score == 0.0
        assert not result.passed


# ── AST bracket conversion ───────────────────────────────────────────────

class TestAstToBracket:
    def test_simple_expression(self):
        import ast
        tree = ast.parse("x = 1")
        bracket = SemanticDriftDimension.ast_to_bracket(tree)
        assert bracket.startswith("{Module")
        assert "Assign" in bracket

    def test_consistent_output(self):
        import ast
        tree = ast.parse("x = 1")
        b1 = SemanticDriftDimension.ast_to_bracket(tree)
        tree2 = ast.parse("x = 1")
        b2 = SemanticDriftDimension.ast_to_bracket(tree2)
        assert b1 == b2
