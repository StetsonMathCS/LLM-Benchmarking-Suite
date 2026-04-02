"""
End-to-end tests for the PartialTransformationDimension.
"""
import pytest
from benchmarks.dimensions.partial_transform import PartialTransformationDimension
from core.base import DimensionResult


@pytest.fixture
def dimension():
    return PartialTransformationDimension()


# ── count_occurrences ─────────────────────────────────────────────────────

class TestCountOccurrences:
    def test_basic_count(self):
        assert PartialTransformationDimension.count_occurrences("for for for", "for") == 3

    def test_case_insensitive(self):
        assert PartialTransformationDimension.count_occurrences("For FOR for", "for") == 3

    def test_no_match(self):
        assert PartialTransformationDimension.count_occurrences("hello", "world") == 0

    def test_special_chars_escaped(self):
        assert PartialTransformationDimension.count_occurrences("a.b a.b", "a.b") == 2

    def test_empty_code(self):
        assert PartialTransformationDimension.count_occurrences("", "for") == 0


# ── extract_ast_nodes ─────────────────────────────────────────────────────

class TestExtractAstNodes:
    def test_valid_python_code(self):
        code = "for i in range(10):\n    print(i)\n"
        nodes, err = PartialTransformationDimension.extract_ast_nodes(code)
        assert err is None
        assert "For" in nodes
        assert "Module" in nodes

    def test_list_comprehension(self):
        code = "result = [i for i in range(10)]\n"
        nodes, err = PartialTransformationDimension.extract_ast_nodes(code)
        assert err is None
        assert "ListComp" in nodes

    def test_syntax_error(self):
        nodes, err = PartialTransformationDimension.extract_ast_nodes("def foo(\n")
        assert err is not None
        assert nodes == set()

    def test_empty_code(self):
        nodes, err = PartialTransformationDimension.extract_ast_nodes("")
        assert err is None
        assert "Module" in nodes


# ── parse_ast_nodes_list ──────────────────────────────────────────────────

class TestParseAstNodesList:
    def test_simple_csv(self):
        result = PartialTransformationDimension.parse_ast_nodes_list("For,If")
        assert result == {"For", "If"}

    def test_dotted_name(self):
        result = PartialTransformationDimension.parse_ast_nodes_list("functools.reduce")
        assert result == {"reduce"}

    def test_colon_format(self):
        result = PartialTransformationDimension.parse_ast_nodes_list("Attribute:append")
        assert result == {"append"}

    def test_empty_string(self):
        result = PartialTransformationDimension.parse_ast_nodes_list("")
        assert result == set()

    def test_none(self):
        result = PartialTransformationDimension.parse_ast_nodes_list(None)
        assert result == set()

    def test_mixed_formats(self):
        result = PartialTransformationDimension.parse_ast_nodes_list("For,functools.reduce,Attribute:append")
        assert result == {"For", "reduce", "append"}


# ── Full evaluation: text transform ───────────────────────────────────────

class TestTextTransformEvaluation:
    def test_perfect_transform(self, dimension):
        original = "for i in range(10):\n    print(i)\n"
        generated = "result = [i for i in range(10)]\nprint(result)\n"
        result = dimension.evaluate(
            language="python",
            original_code=original,
            generated_code=generated,
            transform_from="for",
            transform_to="list comprehension",
        )
        assert isinstance(result, DimensionResult)
        assert 0.0 <= result.score <= 1.0

    def test_no_transform_done(self, dimension):
        code = "for i in range(10):\n    print(i)\n"
        result = dimension.evaluate(
            language="python",
            original_code=code,
            generated_code=code,
            transform_from="for",
            transform_to="while",
        )
        assert isinstance(result, DimensionResult)
        # Original still has "for", transform_to "while" not found
        assert 0.0 <= result.score <= 1.0


# ── Full evaluation: AST transform ───────────────────────────────────────

class TestAstTransformEvaluation:
    def test_for_removed_listcomp_added(self, dimension):
        original = "result = []\nfor i in range(10):\n    result.append(i * 2)\n"
        generated = "result = [i * 2 for i in range(10)]\n"
        result = dimension.evaluate(
            language="python",
            original_code=original,
            generated_code=generated,
            transform_from="for",
            transform_to="list comprehension",
            verify_ast_nodes_removed="For",
            verify_ast_nodes_added="ListComp",
        )
        assert isinstance(result, DimensionResult)
        assert result.score > 0.0
        assert "ast_nodes" in result.details

    def test_ast_node_not_removed(self, dimension):
        code = "for i in range(10):\n    print(i)\n"
        result = dimension.evaluate(
            language="python",
            original_code=code,
            generated_code=code,
            transform_from="for",
            transform_to="while",
            verify_ast_nodes_removed="For",
        )
        # For is still present -> imperfect score
        assert result.score < 1.0

    def test_empty_ast_requirements(self, dimension):
        original = "x = 1\n"
        generated = "y = 1\n"
        result = dimension.evaluate(
            language="python",
            original_code=original,
            generated_code=generated,
            transform_from="x",
            transform_to="y",
        )
        assert isinstance(result, DimensionResult)


# ── Edge cases ────────────────────────────────────────────────────────────

class TestPartialTransformEdgeCases:
    def test_missing_transform_kwargs(self, dimension):
        result = dimension.evaluate(
            language="python",
            original_code="x = 1",
            generated_code="x = 1",
        )
        assert isinstance(result, DimensionResult)
        # Empty transform_from/to defaults lead to counting empty string matches
        assert 0.0 <= result.score <= 1.0

    def test_details_structure(self, dimension):
        original = "for i in range(5):\n    print(i)\n"
        generated = "i = 0\nwhile i < 5:\n    print(i)\n    i += 1\n"
        result = dimension.evaluate(
            language="python",
            original_code=original,
            generated_code=generated,
            transform_from="for",
            transform_to="while",
            verify_ast_nodes_removed="For",
            verify_ast_nodes_added="While",
        )
        assert "text_pattern" in result.details
        assert "ast_nodes" in result.details
        assert "original_ast_nodes" in result.details
        assert "generated_ast_nodes" in result.details
