"""
benchmarks/dimensions/partial_transform.py

Evaluate the code on transformations occured vs. transformations expected.
Verifies both text pattern transformations and AST node changes.
"""
from core.base import (
    BaseDimension,
    DimensionResult
)

import re
import ast


class PartialTransformationDimension(BaseDimension):
    name = "Partial Transformation"
    description = "Evaluates the quality of code transformation by measuring how many instances were successfully transformed from one pattern to another versus how many instances of the original pattern remain untransformed. Verifies AST node additions and removals."

    @staticmethod
    def count_occurrences(code, word):
        # Case insensitive regex matching with escaped special chars
        escaped_word = re.escape(word)
        matches = [(m.start(), m.group()) for m in re.finditer(escaped_word, code, re.IGNORECASE)]
        return len(matches)

    @staticmethod
    def extract_ast_nodes(code):
        """
        Parse code and extract all AST node types present.
        Returns a tuple of (set of node type names, error message or None).
        """
        try:
            tree = ast.parse(code)
            node_types = set()
            for node in ast.walk(tree):
                node_types.add(type(node).__name__)
            return node_types, None
        except SyntaxError as e:
            return set(), f"SyntaxError: {e}"

    @staticmethod
    def parse_ast_nodes_list(ast_nodes_str):
        """
        Parse comma-separated AST node names from CSV column.
        Examples: "For,If" -> {"For", "If"}
                  "functools.reduce,Lambda" -> {"reduce", "Lambda"}
                  "Attribute:append" -> {"append"}
        """
        if not ast_nodes_str or ast_nodes_str.strip() == "":
            return set()
        
        nodes = set()
        for item in ast_nodes_str.split(","):
            item = item.strip()
            if ":" in item:
                # Handle "Attribute:append" format - extract the attribute name
                nodes.add(item.split(":")[1])
            elif "." in item:
                # Handle "functools.reduce" format - take the last part
                nodes.add(item.split(".")[-1])
            elif item:
                nodes.add(item)
        return nodes

    def evaluate(self, language: str, original_code: str, generated_code: str, **kwargs) -> DimensionResult:
        transform_from = kwargs.get("transform_from", "")
        transform_to = kwargs.get("transform_to", "")
        nodes_to_remove = kwargs.get("verify_ast_nodes_removed", "")
        nodes_to_add = kwargs.get("verify_ast_nodes_added", "")

        try:
            # Text pattern matching (original logic)
            orig_count_to = self.count_occurrences(original_code, transform_to)
            generated_count_from = self.count_occurrences(generated_code, transform_from)
            generated_count_to = self.count_occurrences(generated_code, transform_to)

            text_to_occurrences = (generated_count_to - orig_count_to) if generated_count_to > orig_count_to else 0
            text_from_occurrences = generated_count_from

            # AST node verification
            original_nodes, orig_parse_error = self.extract_ast_nodes(original_code)
            generated_nodes, gen_parse_error = self.extract_ast_nodes(generated_code)
            
            nodes_removed_set = self.parse_ast_nodes_list(nodes_to_remove)
            nodes_added_set = self.parse_ast_nodes_list(nodes_to_add)

            # Check if nodes that should be removed are actually gone
            nodes_correctly_removed = 0
            nodes_incorrectly_remaining = 0
            for node in nodes_removed_set:
                if node not in generated_nodes:
                    nodes_correctly_removed += 1
                else:
                    nodes_incorrectly_remaining += 1

            # Check if nodes that should be added are actually present
            nodes_correctly_added = 0
            nodes_missing = 0
            for node in nodes_added_set:
                if node in generated_nodes:
                    nodes_correctly_added += 1
                else:
                    nodes_missing += 1

            # Calculate scores
            total_removals_expected = len(nodes_removed_set) if nodes_removed_set else 0
            total_additions_expected = len(nodes_added_set) if nodes_added_set else 0

            # Text pattern score
            text_score = 1.0
            if text_to_occurrences + text_from_occurrences > 0:
                text_score = text_to_occurrences / (text_to_occurrences + text_from_occurrences)

            # AST score
            ast_score = 1.0
            total_ast_issues = nodes_incorrectly_remaining + nodes_missing
            if total_ast_issues > 0 or total_removals_expected > 0 or total_additions_expected > 0:
                total_ast_expected = nodes_correctly_removed + nodes_correctly_added + nodes_incorrectly_remaining + nodes_missing
                if total_ast_expected > 0:
                    ast_score = (nodes_correctly_removed + nodes_correctly_added) / total_ast_expected

            # Combined score (if both text and AST verification exist, average them)
            if total_removals_expected > 0 or total_additions_expected > 0:
                # Has AST requirements
                if text_to_occurrences + text_from_occurrences > 0:
                    # Has text requirements too
                    final_score = (text_score + ast_score) / 2
                else:
                    # Only AST requirements
                    final_score = ast_score
            else:
                # Only text requirements
                final_score = text_score

            details = {
                "text_pattern": {
                    "target_added": text_to_occurrences,
                    "source_remaining": text_from_occurrences,
                    "score": text_score,
                },
                "ast_nodes": {
                    "removed_correctly": nodes_correctly_removed,
                    "removed_expected": total_removals_expected,
                    "removed_incorrectly_remaining": nodes_incorrectly_remaining,
                    "added_correctly": nodes_correctly_added,
                    "added_expected": total_additions_expected,
                    "added_missing": nodes_missing,
                    "score": ast_score,
                },
                "original_ast_nodes": list(sorted(original_nodes)),
                "generated_ast_nodes": list(sorted(generated_nodes)),
            }
            if orig_parse_error:
                details["original_code_parse_error"] = orig_parse_error
            if gen_parse_error:
                details["generated_code_parse_error"] = gen_parse_error

            return DimensionResult(
                dimension_name=self.name,
                score=final_score,
                passed=final_score >= 0.9,
                details=details,
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