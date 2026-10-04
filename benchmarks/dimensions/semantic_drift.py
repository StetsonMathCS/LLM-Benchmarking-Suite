"""
benchmarks/dimensions/semantic_drift.py

Dimension that tests the code snippet for semantic drift.
Returns the percentage similarity.
"""
from core.base import (
    BaseDimension,
    DimensionResult
)
import ast
import textwrap
import re
from apted import APTED, Config
from apted.helpers import Tree
from clang.cindex import Index, Cursor
from tree_sitter import Language, Parser
import tree_sitter_cpp as tscpp
import subprocess
import tempfile
import json
from typing import Optional

class StructuralSimilarityDimension(BaseDimension):

    dimension_id = "structural_similarity"
    name = "Structural Similarity (SS)"
    description = "AST/tree structural similarity heuristic; it does not establish semantic equivalence."
    _CPP_LANGUAGE = Language(tscpp.language())
    _cpp_parser = Parser(_CPP_LANGUAGE)

    @staticmethod
    def extract_code_block(code: str, language: str = None) -> str:
        """Extract code from markdown blocks if present. Works for any language."""
        if "```" not in code:
            return code
        
        try:
            # Try language-specific block first (```python, ```cpp, etc.)
            if language:
                pattern = rf'```{language}\n(.*?)\n```'
                match = re.search(pattern, code, re.DOTALL)
                if match:
                    return match.group(1)
            
            # Try generic code block (```)
            match = re.search(r'```(?:\w+)?\n(.*?)\n```', code, re.DOTALL)
            if match:
                return match.group(1)
        except Exception:
            pass
        
        return code

    @staticmethod
    def safe_parse_python(code: str) -> ast.Module:
        """Safely parse Python code, handling indentation errors gracefully."""
        # Extract markdown code block if present
        code = StructuralSimilarityDimension.extract_code_block(code, "python")
        
        # Strategy 1: Try as-is
        try:
            return ast.parse(code)
        except (IndentationError, SyntaxError):
            pass
        
        # Strategy 2: Dedent
        try:
            dedented = textwrap.dedent(code)
            return ast.parse(dedented)
        except (IndentationError, SyntaxError):
            pass
        
        # Strategy 3: Strip and dedent
        try:
            stripped = code.strip()
            dedented = textwrap.dedent(stripped)
            return ast.parse(dedented)
        except (IndentationError, SyntaxError):
            pass
        
        # Strategy 4: Wrap in a function if it looks like a code fragment
        try:
            wrapped = f"def _temp():\n{textwrap.indent(code, '    ')}"
            ast.parse(wrapped)
            # If wrapping worked, return the original dedented code as Module
            return ast.parse(textwrap.dedent(code.strip()) or "pass")
        except (IndentationError, SyntaxError):
            pass
        
        # Strategy 5: Try wrapping in class definition
        try:
            wrapped = f"class _Temp:\n{textwrap.indent(code.strip(), '    ')}"
            ast.parse(wrapped)
            return ast.parse("pass")  # Return minimal valid code
        except (IndentationError, SyntaxError):
            pass
        
        # Strategy 6: Ultimate fallback - return minimal valid AST
        # This ensures we always have a parseable result, even for broken code
        return ast.parse("pass")


    @staticmethod
    def ast_to_bracket(node) -> str:
        name = type(node).__name__
        children = list(ast.iter_child_nodes(node))
        if not children:
            return f"{{{name}}}"
        child_str = "".join(StructuralSimilarityDimension.ast_to_bracket(c) for c in children)
        return f"{{{name}{child_str}}}"

    @staticmethod
    def measure_similarity_python(original_code: str, generated_code: str) -> float:
        """Measures the percentage of semantic similarity in python code. returns a score from 0.0 to 1.0"""
        try:
            orig_code = StructuralSimilarityDimension.safe_parse_python(original_code)
            gen_code = StructuralSimilarityDimension.safe_parse_python(generated_code)
        except SyntaxError as e:
            raise ValueError(f"Failed to parse code: {e}")
        
        orig_children = StructuralSimilarityDimension.ast_to_bracket(orig_code)
        gen_children = StructuralSimilarityDimension.ast_to_bracket(gen_code)
        tree1 = Tree.from_text(orig_children)
        tree2 = Tree.from_text(gen_children)
        # compute the tree edit distance
        apted_inst = APTED(tree1, tree2, Config())
        distance = apted_inst.compute_edit_distance()

        # Normalize: max dist is the sum of both the tree sizes
        size1 = sum(1 for _ in ast.walk(orig_code))
        size2 = sum(1 for _ in ast.walk(gen_code))

        max_distance = size1 + size2

        return 1-(distance/max_distance) if max_distance>0 else 1.0

    @staticmethod
    def measure_similarity_cpp(original_code: str, generated_code: str) -> float:
        """Measures semantic similarity between two C++ snippets. Returns a score from 0.0 to 1.0."""
        # Extract markdown code blocks if present
        original_code = StructuralSimilarityDimension.extract_code_block(original_code, "cpp")
        generated_code = StructuralSimilarityDimension.extract_code_block(generated_code, "cpp")
        
        def node_to_bracket(node) -> str:
            node_type = node.type.replace("{", "").replace("}", "")
            children = [c for c in node.children if not c.is_extra]
            if not children:
                return f"{{{node_type}}}"
            return f"{{{node_type}{''.join(node_to_bracket(c) for c in children)}}}"

        def count_nodes(node) -> int:
            children = [c for c in node.children if not c.is_extra]
            return 1 + sum(count_nodes(c) for c in children)

        root1 = StructuralSimilarityDimension._cpp_parser.parse(original_code.encode()).root_node
        root2 = StructuralSimilarityDimension._cpp_parser.parse(generated_code.encode()).root_node

        tree1 = Tree.from_text(node_to_bracket(root1))
        tree2 = Tree.from_text(node_to_bracket(root2))

        distance = APTED(tree1, tree2, Config()).compute_edit_distance()

        size1 = count_nodes(root1)
        size2 = count_nodes(root2)
        max_distance = size1 + size2

        return 1 - (distance / max_distance) if max_distance > 0 else 1.0
    
    @staticmethod
    def js_ast_to_bracket(node):
        if isinstance(node, dict):
            label = node.get("type", "X")
            children = "".join(
                StructuralSimilarityDimension.js_ast_to_bracket(v)
                for v in node.values()
                if isinstance(v, (dict, list))
            )
            return f"{{{label}{children}}}"
        elif isinstance(node, list):
            return "".join(StructuralSimilarityDimension.js_ast_to_bracket(item) for item in node)
        return ""

    @staticmethod
    def parse_js(code: str, tmp_path: str) -> dict:
        """Write code to a temp file and parse it with esprima to avoid string escaping issues."""
        with open(tmp_path, "w", encoding='utf-8') as f:
            f.write(code)
        tmp_path_escaped = tmp_path.replace('\\', '\\\\')
        result = subprocess.run(
            ["node", "-e", f"""
                const esprima = require('esprima');
                const fs = require('fs');
                const code = fs.readFileSync('{tmp_path_escaped}', 'utf8');
                console.log(JSON.stringify(esprima.parseScript(code)));
            """],
            capture_output=True, text=True, encoding='utf-8'
        )
        if result.returncode != 0 or not result.stdout.strip():
            raise ValueError(f"esprima parse error:\n{result.stderr.strip()}")
        return json.loads(result.stdout)

    @staticmethod
    def measure_similarity_javascript(original_code: str, generated_code: str) -> float:
            """Measures semantic similarity between two Javascript code snippets. Returns a score from 0.0 to 1.0"""
            # Extract markdown code blocks if present
            original_code = StructuralSimilarityDimension.extract_code_block(original_code, "javascript")
            generated_code = StructuralSimilarityDimension.extract_code_block(generated_code, "javascript")

            # Install esprima in nodejs
            subprocess.run(['npm','install','esprima'], capture_output=True)
            with tempfile.NamedTemporaryFile(suffix='_one.js', delete=False) as f1, \
                 tempfile.NamedTemporaryFile(suffix='_two.js', delete=False) as f2:
                tmp1, tmp2 = f1.name, f2.name
            bracket1 = StructuralSimilarityDimension.js_ast_to_bracket(StructuralSimilarityDimension.parse_js(original_code, tmp1))
            bracket2 = StructuralSimilarityDimension.js_ast_to_bracket(StructuralSimilarityDimension.parse_js(generated_code, tmp2))
            tree1 = Tree.from_text(bracket1) 
            tree2 = Tree.from_text(bracket2)
            distance = APTED(tree1, tree2, Config()).compute_edit_distance()

            def calc_max_distance(tree):
                children = [c for c in tree.children]
                return 1 + sum(calc_max_distance(c) for c in children)
            max_distance = calc_max_distance(tree1) + calc_max_distance(tree2)
            return 1 - (distance / max_distance) if max_distance > 0 else 1.0

    def evaluate(self, language:str,  original_code: str, generated_code: str, **kwargs) -> DimensionResult:
        try:
            if language=='python':
                result = self.measure_similarity_python(original_code, generated_code)
            elif language=='cpp':
                result = self.measure_similarity_cpp(original_code, generated_code)
            elif language=='javascript':
                result = self.measure_similarity_javascript(original_code, generated_code)
            else:
                raise RuntimeError(f"Cannot benchmark {language} on semantic drift.")
        except ValueError as e:
            # Handle parse errors - return low score rather than failing
            print(f"Warning: Parse error in semantic drift: {e}")
            return DimensionResult(
                dimension_name=self.name,
                score=0.0,  # Return 0 score for unparseable code
                passed=False,  # Mark as failed
                details={"error": str(e)}  # Include error message
            )
        except Exception as e:
            error_msg = f"Error in semantic drift evaluation: {e}"
            print(f"DEBUG: {error_msg}")
            return DimensionResult(
                dimension_name=self.name,
                score=0.0,
                passed=False,
                details={"error": error_msg}
            )        
        return DimensionResult(
            dimension_name=self.name,
            score=result,
        )


# Import compatibility only. Historical "Semantic Drift" reports remain
# historical and are not relabeled during migration.
SemanticDriftDimension = StructuralSimilarityDimension
