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
from apted import APTED, Config
from apted.helpers import Tree
from clang.cindex import Index, Cursor
from tree_edit_distance import edit_distance
from tree_sitter import Language, Parser
import tree_sitter_cpp as tscpp
import subprocess
import json

class SemanticDriftDimension(BaseDimension):

    name = "Semantic Dimension"
    description = "Dimension that tests the code snippet for semantic drift. Returns the percentage similarity"
    _CPP_LANGUAGE = Language(tscpp.language())
    _cpp_parser = Parser(_CPP_LANGUAGE)

    @staticmethod
    def ast_to_bracket(node) -> str:
        name = type(node).__name__
        children = list(ast.iter_child_nodes(node))
        if not children:
            return f"{{{name}}}"
        child_str = "".join(SemanticDriftDimension.ast_to_bracket(c) for c in children)
        return f"{{{name}{child_str}}}"

    @staticmethod
    def measure_similarity_python(original_code: str, generated_code: str) -> float:
        """Measures the percentage of semantic similarity in python code. returns a score from 0.0 to 1.0"""
        orig_code = ast.parse(original_code)
        gen_code = ast.parse(generated_code)
        orig_children = SemanticDriftDimension.ast_to_bracket(orig_code)
        gen_children = SemanticDriftDimension.ast_to_bracket(gen_code)
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
        def node_to_bracket(node) -> str:
            node_type = node.type.replace("{", "").replace("}", "")
            children = [c for c in node.children if not c.is_extra]
            if not children:
                return f"{{{node_type}}}"
            return f"{{{node_type}{''.join(node_to_bracket(c) for c in children)}}}"

        def count_nodes(node) -> int:
            children = [c for c in node.children if not c.is_extra]
            return 1 + sum(count_nodes(c) for c in children)

        root1 = _cpp_parser.parse(original_code.encode()).root_node
        root2 = _cpp_parser.parse(generated_code.encode()).root_node

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
                SemanticDriftDimension.js_ast_to_bracket(v)
                for v in node.values()
                if isinstance(v, (dict, list))
            )
            return f"{{{label}{children}}}"
        elif isinstance(node, list):
            return "".join(SemanticDriftDimension.js_ast_to_bracket(item) for item in node)
        return ""

    @staticmethod
    def parse_js(code: str, tmp_path: str) -> dict:
        """Write code to a temp file and parse it with esprima to avoid string escaping issues."""
        with open(tmp_path, "w") as f:
            f.write(code)
        result = subprocess.run(
            ["node", "-e", f"""
                const esprima = require('esprima');
                const fs = require('fs');
                const code = fs.readFileSync('{tmp_path}', 'utf8');
                console.log(JSON.stringify(esprima.parseScript(code)));
            """],
            capture_output=True, text=True
        )
        if result.returncode != 0 or not result.stdout.strip():
            raise ValueError(f"esprima parse error:\n{result.stderr.strip()}")
        return json.loads(result.stdout)

    @staticmethod
    def measure_similarity_javascript(original_code:str, generated_code:str) -> float:
            """Measures semantic similarity between two Javascript code snippets. Returns a score from 0.0 to 1.0"""
            # Install eprisma in nodejs
            subprocess.run(['npm','install','esprima'])
            bracket1=SemanticDriftDimension.js_ast_to_bracket(parse_js(original_code, "/tmp/one.js"))
            bracket2=SemanticDriftDimension.js_ast_to_bracket(parse_js(generated_code, "/tmp/two.js"))
            tree1=Tree.from_text(bracket1) 
            tree2=Tree.from_text(bracket2)
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
        except Exception as e:
            raise RuntimeError(f"Error occured while semantic drift: {e}")
        return DimensionResult(
            dimension_name=self.name,
            score=result,
        )