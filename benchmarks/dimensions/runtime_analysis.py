"""
benchmarks/dimensions/runtime_analysis.py

Dimension that tests the time and memory complexities of the code.
Scoring Mechanism:
same            → 0.5
2x better       → 0.8889
2x worse        → 0.1111
10x better      → 0.999
10x worse       → 0.001
both zero       → 1.0
orig zero       → 0.0
"""
import time
import tracemalloc
import sys
from core.base import (
    BaseDimension,
    DimensionResult
)
import resource
import os
import subprocess
import tempfile
import json 
import math
from typing import Optional

class RuntimeAnalysisDimension(BaseDimension):
    name = "Runtime Analysis"
    desc = "Memory and execution time delta between original and generated code."
    TIMEOUT = 5
    
    @staticmethod
    def subprocess_run(cmd, env=None):
        """Run a code snippet with memory and time tracking."""
        try:
            wrapper = subprocess.run(
                [sys.executable, "benchmarks/dimensions/subprocess_wrapper.py", json.dumps(cmd)],
                env={**os.environ, **(env or {})},
                capture_output=True,
                text=True,
                timeout=5,
            )
            return json.loads(wrapper.stdout)
        except Exception as e:
            raise RuntimeError(f"Error occurred while evaluating on runtime analysis: {e}")

    @staticmethod
    def run_python(code):
        return RuntimeAnalysisDimension.subprocess_run(
            cmd=[sys.executable, '-c', code], env={"PYTHONDONTWRITEBYTECODE":"1"}
        )

    @staticmethod
    def run_cpp(code):
        with tempfile.NamedTemporaryFile(suffix='.cpp', delete=False) as src:
            src.write(code.encode())
            src_path = src.name
        out_path = src_path.replace('.cpp','.out')
        try:
            compile_result = subprocess.run(
                ['g++', src_path, '-o', out_path],
                capture_output=True, text=True, timeout=10
            )
            if compile_result.returncode!=0:
                return compile_result.stderr
            return RuntimeAnalysisDimension.subprocess_run([out_path])
        except subprocess.TimeoutExpired:
            return "Error: Compilation timed out."
        finally:
            for p in [src_path, out_path]:
                if os.path.exists(p): os.unlink(p)

    @staticmethod
    def run_javascript(code):
        return RuntimeAnalysisDimension.subprocess_run(['node', '--max-old-space-size=128', '-e', code])

    def evaluate(self, language: str, original_code: Optional[str], generated_code: str, **kwargs) -> DimensionResult:
        try:
            # If no original code, just analyze generated code
            if not original_code:
                if language == "python":
                    generated_results = RuntimeAnalysisDimension.run_python(generated_code)
                elif language == "cpp":
                    generated_results = RuntimeAnalysisDimension.run_cpp(generated_code)
                elif language == "javascript":
                    generated_results = RuntimeAnalysisDimension.run_javascript(generated_code)
                else:
                    return DimensionResult(
                        dimension_name=self.name,
                        score=0.0,
                        details={"error": "Evaluation error"}
                    )
                
                if not generated_results.get("stdout"):
                    return DimensionResult(
                        dimension_name=self.name,
                        score=0.0,
                        passed=False,
                        details={
                            "error": generated_results.get("stderr") or "Error occurred",
                        }
                    )
                
                return DimensionResult(
                    dimension_name=self.name,
                    score=1.0,  # No comparison, return perfect score with metrics
                    passed=True,
                    details={
                        "generated_code_runtime": generated_results["wall_time"],
                        "generated_code_memory": generated_results["peak_memory"],
                        "note": "No original code provided for comparison"
                    }
                )
            
            # Standard flow: compare original vs generated
            if language=="python":
                orig_results=RuntimeAnalysisDimension.run_python(original_code)
                generated_results=RuntimeAnalysisDimension.run_python(generated_code)
            elif language=="cpp":
                orig_results=RuntimeAnalysisDimension.run_cpp(original_code)
                generated_results=RuntimeAnalysisDimension.run_cpp(generated_code)
            elif language=="javascript":
                orig_results=RuntimeAnalysisDimension.run_javascript(original_code)
                generated_results=RuntimeAnalysisDimension.run_javascript(generated_code)
            else:
                return DimensionResult(
                    dimension_name=self.name,
                    score=0.0,
                    details={"error":"Evaluation error"}
                )
            if not generated_results["stdout"]:
                return DimensionResult(
                    dimension_name=self.name,
                    score=0.0,
                    passed=False,
                    details={
                        "error" : generated_results["stderr"] or "Error occured",
                    }
                )
            def ratio_score(v1, v2):
                """
                Score based on ratio of generated (v2) to original (v1).
                - v2 < v1 (improvement): score > 0.5, approaches 1.0
                - v2 == v1 (same):        score = 0.5  (neutral, not 1.0)
                - v2 > v1 (regression):   score < 0.5, approaches 0.0
                """
                if v1 == 0 and v2 == 0:
                    return 1.0
                if v1 == 0:
                    return 0.0
                
                ratio = v2 / v1  # <1 = better, >1 = worse
                # log ratio: negative = improvement, positive = regression
                # sigmoid maps this smoothly to (0, 1)
                log_ratio = math.log(ratio)  
                score = 1 / (1 + math.exp(3 * log_ratio))  # 3 controls steepness
                return round(score, 4)

            memory_score = ratio_score(
                orig_results["peak_memory"],
                generated_results["peak_memory"]
            )
            time_score = ratio_score(
                orig_results["wall_time"],
                generated_results["wall_time"]  
            )
            # final score will be normalized sum of memory_score and time_score
            final_score = (0.5*memory_score) + (0.5*time_score) 
            return DimensionResult(
                dimension_name=self.name,
                score = final_score,
                passed=True,
                details={
                    "original_code_results" : orig_results,
                    "generated_code_results": generated_results
                }
            )
        except Exception as e:
            return DimensionResult(
                dimension_name=self.name,
                score=0.0,
                details={"error": str(e)}
            )