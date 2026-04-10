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
import sys
from core.base import (
    BaseDimension,
    DimensionResult
)
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
                    
            # Check if wrapper itself failed
            if wrapper.returncode != 0:
                raise RuntimeError(f"Wrapper failed: {wrapper.stderr}")
            
            # Check if stdout is empty
            if not wrapper.stdout or not wrapper.stdout.strip():
                raise RuntimeError(f"No output from wrapper. stderr: {wrapper.stderr}")
            
            # Try to parse JSON
            try:
                return json.loads(wrapper.stdout)
            except json.JSONDecodeError as je:
                raise RuntimeError(f"Invalid JSON from wrapper: {wrapper.stdout[:100]}. stderr: {wrapper.stderr}")
                
        except subprocess.TimeoutExpired:
            raise RuntimeError("Subprocess execution timed out")
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
            if compile_result.returncode != 0:
                return {
                    "stdout": "",
                    "stderr": compile_result.stderr,
                    "returncode": compile_result.returncode,
                    "wall_time": 0,
                    "total_cpu": 0,
                    "peak_memory": 0
                }
            return RuntimeAnalysisDimension.subprocess_run([out_path])
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": "Compilation timed out.",
                "returncode": 1,
                "wall_time": 0,
                "total_cpu": 0,
                "peak_memory": 0
            }
        finally:
            for p in [src_path, out_path]:
                if os.path.exists(p): os.unlink(p)

    @staticmethod
    def run_javascript(code):
        return RuntimeAnalysisDimension.subprocess_run(['node', '--max-old-space-size=128', '-e', code])

    def evaluate(self, language: str, generated_code: str, original_code: Optional[str] = None, **kwargs) -> DimensionResult:
        try:
            expected_console_output = kwargs.get('expected_console_output', None)
            expected_output = kwargs.get('expected_output', None)

            # Refactoring path: compare expected refactored code vs LLM-generated refactored code
            if expected_console_output is not None and expected_output:
                run_fn = {
                    "python": RuntimeAnalysisDimension.run_python,
                    "cpp": RuntimeAnalysisDimension.run_cpp,
                    "javascript": RuntimeAnalysisDimension.run_javascript,
                }.get(language)
                if run_fn is None:
                    return DimensionResult(
                        dimension_name=self.name,
                        score=0.0,
                        details={"error": f"Unsupported language: {language}"}
                    )
                baseline_results = run_fn(expected_output)
                generated_results = run_fn(generated_code)

                def ratio_score(v1, v2):
                    if v1 == 0 and v2 == 0:
                        return 1.0
                    if v1 == 0:
                        return 0.0
                    ratio = v2 / v1
                    log_ratio = math.log(ratio)
                    score = 1 / (1 + math.exp(3 * log_ratio))
                    return round(score, 4)

                memory_score = ratio_score(
                    baseline_results.get("peak_memory", 0),
                    generated_results.get("peak_memory", 0)
                )
                time_score = ratio_score(
                    baseline_results.get("wall_time", 0),
                    generated_results.get("wall_time", 0)
                )
                final_score = (0.5 * memory_score) + (0.5 * time_score)
                return DimensionResult(
                    dimension_name=self.name,
                    score=final_score,
                    passed=True,
                    details={
                        "expected_refactored_results": baseline_results,
                        "generated_results": generated_results
                    }
                )

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
                    # Code produced no output - return neutral score
                    if generated_results.get("returncode") == 0:
                        return DimensionResult(
                            dimension_name=self.name,
                            score=0.5,
                            passed=True,
                            details={
                                "note": "Code executed but produced no output",
                                "returncode": generated_results.get("returncode")
                            }
                        )
                    else:
                        return DimensionResult(
                            dimension_name=self.name,
                            score=0.0,
                            passed=False,
                            details={
                                "error": generated_results.get("stderr") or "Error occurred",
                                "returncode": generated_results.get("returncode")
                            }
                        )
                
                return DimensionResult(
                    dimension_name=self.name,
                    score=1.0,
                    passed=True,
                    details={
                        "generated_code_runtime": generated_results.get("wall_time"),
                        "generated_code_memory": generated_results.get("peak_memory"),
                        "note": "No original code provided for comparison"
                    }
                )
            
            # Standard flow: compare original vs generated
            if language == "python":
                orig_results = RuntimeAnalysisDimension.run_python(original_code)
                generated_results = RuntimeAnalysisDimension.run_python(generated_code)
            elif language == "cpp":
                orig_results = RuntimeAnalysisDimension.run_cpp(original_code)
                generated_results = RuntimeAnalysisDimension.run_cpp(generated_code)
            elif language == "javascript":
                orig_results = RuntimeAnalysisDimension.run_javascript(original_code)
                generated_results = RuntimeAnalysisDimension.run_javascript(generated_code)
            else:
                return DimensionResult(
                    dimension_name=self.name,
                    score=0.0,
                    details={"error": "Evaluation error"}
                )
            
            if not generated_results.get("stdout"):
                # Generated code produced no output - return neutral score
                if generated_results.get("returncode") == 0:
                    return DimensionResult(
                        dimension_name=self.name,
                        score=0.5,
                        passed=True,
                        details={
                            "note": "Generated code executed but produced no output",
                            "returncode": generated_results.get("returncode")
                        }
                    )
                else:
                    return DimensionResult(
                        dimension_name=self.name,
                        score=0.0,
                        passed=False,
                        details={
                            "error": generated_results.get("stderr") or "Error occurred",
                            "returncode": generated_results.get("returncode")
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
                orig_results.get("peak_memory", 0),
                generated_results.get("peak_memory", 0)
            )
            time_score = ratio_score(
                orig_results.get("wall_time", 0),
                generated_results.get("wall_time", 0)
            )
            # final score will be normalized sum of memory_score and time_score
            final_score = (0.5 * memory_score) + (0.5 * time_score) 
            return DimensionResult(
                dimension_name=self.name,
                score=final_score,
                passed=True,
                details={
                    "original_code_results": orig_results,
                    "generated_code_results": generated_results
                }
            )
        except Exception as e:
            return DimensionResult(
                dimension_name=self.name,
                score=0.0,
                details={"error": str(e)}
            )