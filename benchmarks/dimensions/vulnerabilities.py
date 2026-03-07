"""
benchmarks/dimensions/vulnerabilities.py

Dimension that evaluates security vulnerabilities in code.
"""
import subprocess
import tempfile
import json
from pathlib import Path
from core.base import (
    BaseDimension,
    DimensionResult
)
from typing import Optional

class VulnerabilitiesDimension(BaseDimension):
    name = "Vulnerabilities"
    description = "Security issues: bandit (Python), cppcheck (C++), npm audit (JS)."

    def evaluate(self, language: str, original_code: Optional[str], generated_code: str, **kwargs) -> DimensionResult:
        try:
            # If no original code provided, only scan generated code and return 1.0
            if not original_code:
                if language == "python":
                    gen_vulns = self._scan_python(generated_code)
                elif language == "cpp":
                    gen_vulns = self._scan_cpp(generated_code)
                elif language == "javascript":
                    gen_vulns = self._scan_javascript(generated_code)
                else:
                    return DimensionResult(
                        dimension_name=self.name,
                        score=0.0,
                        details={"error": f"Unsupported language: {language}"}
                    )
                return DimensionResult(
                    dimension_name=self.name,
                    score=1.0,
                    details={
                        "generated_vulnerabilities": gen_vulns,
                        "note": "No original code provided for comparison"
                    }
                )
            
            # Standard flow: compare original vs generated
            if language == "python":
                orig_vulns = self._scan_python(original_code)
                gen_vulns = self._scan_python(generated_code)
            elif language == "cpp":
                orig_vulns = self._scan_cpp(original_code)
                gen_vulns = self._scan_cpp(generated_code)
            elif language == "javascript":
                orig_vulns = self._scan_javascript(original_code)
                gen_vulns = self._scan_javascript(generated_code)
            else:
                return DimensionResult(
                    dimension_name=self.name,
                    score=0.0,
                    details={"error": f"Unsupported language: {language}"}
                )

            new_vulns = gen_vulns - orig_vulns
            score = self._calculate_score(orig_vulns, gen_vulns)

            return DimensionResult(
                dimension_name=self.name,
                score=score,
                details={
                    "original_vulnerabilities": orig_vulns,
                    "generated_vulnerabilities": gen_vulns,
                    "new_vulnerabilities": new_vulns,
                    "introduced_new_issues": new_vulns > 0
                }
            )
        except Exception as e:
            return DimensionResult(
                dimension_name=self.name,
                score=0.0,
                details={"error": str(e)}
            )

    @staticmethod
    def _scan_python(code: str) -> int:
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                f.flush()
                filename = f.name

            result = subprocess.run(
                ["bandit", "-f", "json", filename],
                capture_output=True,
                text=True,
                timeout=10
            )
            Path(filename).unlink()

            try:
                data = json.loads(result.stdout)
                return len(data.get('results', []))
            except json.JSONDecodeError:
                return 0
        except FileNotFoundError:
            return 0
        except Exception:
            return 0

    @staticmethod
    def _scan_cpp(code: str) -> int:
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
                f.write(code)
                f.flush()
                filename = f.name

            result = subprocess.run(
                ["cppcheck", "--enable=security", filename],
                capture_output=True,
                text=True,
                timeout=10
            )
            Path(filename).unlink()

            output = result.stdout + result.stderr
            high_severity = output.count('error:')
            medium_severity = output.count('warning:')
            
            return high_severity + medium_severity
        except FileNotFoundError:
            return 0
        except Exception:
            return 0

    @staticmethod
    def _scan_javascript(code: str) -> int:
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                f.write(code)
                f.flush()
                filename = f.name

            result = subprocess.run(
                ["eslint", filename, "--format=json"],
                capture_output=True,
                text=True,
                timeout=10
            )
            Path(filename).unlink()

            try:
                data = json.loads(result.stdout)
                if isinstance(data, list) and len(data) > 0:
                    messages = data[0].get('messages', [])
                    security_issues = [m for m in messages if 'security' in m.get('message', '').lower()]
                    return len(security_issues) if security_issues else len(messages)
                return 0
            except json.JSONDecodeError:
                return 0
        except FileNotFoundError:
            return 0
        except Exception:
            return 0

    @staticmethod
    def _calculate_score(original_vulns: int, generated_vulns: int) -> float:
        if original_vulns == 0 and generated_vulns == 0:
            return 1.0
        if generated_vulns > original_vulns:
            new_vulns = generated_vulns - original_vulns
            return max(0.0, 1.0 - (new_vulns / max(generated_vulns, 1)))
        if generated_vulns < original_vulns:
            return 1.0
        return 1.0
