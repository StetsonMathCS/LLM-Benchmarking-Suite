"""
benchmarks/dimensions/vulnerabilities.py

Dimension that evaluates security vulnerabilities in code.
"""
import subprocess
import tempfile
import json
import shutil
from pathlib import Path
from core.base import (
    BaseDimension,
    DimensionResult
)
from typing import Optional

class VulnerabilitiesDimension(BaseDimension):
    dimension_id = "vulnerabilities"
    name = "Vulnerabilities"
    description = "Security issues: bandit (Python), cppcheck (C++), npm audit (JS)."

    def evaluate(self, language: str, generated_code: str, original_code: Optional[str] = None, **kwargs) -> DimensionResult:
        try:
            # Get the language for the generated code (may differ from source language in translation tasks)
            generated_code_language = kwargs.get('generated_code_language', language)
            tools = {"python": "bandit", "cpp": "cppcheck", "javascript": "eslint"}
            required = {tools.get(generated_code_language)}
            if original_code:
                required.add(tools.get(language))
            missing = sorted(tool for tool in required if tool and shutil.which(tool) is None)
            if missing:
                return DimensionResult(
                    self.name, 0.0, False,
                    {"error": f"Missing required security tool(s): {', '.join(missing)}", "tools": sorted(required)},
                    dimension_id=self.dimension_id, status="infrastructure_error",
                )
            
            # If no original code provided, only scan generated code and return 1.0
            if not original_code:
                if generated_code_language == "python":
                    gen_vulns = self._scan_python(generated_code)
                elif generated_code_language == "cpp":
                    gen_vulns = self._scan_cpp(generated_code)
                elif generated_code_language == "javascript":
                    gen_vulns = self._scan_javascript(generated_code)
                else:
                    return DimensionResult(
                        dimension_name=self.name,
                        score=0.0,
                        details={"error": f"Unsupported generated code language: {generated_code_language}"}
                    )
                return DimensionResult(
                    dimension_name=self.name,
                    score=1.0,
                    details={
                        "generated_vulnerabilities": gen_vulns,
                        "note": "No original code provided for comparison",
                        "tool": tools[generated_code_language],
                    }
                )
            
            # Standard flow: compare original vs generated
            if language == "python":
                orig_vulns = self._scan_python(original_code)
            elif language == "cpp":
                orig_vulns = self._scan_cpp(original_code)
            elif language == "javascript":
                orig_vulns = self._scan_javascript(original_code)
            else:
                return DimensionResult(
                    dimension_name=self.name,
                    score=0.0,
                    details={"error": f"Unsupported source language: {language}"}
                )
            
            # Scan generated code using its own language
            if generated_code_language == "python":
                gen_vulns = self._scan_python(generated_code)
            elif generated_code_language == "cpp":
                gen_vulns = self._scan_cpp(generated_code)
            elif generated_code_language == "javascript":
                gen_vulns = self._scan_javascript(generated_code)
            else:
                return DimensionResult(
                    dimension_name=self.name,
                    score=0.0,
                    details={"error": f"Unsupported generated code language: {generated_code_language}"}
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
                    "introduced_new_issues": new_vulns > 0,
                    "tools": sorted(required),
                }
            )
        except Exception as e:
            return DimensionResult(
                dimension_name=self.name,
                score=0.0,
                details={"error": str(e)},
                status="infrastructure_error",
            )

    @staticmethod
    def _scan_python(code: str) -> int:
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
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
        except FileNotFoundError as exc:
            raise RuntimeError("bandit executable not found") from exc
        except Exception as exc:
            raise RuntimeError(f"bandit failed: {exc}") from exc

    @staticmethod
    def _scan_cpp(code: str) -> int:
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False, encoding='utf-8') as f:
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
        except FileNotFoundError as exc:
            raise RuntimeError("cppcheck executable not found") from exc
        except Exception as exc:
            raise RuntimeError(f"cppcheck failed: {exc}") from exc

    @staticmethod
    def _scan_javascript(code: str) -> int:
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False, encoding='utf-8') as f:
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
        except FileNotFoundError as exc:
            raise RuntimeError("eslint executable not found") from exc
        except Exception as exc:
            raise RuntimeError(f"eslint failed: {exc}") from exc

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
