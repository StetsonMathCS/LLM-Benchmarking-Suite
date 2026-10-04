"""
benchmarks/dimensions/code_consistency.py

Tests the given code snippets on consistency.
"""

from core.base import(
    BaseDimension,
    DimensionResult
)
import re
from typing import Optional

class CodeConsistencyDimension(BaseDimension):
    dimension_id = "code_consistency"
    name = "Code Consistency"
    description = "Style-consistency heuristic; it is not a functional-correctness measure."

    @staticmethod
    def _python_check(code):
        """Check python code for inconsistencies."""
        issues = 0
        total_checks = 0
        
        # Check indentation consistency (spaces vs tabs)
        lines = code.split('\n')    
        indent_types = set()
        for line in lines:
            if len(line) > 0 and line[0] in (' ', '\t'):
                if line[0] == ' ':
                    indent_types.add('spaces')
                else:
                    indent_types.add('tabs')
        if len(indent_types) > 1:
            issues += 1
        total_checks += 1
        
        # Check naming conventions (either camelCase or snake_case, not mixed)
        camel_case_vars = re.findall(r'\b[a-z]+[A-Z]\w*\b', code)
        snake_case_vars = re.findall(r'\b[a-z]+_[a-z_]*\b', code)
        if len(camel_case_vars) > 0 and len(snake_case_vars) > 0:
            issues += 1
        total_checks += 1
        
        # Check quote consistency
        single_quotes = code.count("'")
        double_quotes = code.count('"')
        if single_quotes > 0 and double_quotes > 0:
            if abs(single_quotes - double_quotes) > max(single_quotes, double_quotes) * 0.3:
                issues += 1
        total_checks += 1
        
        score = max(0.0, 1.0 - (issues / max(1, total_checks)))
        return score

    @staticmethod
    def _cpp_check(code):
        """Check C++ code for inconsistencies."""
        issues = 0
        total_checks = 0
        
        # Check indentation consistency
        lines = code.split('\n')
        indent_types = set()
        for line in lines:
            if len(line) > 0 and line[0] in (' ', '\t'):
                if line[0] == ' ':
                    indent_types.add('spaces')
                else:
                    indent_types.add('tabs')
        if len(indent_types) > 1:
            issues += 1
        total_checks += 1
        
        # Check brace style consistency (K&R vs Allman)
        kr_style = len(re.findall(r'\)\s*\{', code))
        allman_style = len(re.findall(r'\)\s*\n\s*\{', code))
        if kr_style > 0 and allman_style > 0:
            if abs(kr_style - allman_style) > max(kr_style, allman_style) * 0.3:
                issues += 1
        total_checks += 1
        
        # Check naming conventions (either camelCase or snake_case, not mixed)
        camel_case_vars = re.findall(r'\b[a-z]+[A-Z]\w*\b', code)
        snake_case_vars = re.findall(r'\b[a-z]+_[a-z_]*\b', code)
        if len(camel_case_vars) > 0 and len(snake_case_vars) > 0:
            issues += 1
        total_checks += 1
        
        # Check semicolon consistency
        lines_with_semicolon = sum(1 for line in lines if ';' in line and not line.strip().startswith('//'))
        if lines_with_semicolon < len(lines) * 0.3:
            issues += 1
        total_checks += 1
        
        score = max(0.0, 1.0 - (issues / max(1, total_checks)))
        return score

    @staticmethod
    def _javascript_check(code):
        """Check JS code for inconsistencies."""
        issues = 0
        total_checks = 0
        
        # Check indentation consistency
        lines = code.split('\n')
        indent_types = set()
        for line in lines:
            if len(line) > 0 and line[0] in (' ', '\t'):
                if line[0] == ' ':
                    indent_types.add('spaces')
                else:
                    indent_types.add('tabs')
        if len(indent_types) > 1:
            issues += 1
        total_checks += 1
        
        # Check semicolon consistency
        code_lines = [l.strip() for l in lines if l.strip() and not l.strip().startswith('//')]
        semicolon_lines = sum(1 for line in code_lines if line.endswith(';'))
        if len(code_lines) > 0 and semicolon_lines < len(code_lines) * 0.7:
            issues += 1
        total_checks += 1
        
        # Check naming conventions (either camelCase or snake_case, not mixed)
        camel_case_vars = re.findall(r'\b[a-z]+[A-Z]\w*\b', code)
        snake_case_vars = re.findall(r'\b[a-z]+_[a-z_]*\b', code)
        if len(camel_case_vars) > 0 and len(snake_case_vars) > 0:
            issues += 1
        total_checks += 1
        
        # Check quote consistency
        single_quotes = code.count("'")
        double_quotes = code.count('"')
        if single_quotes > 0 and double_quotes > 0:
            if abs(single_quotes - double_quotes) > max(single_quotes, double_quotes) * 0.3:
                issues += 1
        total_checks += 1
        
        score = max(0.0, 1.0 - (issues / max(1, total_checks)))
        return score

    def evaluate(self, language: str, generated_code: str, original_code: Optional[str] = None, **kwargs) -> DimensionResult:
        # Get the language for the generated code (may differ from source language in translation tasks)
        generated_code_language = kwargs.get('generated_code_language', language).lower().strip()
        
        try:
                
            if generated_code_language == 'python':
                score = self._python_check(generated_code)
            elif generated_code_language in ['cpp', 'c++', 'cxx']:
                score = self._cpp_check(generated_code)
            elif generated_code_language in ['javascript', 'js']:
                score = self._javascript_check(generated_code)
            else:
                return DimensionResult(
                    dimension_name=self.name,
                    score=0.0,
                    passed=False,
                    details={
                        "error": "Language not recognized",
                    }
                )
            
            return DimensionResult(
                        dimension_name=self.name,
                        passed=True,
                        score=score, 
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
