"""
End-to-end tests for the CodeRunner utility.
"""
import pytest
from utils.code_runner import CodeRunner, _inject_typing_imports


# ── Typing import injection ───────────────────────────────────────────────

class TestInjectTypingImports:
    def test_adds_missing_optional(self):
        code = "def foo(x: Optional[int]): pass"
        result = _inject_typing_imports(code)
        assert "from typing import Optional" in result

    def test_does_not_duplicate_existing_import(self):
        code = "from typing import Optional\ndef foo(x: Optional[int]): pass"
        result = _inject_typing_imports(code)
        # Should only have one import line
        assert result.count("from typing import") == 1

    def test_no_typing_names_unchanged(self):
        code = "x = 1\ny = 2\n"
        assert _inject_typing_imports(code) == code

    def test_multiple_types_injected(self):
        code = "def foo(x: List[Optional[Dict]]): pass"
        result = _inject_typing_imports(code)
        assert "List" in result
        assert "Optional" in result
        assert "Dict" in result


# ── Python execution ──────────────────────────────────────────────────────

class TestRunPython:
    def test_simple_print(self):
        output = CodeRunner.run_python('print("hello")')
        assert output.succeeded and "hello" in output.stdout

    def test_arithmetic(self):
        output = CodeRunner.run_python('print(2 + 3)')
        assert output.succeeded and "5" in output.stdout

    def test_syntax_error(self):
        output = CodeRunner.run_python('def foo(\n')
        assert not output.succeeded and "SyntaxError" in output.stderr

    def test_timeout(self):
        output = CodeRunner.run_python('import time; time.sleep(100)')
        assert output.timed_out

    def test_multiline_code(self):
        code = "x = 10\ny = 20\nprint(x + y)"
        output = CodeRunner.run_python(code)
        assert "30" in output.stdout

    def test_typing_imports_injected(self):
        code = """\
def greet(name: Optional[str] = None) -> str:
    return f"Hello, {name or 'World'}"

print(greet())
"""
        output = CodeRunner.run_python(code)
        assert "Hello, World" in output.stdout


# ── JavaScript execution ──────────────────────────────────────────────────

class TestRunJavaScript:
    def test_simple_log(self):
        output = CodeRunner.run_javascript('console.log("hello");')
        assert "hello" in output.stdout

    def test_arithmetic(self):
        output = CodeRunner.run_javascript('console.log(2 + 3);')
        assert "5" in output.stdout

    def test_syntax_error(self):
        output = CodeRunner.run_javascript('console.log(')
        assert not output.succeeded and "error" in output.stderr.lower()


# ── C++ execution ─────────────────────────────────────────────────────────

class TestRunCpp:
    def test_simple_program(self):
        code = '#include <iostream>\nint main() { std::cout << "hello"; return 0; }'
        output = CodeRunner.run_cpp(code)
        assert "hello" in output.stdout

    def test_compile_error(self):
        code = "int main( { return 0; }"
        output = CodeRunner.run_cpp(code)
        assert output.compiled is False and "error" in output.stderr.lower()


# ── Edge cases ────────────────────────────────────────────────────────────

class TestCodeRunnerEdgeCases:
    def test_empty_python_code(self):
        output = CodeRunner.run_python("")
        # Empty code produces no output, no error
        assert output.succeeded and output.stdout == ""

    def test_exception_in_python(self):
        output = CodeRunner.run_python("raise ValueError('test')")
        assert "ValueError" in output.stderr

    def test_large_output_python(self):
        output = CodeRunner.run_python("print('x' * 10000)")
        assert len(output.stdout) > 0
