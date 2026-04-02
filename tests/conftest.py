"""
Shared fixtures and helpers for the test suite.
"""
import pytest
from unittest.mock import MagicMock, patch
from core.base import (
    BaseProvider,
    BaseBenchmark,
    BenchmarkResult,
    BenchmarkStatus,
    DimensionResult,
    LLMResponse,
    ModelConfig,
)


class FakeProvider(BaseProvider):
    """A mock LLM provider that returns pre-configured responses."""

    def __init__(self, response_content: str = "", error: str = None):
        config = ModelConfig(
            provider="fake",
            model_name="fake-model",
            system_prompt="You are a helpful coding assistant.",
        )
        super().__init__(config)
        self._response_content = response_content
        self._error = error

    def connect(self) -> bool:
        return True

    def complete(self, prompt: str, system_prompt=None) -> LLMResponse:
        return LLMResponse(
            content=self._response_content,
            model="fake-model",
            provider="fake",
            prompt_tokens=10,
            completion_tokens=20,
            error=self._error,
        )

    def is_available(self) -> bool:
        return True


@pytest.fixture
def fake_provider():
    """Return a factory that creates FakeProvider with given content."""
    def _factory(content: str = "", error: str = None):
        return FakeProvider(response_content=content, error=error)
    return _factory


# ---------------------------------------------------------------------------
# Sample code snippets for tests
# ---------------------------------------------------------------------------

PYTHON_HELLO = 'print("Hello, World!")'

PYTHON_ADD = """\
def add(a, b):
    return a + b

print(add(2, 3))
"""

PYTHON_ADD_BUGGY = """\
def add(a, b):
    return a - b

print(add(2, 3))
"""

PYTHON_ADD_FIXED = """\
def add(a, b):
    return a + b

print(add(2, 3))
"""

PYTHON_SNAKE_CASE = """\
def calculate_sum(first_value, second_value):
    total_sum = first_value + second_value
    return total_sum

result = calculate_sum(10, 20)
print(result)
"""

PYTHON_MIXED_NAMING = """\
def calculateSum(first_value, secondValue):
    totalSum = first_value + secondValue
    return totalSum

result = calculateSum(10, 20)
print(result)
"""

PYTHON_WITH_FOR_LOOP = """\
result = []
for i in range(10):
    result.append(i * 2)
print(result)
"""

PYTHON_WITH_LIST_COMP = """\
result = [i * 2 for i in range(10)]
print(result)
"""

PYTHON_FIBONACCI = """\
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(10))
"""

PYTHON_SIMPLE_TEST = """\
def add(a, b):
    return a + b
"""

PYTHON_SIMPLE_TEST_CODE = """\
assert add(1, 2) == 3
assert add(0, 0) == 0
assert add(-1, 1) == 0
"""

PYTHON_UNITTEST_CODE = """\
import unittest

class TestAdd(unittest.TestCase):
    def test_add_positive(self):
        self.assertEqual(add(1, 2), 3)

    def test_add_zero(self):
        self.assertEqual(add(0, 0), 0)

    def test_add_negative(self):
        self.assertEqual(add(-1, 1), 0)

if __name__ == "__main__":
    unittest.main()
"""

JS_HELLO = 'console.log("Hello, World!");'

JS_ADD = """\
function add(a, b) {
    return a + b;
}
console.log(add(2, 3));
"""

CPP_HELLO = """\
#include <iostream>
int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
"""

PYTHON_VULNERABLE = """\
import os
user_input = "test"
os.system(user_input)
"""

PYTHON_SAFE = """\
import subprocess
user_input = "test"
subprocess.run([user_input], check=True)
"""
