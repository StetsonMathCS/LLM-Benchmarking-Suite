"""Code extraction and structured local execution compatibility helpers."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import tempfile

from facets.evaluation.execution import (
    ExecutionResult,
    ExecutionSettings,
    _communicate,
    _safe_environment,
)


_TYPING_NAMES = {
    "Any", "Callable", "ClassVar", "Dict", "FrozenSet", "Generic", "Iterable",
    "Iterator", "List", "Literal", "Mapping", "MutableMapping", "MutableSequence",
    "NamedTuple", "NoReturn", "Optional", "Protocol", "Sequence", "Set", "Tuple",
    "Type", "TypeVar", "Union",
}


def extract_code(text: str) -> str:
    """Return the first fenced code block, or stripped plain text."""
    if not text:
        return ""
    match = re.search(r"```(?:[\w+.-]+)?\s*\n(.*?)```", text, re.DOTALL)
    if not match:
        match = re.search(r"```(?:[\w+.-]+)?\s*(.*?)```", text, re.DOTALL)
    return (match.group(1) if match else text).strip()


def _inject_typing_imports(code: str) -> str:
    """Legacy dataset adapter for HumanEval annotations missing imports."""
    needed = _TYPING_NAMES & set(re.findall(r"\b([A-Z][a-zA-Z]+)\b", code))
    imported: set[str] = set()
    for chunk in re.findall(r"from typing import ([^\n]+)", code):
        imported.update(name.strip() for name in chunk.split(","))
    missing = needed - imported
    return f"from typing import {', '.join(sorted(missing))}\n{code}" if missing else code


class CodeRunner:
    """Structured local runner.

    This backend is not a security sandbox and is retained for controlled
    offline fixtures. Official evaluation config uses the container backend.
    """

    TIMEOUT = 5.0

    @staticmethod
    def _run_files(files: dict[str, str], command: list[str], timeout: float) -> ExecutionResult:
        settings = ExecutionSettings(backend="local", timeout_s=timeout)
        with tempfile.TemporaryDirectory(prefix="facets-code-") as directory:
            workdir = Path(directory)
            for name, content in files.items():
                (workdir / name).write_text(content, encoding="utf-8")
            return _communicate(command, workdir, settings, _safe_environment(workdir))

    @staticmethod
    def run_python(code: str, timeout: float | None = None) -> ExecutionResult:
        source = _inject_typing_imports(code)
        executable = os.fspath(Path(os.sys.executable).resolve())
        result = CodeRunner._run_files(
            {"candidate.py": source}, [executable, "-I", "candidate.py"], timeout or CodeRunner.TIMEOUT
        )
        result.compiled = result.exit_code == 0 or "SyntaxError" not in result.stderr
        return result

    @staticmethod
    def run_javascript(code: str, timeout: float | None = None) -> ExecutionResult:
        node = shutil.which("node")
        if not node:
            return ExecutionResult(backend="local", diagnostic="node executable not found")
        return CodeRunner._run_files(
            {"candidate.js": code}, [node, "--max-old-space-size=128", "candidate.js"],
            timeout or CodeRunner.TIMEOUT,
        )

    @staticmethod
    def run_cpp(code: str, timeout: float | None = None) -> ExecutionResult:
        compiler = shutil.which("g++")
        if not compiler:
            return ExecutionResult(backend="local", compiled=False, diagnostic="g++ executable not found")
        settings = ExecutionSettings(backend="local", timeout_s=timeout or CodeRunner.TIMEOUT)
        with tempfile.TemporaryDirectory(prefix="facets-cpp-") as directory:
            workdir = Path(directory)
            (workdir / "candidate.cpp").write_text(code, encoding="utf-8")
            compile_result = _communicate(
                [compiler, "candidate.cpp", "-O2", "-o", "candidate"],
                workdir,
                ExecutionSettings(backend="local", timeout_s=10.0),
                _safe_environment(workdir),
            )
            if not compile_result.succeeded:
                compile_result.compiled = False
                return compile_result
            run_result = _communicate(
                [os.fspath(workdir / "candidate")], workdir, settings, _safe_environment(workdir)
            )
            run_result.compiled = True
            return run_result
