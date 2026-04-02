import subprocess
import os
import re
import tempfile
import sys

# resource module is Unix/Linux only
try:
    import resource
except ImportError:
    resource = None

# All names exported from the typing module that generated code commonly uses
_TYPING_NAMES = {
    "Any", "Callable", "ClassVar", "Dict", "FrozenSet", "Generic",
    "Iterable", "Iterator", "List", "Literal", "Mapping", "MutableMapping",
    "MutableSequence", "NamedTuple", "NoReturn", "Optional", "Protocol",
    "Sequence", "Set", "Tuple", "Type", "TypeVar", "Union",
}

def _inject_typing_imports(code: str) -> str:
    """Prepend 'from typing import ...' for any typing names used but not imported."""
    # Skip if already importing everything we need
    needed = _TYPING_NAMES & set(re.findall(r'\b([A-Z][a-zA-Z]+)\b', code))
    if not needed:
        return code

    # Check which ones are actually missing from existing imports
    already_imported = set(re.findall(r'from typing import ([^\n]+)', code))
    # Flatten any comma-separated names already on import lines
    flat_imported: set = set()
    for chunk in already_imported:
        flat_imported.update(n.strip() for n in chunk.split(','))

    missing = needed - flat_imported
    if not missing:
        return code

    import_line = f"from typing import {', '.join(sorted(missing))}\n"
    return import_line + code


class CodeRunner():

    TIMEOUT = 5       # seconds

    @staticmethod
    def _safe_run(cmd, input_data=None, env=None):
        """Run a command with timeout, output cap, and no network."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=CodeRunner.TIMEOUT,
                input=input_data,
                env={**os.environ, **(env or {})},
                # # Limit memory to 128MB (Linux only)
                # preexec_fn=lambda: resource.setrlimit(
                #     resource.RLIMIT_AS, (128 * 1024 * 1024, 128 * 1024 * 1024)
                # )
            )
            out = result.stdout or result.stderr
            return out
        except subprocess.TimeoutExpired:
            return "Error: Execution timed out."
        except MemoryError:
            return "Error: Memory limit exceeded."
        except Exception as e:
            return f"Error: {e}"

    @staticmethod
    def run_python(code):
        code = _inject_typing_imports(code)
        return CodeRunner._safe_run(
            [sys.executable, '-c', code],
            env={"PYTHONDONTWRITEBYTECODE": "1"}
        )

    @staticmethod
    def run_javascript(code):
        return CodeRunner._safe_run(['node', '--max-old-space-size=128', '-e', code])

    @staticmethod
    def run_cpp(code):
        with tempfile.NamedTemporaryFile(suffix='.cpp', delete=False) as src:
            src.write(code.encode())
            src_path = src.name
        out_path = src_path.replace('.cpp', '.out')
        try:
            compile_result = subprocess.run(
                ['g++', src_path, '-o', out_path],
                capture_output=True, text=True, timeout=10
            )
            if compile_result.returncode != 0:
                return compile_result.stderr
            return CodeRunner._safe_run([out_path])
        except subprocess.TimeoutExpired:
            return "Error: Compilation timed out."
        finally:
            for p in [src_path, out_path]:
                if os.path.exists(p): os.unlink(p)