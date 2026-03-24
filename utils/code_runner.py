import subprocess
import os
import tempfile
import sys

# resource module is Unix/Linux only
try:
    import resource
except ImportError:
    resource = None

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