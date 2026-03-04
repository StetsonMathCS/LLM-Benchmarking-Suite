import subprocess
import os
import tempfile
import sys

class CodeRunner():
    
    @staticmethod
    def run_cpp(code):
        with tempfile.NamedTemporaryFile(suffix='.cpp', delete=False) as f:
            f.write(code.encode('utf-8'))
            f.flush()
            try:
                compile_result = subprocess.run(
                    ['g++', f.name, '-o', 'a.out'],
                    capture_output=True, text=True
                )
                if compile_result.returncode != 0:
                    return compile_result.stderr
                run_result = subprocess.run(
                    ['./a.out'],
                    capture_output=True, text=True
                )
                return run_result.stdout if run_result.stdout else run_result.stderr
            finally:
                os.unlink(f.name)
                if os.path.exists('a.out'):
                    os.unlink('a.out')

    @staticmethod
    def run_python(code):
        result = subprocess.run(
            [sys.executable, '-c', code],
            capture_output=True, text=True
        )
        return result.stdout if result.stdout else result.stderr

    @staticmethod
    def run_javascript(code):
        result = subprocess.run(
            ['node', '-e', code],
            capture_output=True, text=True
        )
        return result.stdout if result.stdout else result.stderr