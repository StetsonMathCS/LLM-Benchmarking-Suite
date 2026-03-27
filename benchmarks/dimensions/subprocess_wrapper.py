# benchmarks/dimensions/subprocess_wrapper.py
import subprocess
import time
import json
import sys
import os

# Optional import for Unix-like systems
try:
    import resource
    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False

def main():
    cmd = json.loads(sys.argv[1])  # receive cmd as JSON string
    
    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    
    # Get metrics if resource module available (Unix/Linux/WSL)
    if HAS_RESOURCE:
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu_time = usage.ru_utime + usage.ru_stime
        peak_memory = usage.ru_maxrss
    else:
        # Windows fallback - use wall_time as approximation
        cpu_time = elapsed
        peak_memory = 0  # Not available on Windows
    
    result = {
        "wall_time":   elapsed,
        "total_cpu":   cpu_time,
        "peak_memory": peak_memory,
        "returncode":  proc.returncode,
        "stdout":      proc.stdout,
        "stderr":      proc.stderr,
    }
    print(json.dumps(result))

main()