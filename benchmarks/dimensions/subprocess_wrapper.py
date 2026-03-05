# benchmarks/dimensions/subprocess_wrapper.py
import subprocess
import resource
import time
import json
import sys
import os

def main():
    cmd = json.loads(sys.argv[1])  # receive cmd as JSON string
    
    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    
    result = {
        "wall_time":   elapsed,
        "total_cpu":   usage.ru_utime + usage.ru_stime,
        "peak_memory": usage.ru_maxrss,
        "returncode":  proc.returncode,
        "stdout":      proc.stdout,
        "stderr":      proc.stderr,
    }
    print(json.dumps(result))

main()