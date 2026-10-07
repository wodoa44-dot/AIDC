from __future__ import annotations
import subprocess,time
from datetime import datetime,timezone

def run_instrumented(command:list[str],timeout_s:float):
    """External, explicitly configured workload; records process timing and exit status."""
    start=time.monotonic_ns();utc=datetime.now(timezone.utc).isoformat()
    proc=subprocess.Popen(command,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:return {'start_monotonic_ns':start,'start_utc':utc,'exit_code':proc.wait(timeout=timeout_s),'end_monotonic_ns':time.monotonic_ns()}
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:proc.wait(timeout=5)
        except subprocess.TimeoutExpired:proc.kill();proc.wait()
        return {'start_monotonic_ns':start,'start_utc':utc,'exit_code':proc.returncode,'timed_out':True,'end_monotonic_ns':time.monotonic_ns()}
