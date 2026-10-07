"""Nohup launch only after matching identity, label validation, and smoke."""
import argparse
import datetime
import fcntl
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from common import ROOT, atomic_json
from labeling import validate_spec
from pipeline import identity
from config import load_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["start", "resume", "status", "stop"])
    p.add_argument("--output", type=Path, default=ROOT / "outputs/prepared_v1")
    p.add_argument("--spec", type=Path, default=ROOT / "response_label_spec.yaml")
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    procfile = out / "background_process.json"
    proc = json.loads(procfile.read_text()) if procfile.exists() else {}

    def start_ticks(pid: int) -> str:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]

    def live() -> bool:
        pid = proc.get("pid")
        if not pid:
            return False
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().decode().replace("\0", " ")
            return (
                str(ROOT / "pipeline.py") in cmd
                and str(out) in cmd
                and proc.get("start_ticks") == start_ticks(pid)
            )
        except OSError:
            return False

    if a.action == "status":
        print(
            json.dumps(
                {
                    "process": proc,
                    "live": live(),
                    "progress": json.loads((out / "progress.json").read_text())
                    if (out / "progress.json").exists()
                    else None,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if a.action == "stop":
        if live():
            atomic_json(
                out / "stop_requested.json",
                {
                    "requested_at": datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat(),
                    "pid": proc["pid"],
                },
            )
            os.killpg(proc["pid"], signal.SIGTERM)
            print("Stop requested; rerun resume only with unchanged identity.")
        else:
            print("No matching live process")
        return
    with open(out / "launch.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if live():
            raise SystemExit("Duplicate process refused")
        ready = validate_spec(load_config(a.spec))
        if not ready["ready"]:
            raise SystemExit(
                "Training blocked: " + json.dumps(ready, ensure_ascii=False)
            )
        if not (out / "identity.json").exists() or json.loads(
            (out / "identity.json").read_text()
        ) != identity(a.spec):
            raise SystemExit("Run preflight in new output for this identity first")
        validation = json.loads((out / "label_validation.json").read_text())
        smoke = json.loads((out / "smoke.json").read_text())
        if not validation["measured_label_validation_passed"] or not smoke["passed"]:
            raise SystemExit("Label validation and smoke must pass")
        (out / "stop_requested.json").unlink(missing_ok=True)
        cmd = [
            "nohup",
            sys.executable,
            "-u",
            str(ROOT / "pipeline.py"),
            "--mode",
            "train",
            "--output",
            str(out),
            "--spec",
            str(a.spec.resolve()),
            "--resume",
        ]
        with open(out / "pipeline.log", "ab") as log:
            child = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                cwd=ROOT.parent.parent,
            )
        proc = {
            "pid": child.pid,
            "start_ticks": start_ticks(child.pid),
            "start_time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "command_line": cmd,
            "stdin": "/dev/null",
            "python_unbuffered": True,
        }
        atomic_json(procfile, proc)
        print(json.dumps(proc, ensure_ascii=False, indent=2))
        # Immediate process and log checks, not a completion claim.
        import time

        time.sleep(0.5)
        print("Process live:", live())
        print((out / "pipeline.log").read_text()[-2000:])


if __name__ == "__main__":
    main()
