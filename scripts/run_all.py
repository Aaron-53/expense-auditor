from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


def _stream_output(prefix: str, stream) -> None:
    for raw_line in iter(stream.readline, ""):
        line = raw_line.rstrip()
        if line:
            print(f"[{prefix}] {line}")
    stream.close()


def _launch(name: str, cwd: Path, args: list[str], env: dict[str, str]) -> subprocess.Popen:
    process = subprocess.Popen(
        args,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if process.stdout is not None:
        threading.Thread(target=_stream_output, args=(name, process.stdout), daemon=True).start()
    print(f"[{name}] started with pid {process.pid}")
    return process


def main() -> int:
    env = os.environ.copy()
    env.setdefault("INGESTION_URL", "http://127.0.0.1:8000/ingest")
    env.setdefault("VALIDATION_URL", "http://127.0.0.1:8001/validate-date")
    env.setdefault("DJANGO_ALLOWED_HOSTS", "127.0.0.1,localhost")

    services = [
        _launch(
            "validation",
            ROOT_DIR,
            [PYTHON, "-m", "uvicorn", "services.validation.main:app", "--host", "127.0.0.1", "--port", "8001", "--reload"],
            env,
        ),
        _launch(
            "ingestion",
            ROOT_DIR,
            [PYTHON, "-m", "uvicorn", "services.ingestion.main:app", "--host", "127.0.0.1", "--port", "8000", "--reload"],
            env,
        ),
        _launch(
            "frontend",
            ROOT_DIR / "frontend",
            [PYTHON, "manage.py", "runserver", "127.0.0.1:8002"],
            env,
        ),
    ]

    def _terminate_children() -> None:
        for process in services:
            if process.poll() is None:
                process.terminate()
        deadline = time.time() + 8
        for process in services:
            remaining = max(0.0, deadline - time.time())
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                process.kill()

    try:
        while True:
            exit_codes = [process.poll() for process in services]
            for code in exit_codes:
                if code is not None:
                    if code != 0:
                        print(f"[runner] service exited early with code {code}; stopping the rest")
                        _terminate_children()
                        return code
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("[runner] shutdown requested")
        _terminate_children()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())