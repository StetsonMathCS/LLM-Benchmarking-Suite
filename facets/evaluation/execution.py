"""Bounded process execution used by deterministic FACETS evaluators.

The local backend is intended for trusted/offline fixtures only.  It limits
time and inherited environment data, but it is not a security sandbox.  The
container backend is the revised-v1 real-run default and disables networking.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time
import uuid
from typing import Mapping, Optional, Protocol


OUTPUT_LIMIT = 32_768
SAFE_ENV_KEYS = ("PATH", "LANG", "LC_ALL", "SYSTEMROOT", "WINDIR")


@dataclass(frozen=True)
class ExecutionSettings:
    backend: str = "container"
    timeout_s: float = 10.0
    memory_mb: int = 256
    cpus: float = 1.0
    pids_limit: int = 64
    container_image: str = "facets-evaluator:revised-v1"
    output_limit: int = OUTPUT_LIMIT


@dataclass
class ExecutionResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: Optional[int] = None
    duration_s: float = 0.0
    timed_out: bool = False
    compiled: Optional[bool] = None
    backend: str = "local"
    command: list[str] = field(default_factory=list)
    diagnostic: str = ""

    @property
    def succeeded(self) -> bool:
        return not self.timed_out and self.exit_code == 0 and self.compiled is not False

    def to_dict(self) -> dict:
        return asdict(self)


class ExecutionBackend(Protocol):
    def run_files(
        self, files: Mapping[str, str], command: list[str], *, timeout_s: Optional[float] = None
    ) -> ExecutionResult: ...

    def run_python_files(
        self, files: Mapping[str, str], entry_file: str, *, timeout_s: Optional[float] = None
    ) -> ExecutionResult: ...


_DEFAULT_SETTINGS = ExecutionSettings()


def configure_default(settings: ExecutionSettings) -> None:
    global _DEFAULT_SETTINGS
    _DEFAULT_SETTINGS = settings


def default_settings() -> ExecutionSettings:
    return _DEFAULT_SETTINGS


def _bounded(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n... <truncated {len(value) - limit} characters>"


def _safe_environment(workdir: Path) -> dict[str, str]:
    env = {key: os.environ[key] for key in SAFE_ENV_KEYS if key in os.environ}
    env.update({
        "HOME": str(workdir),
        "TMPDIR": str(workdir),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "NO_PROXY": "*",
        "no_proxy": "*",
    })
    return env


def _communicate(
    command: list[str], workdir: Path, settings: ExecutionSettings, env: dict[str, str],
    cleanup_command: Optional[list[str]] = None,
) -> ExecutionResult:
    start = time.perf_counter()
    proc = subprocess.Popen(
        command,
        cwd=workdir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=settings.timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        if cleanup_command:
            subprocess.run(cleanup_command, capture_output=True, timeout=10, check=False)
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
        stdout, stderr = proc.communicate()
    duration = time.perf_counter() - start
    return ExecutionResult(
        stdout=_bounded(stdout or "", settings.output_limit),
        stderr=_bounded(stderr or "", settings.output_limit),
        exit_code=None if timed_out else proc.returncode,
        duration_s=duration,
        timed_out=timed_out,
        backend=settings.backend,
        command=command,
        diagnostic="wall-time limit exceeded" if timed_out else "",
    )


class LocalBackend:
    """Non-isolating backend for offline development and controlled tests."""

    def __init__(self, settings: ExecutionSettings):
        self.settings = settings

    def run_files(self, files: Mapping[str, str], command: list[str], *, timeout_s: Optional[float] = None) -> ExecutionResult:
        settings = ExecutionSettings(**{
            **asdict(self.settings),
            "backend": "local",
            "timeout_s": timeout_s if timeout_s is not None else self.settings.timeout_s,
        })
        with tempfile.TemporaryDirectory(prefix="facets-exec-") as directory:
            workdir = Path(directory)
            for relative, content in files.items():
                target = workdir / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
            return _communicate(command, workdir, settings, _safe_environment(workdir))

    def run_python_files(
        self, files: Mapping[str, str], entry_file: str, *, timeout_s: Optional[float] = None
    ) -> ExecutionResult:
        return self.run_files(
            files, [os.fspath(Path(os.sys.executable).resolve()), "-I", entry_file], timeout_s=timeout_s
        )


class ContainerBackend:
    """Docker/Podman backend with no network and bounded basic resources."""

    def __init__(self, settings: ExecutionSettings):
        self.settings = settings
        self.runtime = shutil.which("docker") or shutil.which("podman")

    def available(self) -> bool:
        return bool(self.runtime)

    def run_files(self, files: Mapping[str, str], inner_command: list[str], *, timeout_s: Optional[float] = None) -> ExecutionResult:
        if not self.runtime:
            return ExecutionResult(
                backend="container",
                compiled=None,
                diagnostic="docker or podman executable not found",
            )
        settings = ExecutionSettings(**{
            **asdict(self.settings),
            "backend": "container",
            "timeout_s": timeout_s if timeout_s is not None else self.settings.timeout_s,
        })
        with tempfile.TemporaryDirectory(prefix="facets-container-") as directory:
            workdir = Path(directory)
            for relative, content in files.items():
                target = workdir / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
            container_name = f"facets-{uuid.uuid4().hex[:16]}"
            command = [
                self.runtime, "run", "--rm", "--name", container_name, "--network=none", "--read-only",
                "--cap-drop=ALL", "--security-opt=no-new-privileges",
                f"--memory={settings.memory_mb}m", f"--cpus={settings.cpus}",
                f"--pids-limit={settings.pids_limit}", "--tmpfs=/tmp:rw,noexec,nosuid,size=64m",
                "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "PYTHONHASHSEED=0", "-e", "HOME=/tmp",
                "-v", f"{workdir}:/work:ro", "-w", "/work",
                settings.container_image, *inner_command,
            ]
            return _communicate(
                command, workdir, settings, dict(os.environ),
                cleanup_command=[self.runtime, "rm", "-f", container_name],
            )

    def run_python_files(
        self, files: Mapping[str, str], entry_file: str, *, timeout_s: Optional[float] = None
    ) -> ExecutionResult:
        return self.run_files(files, ["python", "-I", entry_file], timeout_s=timeout_s)


def get_backend(settings: ExecutionSettings) -> ExecutionBackend:
    if settings.backend == "local":
        return LocalBackend(settings)
    if settings.backend == "container":
        return ContainerBackend(settings)
    raise ValueError(f"unsupported execution backend: {settings.backend!r}")
