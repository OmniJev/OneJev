import os
import shutil
import subprocess
from pathlib import Path

import pytest

PWSH = os.environ.get("QEV_TEST_PWSH") or shutil.which("pwsh")
pytestmark = pytest.mark.skipif(not PWSH or os.name == "nt", reason="PowerShell with Unix command stubs is required")


def run_launcher(tmp_path, action, backend):
    directory = tmp_path / "docker"
    directory.mkdir()
    script = directory / "start.ps1"
    shutil.copyfile(Path(__file__).parents[1] / "docker/start.ps1", script)
    binary = tmp_path / "bin"
    binary.mkdir()
    calls = tmp_path / "calls"
    docker = binary / "docker"
    docker.write_text('''#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$QEV_TEST_CALLS"
case "$*" in
  *" up"*|*" up -d"*) [ "$QEV_TEST_FAIL_UP" = 1 ] && exit 23 ;;
  *"ps --services"*) printf 'qev-gguf-cpu\\n' ;;
  *"port qev-gguf-cpu"*) printf '0.0.0.0:8000\\n' ;;
esac
exit 0
''')
    nvidia = binary / "nvidia-smi"
    nvidia.write_text("#!/usr/bin/env bash\nprintf 'driver is unavailable\\n'\nexit 9\n")
    for executable in (docker, nvidia):
        executable.chmod(0o755)
    env = {**os.environ, "PATH": f"{binary}:/usr/bin:/bin", "QEV_TEST_CALLS": str(calls), "QEV_TEST_FAIL_UP": "1" if action == "fail" else "0"}
    env.pop("QEV_BACKEND", None)
    result = subprocess.run([str(PWSH), "-NoProfile", "-File", str(script), "-Backend", backend],
                            env=env, capture_output=True, text=True, timeout=20)
    return result, calls.read_text()


def test_failed_compose_up_preserves_exit_code(tmp_path):
    result, calls = run_launcher(tmp_path, "fail", "cpu")
    assert result.returncode == 23, result.stdout + result.stderr
    assert "API and playground" not in result.stdout
    assert " up -d" in calls


def test_failed_gpu_detection_uses_cpu_backend(tmp_path):
    result, calls = run_launcher(tmp_path, "success", "auto")
    assert result.returncode == 0, result.stderr
    assert "--profile cpu build" in calls
    assert "http://localhost:8000" in result.stdout
