import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.name == "nt" or shutil.which("bash") is None, reason="Unix Bash launcher checks")


def test_explicit_render_groups_do_not_abort_launcher(tmp_path):
    directory = tmp_path / "docker"
    directory.mkdir()
    source = Path(__file__).parents[1] / "docker/start.sh"
    script = directory / "start.sh"
    shutil.copyfile(source, script)
    binary = tmp_path / "bin"
    binary.mkdir()
    calls = tmp_path / "calls"
    docker = binary / "docker"
    docker.write_text('''#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$QEV_TEST_CALLS"
case "$*" in
  *"ps --services"*) printf 'qev-gguf-cpu\\n' ;;
  *"port qev-gguf-cpu"*) printf '0.0.0.0:8000\\n' ;;
esac
''')
    getent = binary / "getent"
    getent.write_text('''#!/usr/bin/env bash
case "$2" in
  render) printf 'render:x:109:\\n' ;;
  video) printf 'video:x:44:\\n' ;;
esac
''')
    for executable in (docker, getent):
        executable.chmod(0o755)
    env = {**os.environ, "PATH": f"{binary}:/usr/bin:/bin", "QEV_TEST_CALLS": str(calls),
           "AMD_RENDER_GID": "321", "DRI_RENDER_GID": "654", "AMD_VIDEO_GID": "987"}
    result = subprocess.run(["bash", str(script), "cpu"], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "--profile cpu build" in calls.read_text()
    assert "--profile cpu up -d" in calls.read_text()
    assert "http://localhost:8000" in result.stdout
