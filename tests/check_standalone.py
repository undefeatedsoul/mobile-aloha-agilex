"""Verify a source copy in a fresh temporary environment, without the parent repo.

Run with the project Python and uv available. Dependencies must already be cached;
the installation is offline and the temporary environment is removed on exit.
"""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def run(command, *, cwd, env, timeout=120):
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"Command failed: {command}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def main():
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required for the standalone installation check")
    child_env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        child_env.pop(key, None)
    with tempfile.TemporaryDirectory(prefix="agilex-pi-standalone-") as temporary:
        task_dir = Path(temporary)
        source = task_dir / "source"
        source.mkdir()
        for path in ROOT.glob("*.py"):
            shutil.copy2(path, source / path.name)
        for name in ("pyproject.toml", "README.md"):
            shutil.copy2(ROOT / name, source / name)
        (source / "monitor").mkdir()
        for path in (ROOT / "monitor").glob("*.py"):
            shutil.copy2(path, source / "monitor" / path.name)
        shutil.copytree(
            ROOT / "packages" / "openpi-client", source / "packages" / "openpi-client",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info", ".venv"),
        )
        venv = task_dir / "venv"
        run([uv, "venv", "--offline", "--python", sys.executable, str(venv)], cwd=task_dir, env=child_env)
        python = str(venv / "bin" / "python")
        run(
            [uv, "pip", "install", "--offline", "--python", python,
             "-e", str(source / "packages" / "openpi-client"), "-e", str(source)],
            cwd=task_dir, env=child_env,
        )
        code = """
import importlib.abc
from pathlib import Path
import sys

class BlockParentAndHardware(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'examples', 'rospy', 'cv_bridge', 'piper_sdk', 'sensor_msgs', 'geometry_msgs', 'nav_msgs', 'std_msgs'}:
            raise AssertionError('Unexpected import: ' + fullname)

sys.meta_path.insert(0, BlockParentAndHardware())
import mobile_aloha_agilex
import mobile_aloha_agilex.env
import mobile_aloha_agilex.main
import mobile_aloha_agilex.monitor
import openpi_client
source = Path(sys.argv[1]).resolve()
assert Path(mobile_aloha_agilex.__file__).resolve() == source / '__init__.py'
assert Path(openpi_client.__file__).resolve() == source / 'packages/openpi-client/src/openpi_client/__init__.py'
"""
        run([python, "-c", code, str(source)], cwd=task_dir, env=child_env)
        commands = [
            [python, "-m", "mobile_aloha_agilex.main", "--help"],
            [str(venv / "bin" / "agilex-infer"), "--help"],
            [python, str(source / "main.py"), "--help"],
        ]
        for command in commands:
            output = run(command, cwd=task_dir, env=child_env, timeout=15)
            assert "--control-mode" in output
            assert "--async-infer" not in output
            assert "--record" not in output
        print("Standalone installation, offline imports and all three CLI entrypoints passed.")


if __name__ == "__main__":
    main()
