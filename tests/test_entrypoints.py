"""Offline checks for installation, CLI assembly and the hardware boundary."""

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from openpi_client import websocket_client_policy
import pytest

import mobile_aloha_agilex
from mobile_aloha_agilex import env, main


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("entrypoint", ["module", "command", "script"])
def test_help_from_another_directory(entrypoint, tmp_path):
    commands = {
        "module": [sys.executable, "-m", "mobile_aloha_agilex.main", "--help"],
        "command": [str(Path(sys.executable).parent / "agilex-infer"), "--help"],
        "script": [sys.executable, str(ROOT / "main.py"), "--help"],
    }
    child_env = dict(os.environ)
    child_env.pop("PYTHONPATH", None)
    result = subprocess.run(commands[entrypoint], cwd=tmp_path, env=child_env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert "--control-mode" in result.stdout
    for removed in ("--async-infer", "--trigger-step", "--blend-window", "--record", "--record-rate", "--record-file"):
        assert removed not in result.stdout


def test_offline_imports_do_not_load_hardware(tmp_path):
    code = """
import importlib.abc
import sys

class BlockHardware(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'rospy', 'cv_bridge', 'piper_sdk', 'sensor_msgs', 'geometry_msgs', 'nav_msgs', 'std_msgs'}:
            raise AssertionError('Hardware imported: ' + fullname)

sys.meta_path.insert(0, BlockHardware())
import mobile_aloha_agilex
import mobile_aloha_agilex.main
import mobile_aloha_agilex.env
import mobile_aloha_agilex.monitor
assert 'mobile_aloha_agilex.real_env' not in sys.modules
"""
    child_env = dict(os.environ)
    child_env.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=child_env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("mode", ["position", "mit"])
def test_adapter_constructs_existing_backend(monkeypatch, mode):
    calls = []
    backend = SimpleNamespace(reset=lambda: "timestep", close=lambda: calls.append("close"))

    def make_real_env(**kwargs):
        calls.append(kwargs)
        return backend

    module = SimpleNamespace(make_real_env=make_real_env)
    monkeypatch.setitem(sys.modules, "mobile_aloha_agilex.real_env", module)
    monkeypatch.setattr(mobile_aloha_agilex, "real_env", module, raising=False)
    mit_config = {"kp": 10.0, "kd": 0.8} if mode == "mit" else None
    adapter = env.AlohaRealEnvironment(reset_position=[0.0] * 6, control_mode=mode, mit_config=mit_config)
    assert calls == [{
        "init_node": True,
        "reset_position": [0.0] * 6,
        "control_mode": mode,
        "mit_config": mit_config,
    }]
    adapter.reset()
    assert adapter._ts == "timestep"
    assert adapter.is_episode_complete() is False
    adapter.close()
    assert calls[-1] == "close"


@pytest.mark.parametrize("mode", ["position", "mit"])
@pytest.mark.parametrize("fail", [False, True])
def test_main_runs_sync_policy_and_closes_environment(monkeypatch, mode, fail):
    created = []
    policies = []

    class Policy:
        def __init__(self, **kwargs):
            self.connection = kwargs
            self.calls = 0
            policies.append(self)

        def get_server_metadata(self):
            return {"reset_pose": [0.0] * 6}

        def infer(self, observation):
            self.calls += 1
            if fail:
                raise RuntimeError("test inference failure")
            return {"actions": np.full((2, 14), self.calls, dtype=np.float32)}

        def reset(self):
            pass

    class Environment:
        def __init__(self, **kwargs):
            self.config = kwargs
            self.actions = []
            self.resets = 0
            self.closed = False
            created.append(self)

        def reset(self):
            self.resets += 1

        def get_observation(self):
            return {"state": np.zeros(14), "images": {}}

        def apply_action(self, action):
            self.actions.append(action)

        def is_episode_complete(self):
            return False

        def close(self):
            self.closed = True

    monkeypatch.setattr(websocket_client_policy, "WebsocketClientPolicy", Policy)
    monkeypatch.setattr(env, "AlohaRealEnvironment", Environment)
    args = main.Args(control_mode=mode, action_horizon=2, max_episode_steps=1)
    if fail:
        with pytest.raises(RuntimeError, match="test inference failure"):
            main.main(args)
    else:
        main.main(args)
        # Preserve the original Runtime's N+1 step behavior and final reset.
        assert len(created[0].actions) == 2
        assert created[0].resets == 2
        assert [action["_chunk_step"] for action in created[0].actions] == [0, 1]
    assert created[0].closed
    assert policies[0].calls == 1
    assert policies[0].connection == {"host": "10.19.134.58", "port": 8000}
    assert created[0].config["reset_position"] == [0.0] * 6
    assert "model_type" not in created[0].config
    assert "record_config" not in created[0].config
    expected_mit = {"can_port_left": "can_left", "can_port_right": "can_right", "kp": 10.0, "kd": 0.8}
    assert created[0].config["mit_config"] == (expected_mit if mode == "mit" else None)


@pytest.mark.parametrize("arguments", [["--record"], ["--record-rate", "50"], ["--record-file", "pi_run"]])
def test_removed_record_flags_fail_before_hardware_start(arguments, tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "mobile_aloha_agilex.main", *arguments],
        cwd=tmp_path, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode != 0
    assert "Unrecognized options" in result.stdout + result.stderr
