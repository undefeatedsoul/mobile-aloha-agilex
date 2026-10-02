"""Characterization checks against the initial snapshot; never import hardware."""

import ast
import copy
from pathlib import Path
import subprocess
from types import SimpleNamespace

import einops
import numpy as np
from openpi_client import image_tools
from openpi_client.action_chunk_broker import ActionChunkBroker
import pytest


ROOT = Path(__file__).resolve().parents[1]
BASELINE = "854dc3f"


def baseline_source(path):
    result = subprocess.run(
        ["git", "show", f"{BASELINE}:{path}"], cwd=ROOT, capture_output=True, text=True
    )
    if result.returncode:
        pytest.skip("Initial Git snapshot is required for characterization checks")
    return result.stdout


def adapter_methods(source):
    """Extract adapter methods without evaluating ROS/SDK imports or constructors."""
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AlohaRealEnvironment")
    methods = [
        node for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"get_observation", "apply_action", "reset", "close", "is_episode_complete"}
    ]
    namespace = {
        "einops": einops,
        "image_tools": image_tools,
        "override": lambda method: method,
        "DEBUG_SAVE_IMAGES": False,
        "_debug_step_counter": 0,
    }
    exec(compile(ast.Module(body=methods, type_ignores=[]), "<adapter>", "exec"), namespace)
    return type("Adapter", (), {method.name: namespace[method.name] for method in methods})


def assert_equal(expected, actual):
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_equal(expected[key], actual[key])
    else:
        assert actual == expected


@pytest.mark.parametrize("shape", [(480, 640, 3), (224, 224, 3), (257, 321, 3)])
def test_pi_observation_matches_snapshot(shape):
    rng = np.random.default_rng(42)
    observation = {
        "qpos": np.linspace(-1, 1, 14, dtype=np.float32),
        "images": {
            name: rng.integers(0, 256, shape, dtype=np.uint8)
            for name in ("cam_high", "cam_left_wrist", "cam_right_wrist")
        },
    }
    observation["images"]["cam_high_depth"] = None
    results = []
    for source in (baseline_source("env.py"), (ROOT / "env.py").read_text()):
        adapter = adapter_methods(source)()
        adapter._ts = SimpleNamespace(observation=copy.deepcopy(observation))
        adapter._render_height = adapter._render_width = 224
        adapter._model_type = "pi"
        result = adapter.get_observation()
        # Existing in-place mutation is deliberately retained in this refactor.
        assert result["images"] is adapter._ts.observation["images"]
        assert "cam_high_depth" not in result["images"]
        assert all(image.shape == (3, 224, 224) for image in result["images"].values())
        results.append(result)
    assert_equal(*results)


@pytest.mark.parametrize("mode,has_velocity", [("position", False), ("position", True), ("mit", False), ("mit", True)])
def test_execution_interface_matches_snapshot(mode, has_velocity):
    action = {
        "actions": np.arange(14, dtype=np.float32),
        "_chunk_id": 3,
        "_chunk_step": 0,
        "_in_blend": False,
        "_chunk_actions_full": np.zeros((30, 14), dtype=np.float32),
        "_chunk_skip_steps": 0,
        "_chunk_blend_window": 0,
    }
    if has_velocity:
        action["velocities"] = np.linspace(-0.1, 0.1, 14, dtype=np.float32)
        action["_chunk_velocities_full"] = np.zeros((30, 14), dtype=np.float32)
    calls = []
    for source in (baseline_source("env.py"), (ROOT / "env.py").read_text()):
        adapter = adapter_methods(source)()
        adapter._control_mode = mode
        sentinel = object()

        def step(positions, **kwargs):
            calls.append({"positions": positions, **kwargs})
            return sentinel

        adapter._env = SimpleNamespace(step=step)
        adapter.apply_action(action)
        assert adapter._ts is sentinel
    assert_equal(*calls)
    assert ("velocities" in calls[1]) == (mode == "mit" and has_velocity)


class FakePolicy:
    def __init__(self, has_velocity):
        self.calls = 0
        self.resets = 0
        self.has_velocity = has_velocity

    def infer(self, obs):
        self.calls += 1
        result = {"actions": np.arange(70, dtype=np.float32).reshape(5, 14) + self.calls * 100}
        if self.has_velocity:
            result["velocities"] = result["actions"] / 1000
        result["server_timing"] = {"infer_ms": 12.0}
        return result

    def reset(self):
        self.resets += 1


@pytest.mark.parametrize("has_velocity", [False, True])
@pytest.mark.parametrize("horizon", [3, 5])
def test_sync_chunk_sequence_matches_snapshot(has_velocity, horizon):
    namespace = {}
    exec(compile(baseline_source("packages/openpi-client/src/openpi_client/action_chunk_broker.py"), "<broker>", "exec"), namespace)
    old_policy, new_policy = FakePolicy(has_velocity), FakePolicy(has_velocity)
    old = namespace["ActionChunkBroker"](old_policy, horizon, chunk_sleep=0)
    new = ActionChunkBroker(new_policy, horizon, chunk_sleep=0)
    for index in range(horizon * 2 + 1):
        assert_equal(old.infer({"state": np.zeros(14)}), new.infer({"state": np.zeros(14)}))
        assert old_policy.calls == new_policy.calls == index // horizon + 1
    old.reset()
    new.reset()
    assert old_policy.resets == new_policy.resets == 1
    assert_equal(old.infer({}), new.infer({}))


@pytest.mark.parametrize("filename", ["real_env.py", "robot_utils.py", "mit_controller.py", "constants.py"])
def test_hardware_implementation_unchanged_except_imports_and_disabled_monitor(filename):
    class WithoutImports(ast.NodeTransformer):
        def visit_Try(self, node):
            # Normalize only the removed optional left-arm monitor import.
            if (
                len(node.body) == 1
                and isinstance(node.body[0], ast.ImportFrom)
                and len(node.body[0].names) == 1
                and node.body[0].names[0].name == "monitor_joints_left_mit"
                and node.body[0].names[0].asname == "_timeline_monitor"
                and len(node.handlers) == 1
                and isinstance(node.handlers[0].type, ast.Name)
                and node.handlers[0].type.id == "Exception"
                and ast.dump(ast.Module(body=node.handlers[0].body, type_ignores=[]))
                == ast.dump(ast.parse("_timeline_monitor = None"))
                and not node.orelse
                and not node.finalbody
            ):
                return ast.parse("_timeline_monitor = None").body[0]
            return self.generic_visit(node)

        def visit_Import(self, node):
            return None

        def visit_ImportFrom(self, node):
            return None

    before = WithoutImports().visit(ast.parse(baseline_source(filename)))
    after = WithoutImports().visit(ast.parse((ROOT / filename).read_text()))
    assert ast.dump(before) == ast.dump(after)


@pytest.mark.parametrize("filename", [
    "packages/openpi-client/src/openpi_client/action_chunk_broker.py",
    "packages/openpi-client/src/openpi_client/websocket_client_policy.py",
    "packages/openpi-client/src/openpi_client/runtime/runtime.py",
])
def test_sync_transport_and_scheduling_unchanged(filename):
    assert (ROOT / filename).read_text() == baseline_source(filename)
