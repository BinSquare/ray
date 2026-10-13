"""SandboxRuntime-layer tests with fake manager/backend (no runsc needed).

These pin that ``create()`` arguments actually reach the backend's
``SandboxConfig`` — the layer where a declared-but-unforwarded field once
shipped behind a green suite.
"""

import sys
import time

import pytest

from ray.experimental.sandbox import runtime as runtime_mod
from ray.experimental.sandbox.backend.base import (
    BaseSandboxBackend,
    ExecResult,
    SandboxStatus,
)
from ray.experimental.sandbox.backend.gvisor import GVisorSandboxBackend
from ray.experimental.sandbox.runtime import SandboxRuntime


class _FakeImageManager:
    def pull_image(self, image, timeout_seconds=120.0):
        return f"/tmp/fake-images/{image}"


class _FakeBackend:
    def __init__(self):
        self.configs = []
        self.deleted = []

    def create_sandbox(self, config):
        self.configs.append(config)
        return "ray-sandbox-fake0001"

    def delete_sandbox(self, sandbox_id):
        self.deleted.append(sandbox_id)

    def exec_command(self, sandbox_id, command, timeout=None, cwd=None, env=None):
        return ExecResult(exit_code=0, stdout="", stderr="", duration_seconds=0.0)

    def write_file(self, sandbox_id, path, content):
        pass

    def read_file(self, sandbox_id, path):
        return b""

    def get_status(self, sandbox_id):
        return SandboxStatus.RUNNING


def _fake_runtime():
    runtime = SandboxRuntime()
    runtime._image_manager = _FakeImageManager()
    runtime._backend = _FakeBackend()
    return runtime


def test_create_forwards_config_fields_to_backend():
    runtime = _fake_runtime()
    runtime.create(
        image="fake:latest",
        cpu=2.0,
        memory="1Gi",
        env={"A": "1"},
        workdir="/app",
        network="host",
        dns=["10.0.0.2"],
        capabilities=["CAP_CHOWN"],
        rootless=True,
        readonly=False,
        shell="/bin/bash",
    )
    (config,) = runtime._backend.configs
    assert config.image == "fake:latest"
    assert config.cpu == 2.0
    assert config.env == {"A": "1"}
    assert config.workdir == "/app"
    assert config.network == "host"
    assert config.dns == ["10.0.0.2"]
    assert config.capabilities == ["CAP_CHOWN"]
    assert config.rootless is True
    assert config.readonly is False
    assert config.shell == "/bin/bash"


def test_ttl_is_enforced_by_the_runtime():
    runtime = _fake_runtime()
    instance_id = runtime.create(image="fake:latest", ttl_seconds=1)
    assert runtime._backend.deleted == []
    deadline = time.monotonic() + 5
    while not runtime._backend.deleted and time.monotonic() < deadline:
        time.sleep(0.05)
    assert runtime._backend.deleted == [instance_id]
    # The timer is gone after firing (delete popped it).
    assert instance_id not in runtime._ttl_timers


def test_delete_cancels_the_ttl_timer():
    runtime = _fake_runtime()
    instance_id = runtime.create(image="fake:latest", ttl_seconds=3600)
    timer = runtime._ttl_timers[instance_id]
    runtime.delete(instance_id)
    assert instance_id not in runtime._ttl_timers
    # cancel() sets the finished event; the thread itself may take a beat to
    # exit, so assert the deterministic signal rather than thread liveness.
    assert timer.finished.is_set()
    assert runtime._backend.deleted == [instance_id]


def test_no_ttl_by_default():
    runtime = _fake_runtime()
    instance_id = runtime.create(image="fake:latest")
    assert instance_id not in runtime._ttl_timers
    (config,) = runtime._backend.configs
    assert config.ttl_seconds is None
    assert config.shell == "/bin/bash"


class _PluginBackend(_FakeBackend, BaseSandboxBackend):
    """A backend as a third-party package would ship it."""

    def __init__(self, image_manager=None):
        _FakeBackend.__init__(self)
        BaseSandboxBackend.__init__(self, image_manager or _FakeImageManager())


class _EntryPoint:
    def __init__(self, name, target):
        self.name, self._target = name, target

    def load(self):
        return self._target


def _register(monkeypatch, **backends):
    import importlib.metadata

    def entry_points(group=None):
        assert group == runtime_mod.BACKEND_ENTRY_POINT_GROUP
        return [_EntryPoint(name, cls) for name, cls in backends.items()]

    monkeypatch.setattr(importlib.metadata, "entry_points", entry_points)


def test_gvisor_is_the_default_backend(monkeypatch):
    monkeypatch.delenv(runtime_mod.BACKEND_ENV_VAR, raising=False)
    runtime = SandboxRuntime()
    assert isinstance(runtime.backend, GVisorSandboxBackend)
    assert runtime.image_manager is runtime.backend.image_manager


def test_explicit_backend_and_its_image_manager_are_used():
    backend = _PluginBackend()
    runtime = SandboxRuntime(backend=backend)
    assert runtime.backend is backend
    assert runtime.image_manager is backend.image_manager
    assert runtime.create(image="python:3.12-slim") == "ray-sandbox-fake0001"
    assert backend.configs[0].image == "python:3.12-slim"


def test_env_var_selects_a_registered_backend(monkeypatch):
    _register(monkeypatch, plugin=_PluginBackend)
    monkeypatch.setenv(runtime_mod.BACKEND_ENV_VAR, "plugin")
    assert isinstance(SandboxRuntime().backend, _PluginBackend)


def test_unknown_backend_lists_the_installed_ones(monkeypatch):
    _register(monkeypatch, plugin=_PluginBackend)
    monkeypatch.setenv(runtime_mod.BACKEND_ENV_VAR, "nope")
    with pytest.raises(ValueError, match=r"'nope'.*\['gvisor', 'plugin'\]"):
        SandboxRuntime()


def test_registered_object_must_be_a_backend(monkeypatch):
    _register(monkeypatch, broken=_FakeBackend)
    monkeypatch.setenv(runtime_mod.BACKEND_ENV_VAR, "broken")
    with pytest.raises(TypeError, match="not a BaseSandboxBackend"):
        SandboxRuntime()


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
