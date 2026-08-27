"""Process-level tests for the isolated worker and its bootstrap."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures"
BOOTSTRAP = REPO_ROOT / "tree_counter" / "runtime" / "worker_bootstrap.py"
TIMEOUT_SECONDS = 60


def _environment(**overrides: str) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("TREE_COUNTER_WORKER_BACKEND", None)
    for name in list(env):
        if name.startswith("TREE_COUNTER_FAKE_"):
            del env[name]
    env["PYTHONPATH"] = os.pathsep.join(
        (str(REPO_ROOT), str(FIXTURES), env.get("PYTHONPATH", ""))
    ).rstrip(os.pathsep)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.update(overrides)
    return env


class WorkerSession:
    """Drive one real worker process over its stdin/stdout pipes."""

    def __init__(self, command: list[str], env: dict[str, str]) -> None:
        # Fixed argument vector, never a shell command line.
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=str(REPO_ROOT),
        )

    def send(self, message: dict[str, object]) -> None:
        from tree_counter.core.protocol import encode_message

        assert self._process.stdin is not None
        self._process.stdin.write(encode_message(message))
        self._process.stdin.flush()

    def send_raw(self, payload: bytes) -> None:
        assert self._process.stdin is not None
        self._process.stdin.write(payload)
        self._process.stdin.flush()

    def finish(self) -> tuple[int, list[dict[str, object]], str]:
        from tree_counter.core.protocol import decode_message

        stdout, stderr = self._process.communicate(timeout=TIMEOUT_SECONDS)
        messages = [
            decode_message(line)
            for line in stdout.splitlines()
            if line.strip()
        ]
        return (
            self._process.returncode,
            messages,
            stderr.decode("utf-8", errors="replace"),
        )

    def kill(self) -> None:
        if self._process.poll() is None:
            self._process.kill()
            self._process.wait(timeout=TIMEOUT_SECONDS)


@pytest.fixture
def session(request: pytest.FixtureRequest):
    sessions: list[WorkerSession] = []

    def _start(
        *, bootstrap: bool = False, **environment: str
    ) -> WorkerSession:
        if bootstrap:
            command = [sys.executable, "-I", str(BOOTSTRAP)]
        else:
            command = [sys.executable, "-m", "tree_counter.worker"]
        started = WorkerSession(command, _environment(**environment))
        sessions.append(started)
        return started

    yield _start
    for started in sessions:
        started.kill()


def _hello() -> dict[str, object]:
    return {"type": "hello", "protocol_version": 1, "request_id": "req-hello"}


def _inspect() -> dict[str, object]:
    return {
        "type": "inspect_model",
        "protocol_version": 1,
        "request_id": "req-inspect",
        "model_path": "/models/best.onnx",
        "model_sha256": "c" * 64,
    }


def _settings(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "confidence": 0.25,
        "nms_iou": 0.70,
        "duplicate_iou": 0.50,
        "tile_size": 640,
        "overlap_percent": 20,
        "selected_class_ids": [],
        "requested_device": "cpu",
    }
    values.update(overrides)
    return values


def _start_run(workspace: Path, tile_count: int, **overrides) -> dict:
    message: dict[str, object] = {
        "type": "start_run",
        "protocol_version": 1,
        "request_id": "req-start",
        "run_id": "run-1",
        "workspace": str(workspace),
        "model_path": "/models/best.onnx",
        "model_sha256": "c" * 64,
        "tile_count": tile_count,
        "settings": _settings(),
    }
    message.update(overrides)
    return message


def _tile_message(
    tile_id: str,
    tile_path: str,
    x_offset: int = 0,
    y_offset: int = 0,
    valid: int = 640,
) -> dict[str, object]:
    return {
        "type": "tile",
        "protocol_version": 1,
        "request_id": f"req-{tile_id}",
        "run_id": "run-1",
        "tile_id": tile_id,
        "tile_path": tile_path,
        "tile_encoding": "rgb8",
        "x_offset": x_offset,
        "y_offset": y_offset,
        "valid_width": valid,
        "valid_height": valid,
        "model_width": 640,
        "model_height": 640,
    }


def _write_tile(workspace: Path, name: str) -> str:
    path = workspace / name
    path.write_bytes(b"\x89PNG\r\n\x1a\n fake tile payload")
    return name


def _types(messages: list[dict[str, object]]) -> list[str]:
    return [str(message["type"]) for message in messages]


def _detections(messages: list[dict[str, object]]) -> list[dict]:
    """Reassemble the result set the way the host does, batch by batch."""

    batches = [item for item in messages if item["type"] == "detections"]
    assert [item["batch_index"] for item in batches] == list(
        range(len(batches))
    )
    collected: list[dict] = []
    for batch in batches:
        collected.extend(batch["detections"])
    completed = [
        item for item in messages if item["type"] == "run_completed"
    ]
    if completed:
        assert completed[0]["detection_count"] == len(collected)
        assert completed[0]["batch_count"] == len(batches)
    return collected


def test_handshake_reports_a_worker_hello(session) -> None:
    worker = session(TREE_COUNTER_WORKER_BACKEND="fake")
    worker.send(_hello())
    worker.send({"type": "cancel", "protocol_version": 1, "request_id": "c"})

    code, messages, _ = worker.finish()

    assert code == 0
    assert _types(messages) == ["hello", "cancelled"]
    assert messages[0]["request_id"] == "req-hello"
    assert messages[0]["protocol_version"] == 1


def test_inspect_model_returns_class_names(session) -> None:
    worker = session(
        TREE_COUNTER_WORKER_BACKEND="fake",
        TREE_COUNTER_FAKE_CLASS_NAMES="oil_palm,shade_tree",
    )
    worker.send(_hello())
    worker.send(_inspect())
    worker.send({"type": "cancel", "protocol_version": 1, "request_id": "c"})

    code, messages, _ = worker.finish()

    assert code == 0
    info = messages[1]
    assert info["type"] == "model_info"
    assert info["class_names"] == ["oil_palm", "shade_tree"]
    assert info["backend"] == "fake"


def test_full_run_emits_progress_and_deduplicated_detections(
    session, tmp_path: Path
) -> None:
    first = _write_tile(tmp_path, "tile_a.png")
    second = _write_tile(tmp_path, "tile_b.png")
    worker = session(TREE_COUNTER_WORKER_BACKEND="fake")
    worker.send(_hello())
    worker.send(_start_run(tmp_path, 2))
    worker.send(_tile_message("r00000_c00000", first))
    worker.send(_tile_message("r00000_c00001", second, x_offset=512))
    worker.send(
        {
            "type": "finish_tiles",
            "protocol_version": 1,
            "request_id": "req-finish",
            "run_id": "run-1",
        }
    )

    code, messages, stderr = worker.finish()

    assert code == 0
    assert _types(messages) == [
        "hello",
        "run_started",
        "tile_completed",
        "progress",
        "tile_completed",
        "progress",
        "detections",
        "run_completed",
    ]
    # Each tile yields one surviving detection: one is suppressed by NMS,
    # one falls below the confidence threshold, and one is centered outside
    # the valid area.
    assert [
        message["detection_count"]
        for message in messages
        if message["type"] == "tile_completed"
    ] == [1, 1]
    progress = [
        message for message in messages if message["type"] == "progress"
    ]
    assert [item["completed_tiles"] for item in progress] == [1, 2]
    assert all(item["total_tiles"] == 2 for item in progress)

    completed = messages[-1]
    detections = _detections(messages)
    assert len(detections) == 2
    assert detections[0]["box"] == [1.0, 1.0, 11.0, 11.0]
    assert detections[1]["box"] == [513.0, 1.0, 523.0, 11.0]
    assert detections[0]["tile_ids"] == ["r00000_c00000"]
    assert completed["duration_seconds"] >= 0.0
    assert stderr.strip() == ""


def test_overlapping_tiles_merge_into_one_detection(
    session, tmp_path: Path
) -> None:
    first = _write_tile(tmp_path, "tile_a.png")
    second = _write_tile(tmp_path, "tile_b.png")
    worker = session(TREE_COUNTER_WORKER_BACKEND="fake")
    worker.send(_hello())
    worker.send(_start_run(tmp_path, 2))
    worker.send(_tile_message("r00000_c00000", first))
    # A one-pixel shift keeps the two boxes far above the duplicate IoU.
    worker.send(_tile_message("r00000_c00001", second, x_offset=1))
    worker.send(
        {
            "type": "finish_tiles",
            "protocol_version": 1,
            "request_id": "req-finish",
            "run_id": "run-1",
        }
    )

    code, messages, _ = worker.finish()

    assert code == 0
    detections = _detections(messages)
    assert len(detections) == 1
    assert detections[0]["merged_count"] == 2
    assert detections[0]["tile_ids"] == ["r00000_c00000", "r00000_c00001"]


def test_multi_class_detections_are_never_merged(
    session, tmp_path: Path
) -> None:
    tile = _write_tile(tmp_path, "tile_a.png")
    worker = session(
        TREE_COUNTER_WORKER_BACKEND="fake",
        TREE_COUNTER_FAKE_CLASS_NAMES="oil_palm,shade_tree",
    )
    worker.send(_hello())
    worker.send(_start_run(tmp_path, 1))
    worker.send(_tile_message("r00000_c00000", tile))
    worker.send(
        {
            "type": "finish_tiles",
            "protocol_version": 1,
            "request_id": "req-finish",
            "run_id": "run-1",
        }
    )

    code, messages, _ = worker.finish()

    assert code == 0
    detections = _detections(messages)
    assert sorted(item["class_id"] for item in detections) == [0, 1]


def test_selected_classes_restrict_the_result(
    session, tmp_path: Path
) -> None:
    tile = _write_tile(tmp_path, "tile_a.png")
    worker = session(
        TREE_COUNTER_WORKER_BACKEND="fake",
        TREE_COUNTER_FAKE_CLASS_NAMES="oil_palm,shade_tree",
    )
    worker.send(_hello())
    worker.send(
        _start_run(
            tmp_path, 1, settings=_settings(selected_class_ids=[1])
        )
    )
    worker.send(_tile_message("r00000_c00000", tile))
    worker.send(
        {
            "type": "finish_tiles",
            "protocol_version": 1,
            "request_id": "req-finish",
            "run_id": "run-1",
        }
    )

    code, messages, _ = worker.finish()

    assert code == 0
    detections = _detections(messages)
    assert [item["class_id"] for item in detections] == [1]


def test_cancellation_mid_run_ends_the_process_cleanly(
    session, tmp_path: Path
) -> None:
    tile = _write_tile(tmp_path, "tile_a.png")
    worker = session(TREE_COUNTER_WORKER_BACKEND="fake")
    worker.send(_hello())
    worker.send(_start_run(tmp_path, 5))
    worker.send(_tile_message("r00000_c00000", tile))
    worker.send(
        {
            "type": "cancel",
            "protocol_version": 1,
            "request_id": "req-cancel",
            "run_id": "run-1",
        }
    )

    code, messages, _ = worker.finish()

    assert code == 0
    assert _types(messages)[-1] == "cancelled"
    assert messages[-1]["run_id"] == "run-1"
    assert "run_completed" not in _types(messages)


def test_a_short_run_fails_without_partial_success(
    session, tmp_path: Path
) -> None:
    tile = _write_tile(tmp_path, "tile_a.png")
    worker = session(TREE_COUNTER_WORKER_BACKEND="fake")
    worker.send(_hello())
    worker.send(_start_run(tmp_path, 4))
    worker.send(_tile_message("r00000_c00000", tile))
    worker.send(
        {
            "type": "finish_tiles",
            "protocol_version": 1,
            "request_id": "req-finish",
            "run_id": "run-1",
        }
    )

    code, messages, _ = worker.finish()

    assert code == 1
    assert _types(messages)[-1] == "error"
    assert messages[-1]["code"] == "worker_protocol_failure"
    assert "run_completed" not in _types(messages)


def test_an_extra_tile_is_rejected_before_backend_inference(
    session, tmp_path: Path
) -> None:
    tile = _write_tile(tmp_path, "tile_a.png")
    worker = session(TREE_COUNTER_WORKER_BACKEND="fake")
    worker.send(_hello())
    worker.send(_start_run(tmp_path, 1))
    worker.send(_tile_message("r00000_c00000", tile))
    worker.send(_tile_message("r00000_c00001", tile))

    code, messages, _ = worker.finish()

    assert code == 1
    assert _types(messages).count("tile_completed") == 1
    assert messages[-1]["type"] == "error"
    assert "run_completed" not in _types(messages)


def test_malformed_host_input_fails_closed(session) -> None:
    worker = session(TREE_COUNTER_WORKER_BACKEND="fake")
    worker.send(_hello())
    worker.send_raw(b"{not json at all}\n")

    code, messages, stderr = worker.finish()

    assert code == 1
    assert _types(messages) == ["hello", "error"]
    assert messages[-1]["code"] == "worker_protocol_failure"
    assert "worker_protocol_failure" in stderr


def test_an_illegal_transition_fails_closed(session, tmp_path: Path) -> None:
    tile = _write_tile(tmp_path, "tile_a.png")
    worker = session(TREE_COUNTER_WORKER_BACKEND="fake")
    worker.send(_hello())
    worker.send(_tile_message("r00000_c00000", tile))

    code, messages, _ = worker.finish()

    assert code == 1
    assert messages[-1]["type"] == "error"
    assert messages[-1]["code"] == "worker_protocol_failure"


def test_a_traversing_tile_path_is_rejected(session, tmp_path: Path) -> None:
    worker = session(TREE_COUNTER_WORKER_BACKEND="fake")
    worker.send(_hello())
    worker.send(_start_run(tmp_path, 1))
    worker.send_raw(
        b'{"type":"tile","protocol_version":1,"request_id":"r",'
        b'"run_id":"run-1","tile_id":"t","tile_path":"../escape.png",'
        b'"x_offset":0,"y_offset":0,"valid_width":640,"valid_height":640,'
        b'"model_width":640,"model_height":640}\n'
    )

    code, messages, _ = worker.finish()

    assert code == 1
    assert messages[-1]["code"] == "worker_protocol_failure"


def test_unexpected_eof_is_a_terminal_error(session, tmp_path: Path) -> None:
    worker = session(TREE_COUNTER_WORKER_BACKEND="fake")
    worker.send(_hello())
    worker.send(_start_run(tmp_path, 1))

    code, messages, _ = worker.finish()

    assert code == 1
    assert messages[-1]["type"] == "error"
    assert messages[-1]["code"] == "worker_protocol_failure"


def test_a_backend_failure_reports_a_safe_process_error(
    session, tmp_path: Path
) -> None:
    tile = _write_tile(tmp_path, "tile_a.png")
    worker = session(
        TREE_COUNTER_WORKER_BACKEND="fake",
        TREE_COUNTER_FAKE_FAIL_ON_TILE="r00000_c00000",
    )
    worker.send(_hello())
    worker.send(_start_run(tmp_path, 1))
    worker.send(_tile_message("r00000_c00000", tile))

    code, messages, stderr = worker.finish()

    assert code == 1
    assert messages[-1]["type"] == "error"
    assert messages[-1]["code"] == "worker_process_failure"
    assert "scripted tile failure" in stderr
    assert "scripted tile failure" not in str(messages[-1])


def test_a_missing_tile_file_is_reported(session, tmp_path: Path) -> None:
    worker = session(TREE_COUNTER_WORKER_BACKEND="fake")
    worker.send(_hello())
    worker.send(_start_run(tmp_path, 1))
    worker.send(_tile_message("r00000_c00000", "absent.png"))

    code, messages, _ = worker.finish()

    assert code == 1
    assert messages[-1]["code"] == "worker_process_failure"


def test_invalid_settings_are_rejected(session, tmp_path: Path) -> None:
    worker = session(TREE_COUNTER_WORKER_BACKEND="fake")
    worker.send(_hello())
    worker.send(
        _start_run(tmp_path, 1, settings=_settings(tile_size=641))
    )

    code, messages, _ = worker.finish()

    assert code == 1
    assert messages[-1]["type"] == "error"
    assert messages[-1]["code"] == "invalid_settings"


def test_stdout_carries_protocol_lines_only(session, tmp_path: Path) -> None:
    from tree_counter.core.protocol import validate_worker_message

    tile = _write_tile(tmp_path, "tile_a.png")
    worker = session(TREE_COUNTER_WORKER_BACKEND="fake")
    worker.send(_hello())
    worker.send(_start_run(tmp_path, 1))
    worker.send(_tile_message("r00000_c00000", tile))
    worker.send(
        {
            "type": "finish_tiles",
            "protocol_version": 1,
            "request_id": "req-finish",
            "run_id": "run-1",
        }
    )

    code, messages, stderr = worker.finish()

    assert code == 0
    for message in messages:
        validate_worker_message(message)
    assert stderr == ""


def test_start_run_preserves_backend_warnings(session, tmp_path: Path) -> None:
    worker = session(
        TREE_COUNTER_WORKER_BACKEND="fake",
        TREE_COUNTER_FAKE_WARNINGS="cpu_fallback,provider_fallback",
    )
    worker.send(_hello())
    worker.send(_start_run(tmp_path, 0))
    worker.send(
        {
            "type": "finish_tiles",
            "protocol_version": 1,
            "request_id": "req-finish",
            "run_id": "run-1",
        }
    )

    code, messages, _ = worker.finish()

    assert code == 0
    started = next(item for item in messages if item["type"] == "run_started")
    assert started["warnings"] == ["cpu_fallback", "provider_fallback"]


def test_without_a_backend_the_run_fails_with_a_runtime_error(
    session, tmp_path: Path
) -> None:
    worker = session()
    worker.send(_hello())
    worker.send(_start_run(tmp_path, 1))

    code, messages, _ = worker.finish()

    assert code == 1
    assert _types(messages) == ["hello", "error"]
    assert messages[-1]["code"] == "missing_runtime"


def test_an_unknown_backend_name_fails_closed(
    session, tmp_path: Path
) -> None:
    worker = session(TREE_COUNTER_WORKER_BACKEND="curl|sh")
    worker.send(_hello())
    worker.send(_start_run(tmp_path, 1))

    code, messages, _ = worker.finish()

    assert code == 1
    assert messages[-1]["code"] == "missing_runtime"


def test_the_worker_rejects_command_line_arguments() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "tree_counter.worker", "--evil"],
        capture_output=True,
        env=_environment(),
        cwd=str(REPO_ROOT),
        timeout=TIMEOUT_SECONDS,
    )

    assert completed.returncode == 2
    assert completed.stdout == b""


class TestBootstrap:
    """The fixed bootstrap script is the only supported launch path."""

    def test_isolated_launch_completes_a_handshake(self, session) -> None:
        worker = session(bootstrap=True, TREE_COUNTER_WORKER_BACKEND="fake")
        worker.send(_hello())
        worker.send(
            {"type": "cancel", "protocol_version": 1, "request_id": "c"}
        )

        code, messages, _ = worker.finish()

        assert code == 0
        assert _types(messages) == ["hello", "cancelled"]

    def test_isolated_launch_ignores_pythonpath_backends(
        self, session, tmp_path: Path
    ) -> None:
        # ``-I`` drops PYTHONPATH, so the fixture backend is unreachable and
        # the worker must fail closed rather than run something unexpected.
        worker = session(bootstrap=True, TREE_COUNTER_WORKER_BACKEND="fake")
        worker.send(_hello())
        worker.send(_start_run(tmp_path, 1))

        code, messages, _ = worker.finish()

        assert code == 1
        assert messages[-1]["code"] == "missing_runtime"

    def test_it_never_writes_bytecode_into_the_plugin(
        self, session
    ) -> None:
        # Isolated mode ignores PYTHONDONTWRITEBYTECODE, so the bootstrap
        # has to disable bytecode writing itself. Compare the exact set of
        # compiled files before and after: other tooling may already have
        # left caches here, but this run must add none.
        package = REPO_ROOT / "tree_counter"
        before = {path for path in package.rglob("*.py[co]")}

        worker = session(bootstrap=True, TREE_COUNTER_WORKER_BACKEND="fake")
        worker.send(_hello())
        worker.send(
            {"type": "cancel", "protocol_version": 1, "request_id": "c"}
        )
        code, _, _ = worker.finish()

        assert code == 0
        assert {path for path in package.rglob("*.py[co]")} == before

    def test_it_rejects_command_line_arguments(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-I", str(BOOTSTRAP), "--evil"],
            capture_output=True,
            cwd=str(REPO_ROOT),
            timeout=TIMEOUT_SECONDS,
        )

        assert completed.returncode == 2
        assert completed.stdout == b""

    def test_it_resolves_the_installed_plugin_parent(self) -> None:
        from tree_counter.runtime.worker_bootstrap import (
            resolve_plugin_parent,
        )

        assert resolve_plugin_parent(BOOTSTRAP) == REPO_ROOT

    def test_it_rejects_a_relocated_script(self, tmp_path: Path) -> None:
        from tree_counter.runtime.worker_bootstrap import (
            BootstrapError,
            resolve_plugin_parent,
        )

        stray = tmp_path / "elsewhere" / "runtime"
        stray.mkdir(parents=True)
        script = stray / "worker_bootstrap.py"
        script.write_text("", encoding="utf-8")

        with pytest.raises(BootstrapError):
            resolve_plugin_parent(script)

    def test_it_rejects_an_incomplete_package(self, tmp_path: Path) -> None:
        from tree_counter.runtime.worker_bootstrap import (
            BootstrapError,
            resolve_plugin_parent,
        )

        runtime = tmp_path / "tree_counter" / "runtime"
        runtime.mkdir(parents=True)
        script = runtime / "worker_bootstrap.py"
        script.write_text("", encoding="utf-8")

        with pytest.raises(BootstrapError):
            resolve_plugin_parent(script)

    def test_the_search_path_keeps_only_the_verified_parent(self) -> None:
        from tree_counter.runtime.worker_bootstrap import build_sys_path

        result = build_sys_path(
            REPO_ROOT, ["", ".", str(REPO_ROOT), "/usr/lib/python3.12"]
        )

        assert result[0] == str(REPO_ROOT)
        assert result.count(str(REPO_ROOT)) == 1
        assert "" not in result
        assert "." not in result


def test_a_result_set_larger_than_one_message_is_streamed(
    session, tmp_path: Path
) -> None:
    from tree_counter.core.protocol import (
        MAX_MESSAGE_BYTES,
        ProtocolError,
        encode_message,
    )

    # A single JSONL line is capped, so a large-area count must arrive in
    # batches instead of failing with a protocol error. The load is spread
    # over tiles the way a real run spreads it, rather than piling an
    # unrealistic number of detections into one tile.
    per_tile = 500
    tiles = 20
    total = per_tile * tiles
    tile = _write_tile(tmp_path, "tile_a.png")
    worker = session(
        TREE_COUNTER_WORKER_BACKEND="fake",
        TREE_COUNTER_FAKE_BULK_DETECTIONS=str(per_tile),
    )
    worker.send(_hello())
    worker.send(_start_run(tmp_path, tiles))
    for index in range(tiles):
        worker.send(
            _tile_message(
                f"r00000_c{index:05d}", tile, x_offset=index * 1000
            )
        )
    worker.send(
        {
            "type": "finish_tiles",
            "protocol_version": 1,
            "request_id": "req-finish",
            "run_id": "run-1",
        }
    )

    code, messages, stderr = worker.finish()

    assert code == 0
    assert stderr == ""
    detections = _detections(messages)
    assert len(detections) == total

    # The whole set genuinely does not fit in one line, so batching is
    # what makes this run possible rather than a stylistic choice.
    with pytest.raises(ProtocolError):
        encode_message(
            {
                "type": "run_completed",
                "protocol_version": 1,
                "request_id": "req-finish",
                "run_id": "run-1",
                "detections": detections,
                "duration_seconds": 1.0,
            }
        )
    batches = [item for item in messages if item["type"] == "detections"]
    assert len(batches) > 1
    for batch in batches:
        assert batch["detections"]
        assert len(encode_message(batch)) <= MAX_MESSAGE_BYTES
    assert _types(messages)[-1] == "run_completed"
