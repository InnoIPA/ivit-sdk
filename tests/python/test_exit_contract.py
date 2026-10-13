"""The interpreter must exit cleanly while detached inference work is still running."""

import subprocess
import sys

import pytest

from test_model_contract import model_file, sdk  # noqa: F401  (sdk is a fixture)

SCRIPT = """
import sys, time, numpy as np, ivit
model = ivit.load(sys.argv[1], device="cpu", backend="openvino", task="classification",
                  decoder="logits", use_cache=False)
image = np.zeros((2, 4, 3), np.uint8)
model.on("infer_start", lambda ctx: time.sleep(0.2))  # keep the workers busy past interpreter exit
for _ in range(16):
    model.submit_inference(image, lambda results: None)
# exit right away: worker threads are still running and drop their callbacks during shutdown
"""


def test_exit_after_submit_inference_does_not_crash(sdk, tmp_path):
    path = model_file(tmp_path / "tiny.onnx", [1, 3, 2, 4], [[0.1, 0.9]])
    # The crash was a race (about half of the runs), so one run proves little.
    for attempt in range(12):
        done = subprocess.run([sys.executable, "-c", SCRIPT, path], capture_output=True, text=True)
        assert done.returncode == 0, f"run {attempt}: exit code {done.returncode}\n{done.stderr[-600:]}"


def test_wait_for_pending_work_returns_after_every_callback_ran(sdk, tmp_path):
    import numpy as np

    path = model_file(tmp_path / "tiny.onnx", [1, 3, 2, 4], [[0.1, 0.9]])
    model = sdk.load(path, device="cpu", backend="openvino", task="classification",
                     decoder="logits", use_cache=False)
    image = np.zeros((2, 4, 3), np.uint8)
    seen = []
    for _ in range(20):
        model.submit_inference(image, lambda results: seen.append(1))
    assert sdk.wait_for_pending_work(30.0) is True
    assert len(seen) == 20
    assert sdk.wait_for_pending_work(0.0) is True  # nothing left
