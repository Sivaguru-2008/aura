"""Regression test for AURA server vision engine startup path and diagnostics."""
import os
import pytest
pytest.importorskip("torch")
from unittest import mock
from aura.services.vision.engine import VisionEngine


def test_vision_engine_load_success():
    # Verify that loading succeeds without raising any exceptions
    engine = VisionEngine.load()
    assert engine is not None


def test_vision_engine_load_failure_diagnostics():
    # Force loading failure by mocking VisionModel to raise an error
    with mock.patch("aura.ml.vision_cxr.inference.VisionModel", side_effect=ValueError("Simulated DLL loading failure")):
        with mock.patch.dict(os.environ, {"AURA_ALLOW_FALLBACK_VISION": "0"}):
            with pytest.raises(RuntimeError) as exc_info:
                VisionEngine.load()

            err_msg = str(exc_info.value)

            # Assert all the required diagnostic report sections are present in the error report
            assert "VisionEngine startup failed" in err_msg
            assert "Checkpoint:" in err_msg
            assert "Python:" in err_msg
            assert "Torch:" in err_msg
            assert "Torch CUDA available:" in err_msg
            assert "Checkpoint load:" in err_msg
            assert "Model construction:" in err_msg
            assert "State dict load:" in err_msg
            assert "TorchVision import:" in err_msg
            assert "Native dependency:" in err_msg
            assert "Original exception:" in err_msg
            assert "Simulated DLL loading failure" in err_msg
