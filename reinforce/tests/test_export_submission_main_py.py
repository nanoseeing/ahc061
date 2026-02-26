from __future__ import annotations

import base64
import io
import zlib

import numpy as np
import pytest
import torch

from reinforce.ppo_discrete.cli.export_submission_main_py import (
    _extract_vecnorm_for_export,
    _pack_arrays_b85,
    _state_dict_to_numpy,
    build_submission_source,
)


class TestExportSubmissionMainPy:
    def test_state_dict_to_numpy_dtype(self) -> None:
        sd = {
            "w": torch.ones((2, 3), dtype=torch.float32),
            "b": torch.ones((3,), dtype=torch.float64),
            "i": torch.ones((1,), dtype=torch.int64),
        }
        out = _state_dict_to_numpy(sd, dtype="fp16")
        assert out["w"].dtype == np.float16
        assert out["b"].dtype == np.float16
        assert out["i"].dtype == np.int64

    def test_pack_arrays_roundtrip(self) -> None:
        arrays = {
            "a": np.arange(6, dtype=np.float32).reshape(2, 3),
            "b": np.asarray([1, 2, 3], dtype=np.int32),
        }
        blob = _pack_arrays_b85(arrays, compress_level=9)
        raw = zlib.decompress(base64.b85decode(blob.encode("ascii")))
        with np.load(io.BytesIO(raw), allow_pickle=False) as z:
            np.testing.assert_allclose(z["a"], arrays["a"])
            np.testing.assert_array_equal(z["b"], arrays["b"])

    def test_build_submission_source_compiles(self) -> None:
        meta = {
            "model_type": "StudentMBoardAgent",
            "model_kwargs": {"board_channels": 7, "board_size": 10, "global_dim": 49, "num_blocks": 2},
            "obs_dim": 749,
            "action_dim": 100,
            "deterministic": True,
            "use_action_mask": True,
            "bayes_num_particles": 128,
            "bayes_seed": 0,
            "bayes_resample_ess_frac": 0.55,
            "use_vecnorm": False,
            "vecnorm": {},
        }
        blob = _pack_arrays_b85({"stem.0.weight": np.zeros((1, 1, 1, 1), dtype=np.float32)}, compress_level=1)
        src = build_submission_source(meta=meta, packed_blob_b85=blob)
        compile(src, "<generated-main.py>", "exec")

    def test_extract_vecnorm_modes(self) -> None:
        meta = {
            "vecnormalize_state": {
                "epsilon": 1e-8,
                "clip_obs": 10.0,
                "obs_rms": {
                    "mean": np.asarray([1.0, 2.0], dtype=np.float64),
                    "var": np.asarray([3.0, 4.0], dtype=np.float64),
                },
            }
        }
        use_auto, vec_meta_auto, vec_arrays_auto = _extract_vecnorm_for_export(meta, dtype="fp16", mode="auto")
        assert use_auto
        assert "__vec_obs_mean" in vec_arrays_auto
        assert vec_arrays_auto["__vec_obs_mean"].dtype == np.float16
        assert float(vec_meta_auto["epsilon"]) == pytest.approx(1e-8, abs=1e-12)

        use_off, _vec_meta_off, vec_arrays_off = _extract_vecnorm_for_export(meta, dtype="fp32", mode="off")
        assert not use_off
        assert vec_arrays_off == {}

        with pytest.raises(ValueError):
            _extract_vecnorm_for_export({}, dtype="fp32", mode="on")
