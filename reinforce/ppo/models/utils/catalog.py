from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelPreset:
    id: str
    model_config: dict[str, Any]
    default_feature_id: str | None = None
    note: str = ""


_MODEL_PRESET_TABLE: dict[str, ModelPreset] = {
    "student_m_submit_v1": ModelPreset(
        id="student_m_submit_v1",
        model_config={
            "type": "StudentMBoardAgent",
            "kwargs": {
                "board_channels": 46,
                "board_size": 10,
                "global_dim": 0,
                "width": 48,
                "num_blocks": 2,
                "global_hidden_dim": 64,
                "value_hidden_dims": [64],
                "activation": "tanh",
                "use_global_film": False,
                "use_global_policy_bias": False,
            },
        },
        default_feature_id="submit_v1",
        note="submit-v1 student baseline.",
    ),
    "teacher_p0_v1_88ch": ModelPreset(
        id="teacher_p0_v1_88ch",
        model_config={
            "type": "TeacherP0V1BoardAgent",
            "kwargs": {
                "board_channels": 88,
                "board_size": 10,
                "global_dim": 0,
                "width": 64,
                "num_blocks": 4,
                "global_hidden_dim": 64,
                "value_hidden_dims": [128, 64],
                "activation": "tanh",
                "use_global_film": False,
                "use_global_policy_bias": False,
            },
        },
        default_feature_id="teacher_p0_v1_88ch",
        note="Teacher P0 v1 preset.",
    ),
    "exp002_submit_v1_88ch": ModelPreset(
        id="exp002_submit_v1_88ch",
        model_config={
            "type": "Exp002ResNetBoardAgent",
            "kwargs": {
                "board_channels": 88,
                "board_size": 10,
                "global_dim": 0,
                "hidden_channels": 64,
                "blocks": 6,
                "aux_opp_param_head": False,
            },
        },
        default_feature_id="teacher_p0_v1_88ch",
        note="exp002 submit-v1(resnet_v1) architecture on 88ch teacher features.",
    ),
}


def list_model_presets() -> list[ModelPreset]:
    out = list(_MODEL_PRESET_TABLE.values())
    out.sort(key=lambda x: x.id)
    return out


def has_model_preset(preset_id: str) -> bool:
    return str(preset_id) in _MODEL_PRESET_TABLE


def get_model_preset(preset_id: str) -> ModelPreset:
    key = str(preset_id).strip()
    if not key:
        raise ValueError("model preset id must not be empty")
    out = _MODEL_PRESET_TABLE.get(key)
    if out is None:
        names = ", ".join(p.id for p in list_model_presets())
        raise ValueError(f"unknown model preset: {preset_id!r}; available={names}")
    return out


def get_model_config_from_preset(preset_id: str) -> dict[str, Any]:
    return copy.deepcopy(get_model_preset(preset_id).model_config)
