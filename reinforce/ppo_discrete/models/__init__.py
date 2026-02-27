from __future__ import annotations

from .nets.discrete_board import DiscreteBoardAgent, DiscreteBoardCNNFCAgent, DiscreteBoardMLPAgent
from .nets.exp002_resnet import Exp002ResNetBoardAgent
from .nets.student_m import StudentMBoardAgent, TeacherP0V1BoardAgent
from .utils.catalog import ModelPreset, get_model_config_from_preset, get_model_preset, has_model_preset, list_model_presets
from .utils.builder import build_agent, load_model_config_from_sources, normalize_model_config
from .utils.registry import list_registered_models, register_model, resolve_model_class


register_model("DiscreteBoardAgent", DiscreteBoardAgent)
register_model("ppo_discrete.DiscreteBoardAgent", DiscreteBoardAgent)
register_model("DiscreteBoardMLPAgent", DiscreteBoardMLPAgent)
register_model("ppo_discrete.DiscreteBoardMLPAgent", DiscreteBoardMLPAgent)
register_model("DiscreteBoardCNNFCAgent", DiscreteBoardCNNFCAgent)
register_model("ppo_discrete.DiscreteBoardCNNFCAgent", DiscreteBoardCNNFCAgent)
register_model("StudentMBoardAgent", StudentMBoardAgent)
register_model("ppo_discrete.StudentMBoardAgent", StudentMBoardAgent)
register_model("TeacherP0V1BoardAgent", TeacherP0V1BoardAgent)
register_model("ppo_discrete.TeacherP0V1BoardAgent", TeacherP0V1BoardAgent)
register_model("Exp002ResNetBoardAgent", Exp002ResNetBoardAgent)
register_model("ppo_discrete.Exp002ResNetBoardAgent", Exp002ResNetBoardAgent)


__all__ = [
    "register_model",
    "resolve_model_class",
    "list_registered_models",
    "normalize_model_config",
    "load_model_config_from_sources",
    "build_agent",
    "DiscreteBoardAgent",
    "DiscreteBoardMLPAgent",
    "DiscreteBoardCNNFCAgent",
    "Exp002ResNetBoardAgent",
    "StudentMBoardAgent",
    "TeacherP0V1BoardAgent",
    "ModelPreset",
    "list_model_presets",
    "has_model_preset",
    "get_model_preset",
    "get_model_config_from_preset",
]
