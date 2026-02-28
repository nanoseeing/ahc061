"""Discrete-board PPO framework scaffold.

This package defines reusable modules for building a contest-specific PPO trainer.
It intentionally mirrors CleanRL's PPO decomposition while separating environment-
specific logic from generic algorithm code.
"""

from .ppo.config import PPOConfig
from .ppo.rollout_buffer import RolloutBuffer
from .ppo.trainer import PPOTrainer
from .models.nets.exp002_resnet import Exp002ResNetBoardAgent
from .models.nets.student_m import StudentMBoardAgent, TeacherP0V1BoardAgent
from .models.nets.discrete_board import DiscreteBoardAgent
from .models import (
    build_agent,
    get_model_config_from_preset,
    get_model_preset,
    has_model_preset,
    list_model_presets,
    list_registered_models,
    normalize_model_config,
)
from .utils.experiment import RunLayout, create_run_layout, make_run_name, update_manifest
from .utils.tracking import MetricTracker

__all__ = [
    "PPOConfig",
    "RunLayout",
    "make_run_name",
    "create_run_layout",
    "update_manifest",
    "DiscreteBoardAgent",
    "Exp002ResNetBoardAgent",
    "StudentMBoardAgent",
    "TeacherP0V1BoardAgent",
    "build_agent",
    "normalize_model_config",
    "list_registered_models",
    "list_model_presets",
    "has_model_preset",
    "get_model_preset",
    "get_model_config_from_preset",
    "RolloutBuffer",
    "MetricTracker",
    "PPOTrainer",
]
