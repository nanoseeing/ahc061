"""`test_checkpoint_env_api_and_models` のテストモジュール。"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from reinforce.ppo.pipeline.model_checkpoint_service import load_agent_checkpoint, save_agent_checkpoint
from reinforce.ppo.models.nets.discrete_board import DiscreteBoardAgent
from reinforce.ppo.models import (
    Exp002ResNetBoardAgent,
    StudentMBoardAgent,
    TeacherP0V1BoardAgent,
    build_agent,
    get_model_config_from_preset,
)
from reinforce.ppo.train.ppo_service import _load_initial_weights


class TestPPOCoreModelAndCheckpoint:
    """`TestPPOCoreModelAndCheckpoint` のテストケース。"""
    def test_discrete_agent_deterministic_action_with_mask(self) -> None:
        """`discrete_agent_deterministic_action_with_mask` の振る舞いを検証する。"""
        agent = DiscreteBoardAgent(
            obs_shape=(4,),
            action_dim=3,
            model_type="mlp",
            mlp_hidden_dims=(8,),
        )
        obs = torch.zeros((1, 4), dtype=torch.float32)
        mask = torch.tensor([[False, True, False]], dtype=torch.bool)
        act = agent.act(obs, action_mask=mask, deterministic=True)
        assert int(act.item()) == 1

    def test_student_m_deterministic_action_with_mask(self) -> None:
        """`student_m_deterministic_action_with_mask` の振る舞いを検証する。"""
        agent = StudentMBoardAgent(
            obs_shape=(4,),
            action_dim=4,
            board_channels=1,
            board_size=2,
            global_dim=0,
            width=8,
            num_blocks=0,
        )
        obs = torch.zeros((1, 4), dtype=torch.float32)
        mask = torch.tensor([[False, False, True, False]], dtype=torch.bool)
        act = agent.act(obs, action_mask=mask, deterministic=True)
        assert int(act.item()) == 2

    def test_build_agent_student_m(self) -> None:
        """`build_agent_student_m` の振る舞いを検証する。"""
        agent, resolved = build_agent(
            obs_shape=(4,),
            action_dim=4,
            model_config={
                "type": "StudentMBoardAgent",
                "kwargs": {
                    "board_channels": 1,
                    "board_size": 2,
                    "global_dim": 0,
                    "width": 8,
                    "num_blocks": 0,
                },
            },
        )
        assert agent.__class__.__name__ == "StudentMBoardAgent"
        assert resolved["type"] == "StudentMBoardAgent"

    def test_build_agent_teacher_p0_v1(self) -> None:
        """`build_agent_teacher_p0_v1` の振る舞いを検証する。"""
        agent, resolved = build_agent(
            obs_shape=(88 * 10 * 10,),
            action_dim=100,
            model_config={
                "type": "TeacherP0V1BoardAgent",
                "kwargs": {},
            },
        )
        assert isinstance(agent, TeacherP0V1BoardAgent)
        assert resolved["type"] == "TeacherP0V1BoardAgent"
        obs = torch.zeros((2, 88 * 10 * 10), dtype=torch.float32)
        logits = agent.get_logits(obs)
        value = agent.get_value(obs)
        assert tuple(logits.shape) == (2, 100)
        assert tuple(value.shape) == (2, 1)

    def test_build_agent_exp002_submit_v1_88ch(self) -> None:
        """`build_agent_exp002_submit_v1_88ch` の振る舞いを検証する。"""
        model_cfg = get_model_config_from_preset("exp002_submit_v1_88ch")
        agent, resolved = build_agent(
            obs_shape=(88 * 10 * 10,),
            action_dim=100,
            model_config=model_cfg,
        )
        assert isinstance(agent, Exp002ResNetBoardAgent)
        assert resolved["type"] == "Exp002ResNetBoardAgent"
        obs = torch.zeros((2, 88 * 10 * 10), dtype=torch.float32)
        logits = agent.get_logits(obs)
        value = agent.get_value(obs)
        assert tuple(logits.shape) == (2, 100)
        assert tuple(value.shape) == (2, 1)

    def test_build_agent_exp002_submit_v1_88ch_with_submit_v1_obs(self) -> None:
        """`build_agent_exp002_submit_v1_88ch_with_submit_v1_obs` の振る舞いを検証する。"""
        model_cfg = get_model_config_from_preset("exp002_submit_v1_88ch")
        agent, resolved = build_agent(
            obs_shape=(46 * 10 * 10,),
            action_dim=100,
            model_config=model_cfg,
        )
        assert isinstance(agent, Exp002ResNetBoardAgent)
        assert int(agent.board_channels) == 46
        assert resolved["type"] == "Exp002ResNetBoardAgent"
        assert int(resolved["kwargs"]["board_channels"]) == 46
        obs = torch.zeros((2, 46 * 10 * 10), dtype=torch.float32)
        logits = agent.get_logits(obs)
        value = agent.get_value(obs)
        assert tuple(logits.shape) == (2, 100)
        assert tuple(value.shape) == (2, 1)

    def test_checkpoint_roundtrip(self) -> None:
        """`checkpoint_roundtrip` の振る舞いを検証する。"""
        agent, _resolved = build_agent(
            obs_shape=(4,),
            action_dim=3,
            model_config={
                "type": "DiscreteBoardAgent",
                "kwargs": {
                    "model_type": "mlp",
                    "mlp_hidden_dims": [8],
                },
            },
        )
        optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
        with torch.no_grad():
            first_param = next(agent.parameters())
            first_param.fill_(0.1234)

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ckpt.pt"
            save_agent_checkpoint(
                p,
                agent,
                optimizer=optimizer,
                meta={"tag": "unit-test"},
            )
            loaded_agent, meta = load_agent_checkpoint(p, device="cpu")
            assert meta.get("tag") == "unit-test"

            payload = torch.load(p, map_location="cpu", weights_only=False)
            assert "optimizer_state_dict" in payload
            assert "model_config" in payload

            p1 = next(agent.parameters()).detach().cpu().numpy()
            p2 = next(loaded_agent.parameters()).detach().cpu().numpy()
            np.testing.assert_allclose(p1, p2, rtol=1e-7, atol=1e-7)

    def test_load_initial_weights_accepts_compile_wrapped_module(self) -> None:
        """`load_initial_weights_accepts_compile_wrapped_module` の振る舞いを検証する。"""
        class _CompileLikeWrapper(torch.nn.Module):
            """`_CompileLikeWrapper` を表すクラス。"""
            def __init__(self, mod: torch.nn.Module) -> None:
                """インスタンスを初期化する。

                Args:
                    mod (torch.nn.Module): mod の値。
                """
                super().__init__()
                self._orig_mod = mod

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """`forward` を実行する。

                Args:
                    x (torch.Tensor): 入力テンソル。

                Returns:
                    torch.Tensor: 計算結果。
                """
                return self._orig_mod(x)

        base = torch.nn.Linear(4, 3)
        with torch.no_grad():
            base.weight.zero_()
            base.bias.zero_()

        donor = torch.nn.Linear(4, 3)
        with torch.no_grad():
            donor.weight.fill_(0.25)
            donor.bias.fill_(0.75)

        wrapper = _CompileLikeWrapper(base)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "init.pt"
            torch.save({"model_state_dict": donor.state_dict(), "meta": {"kind": "unit-test"}}, p)
            meta = _load_initial_weights(p, wrapper, torch.device("cpu"))

        assert meta.get("kind") == "unit-test"
        np.testing.assert_allclose(base.weight.detach().cpu().numpy(), donor.weight.detach().cpu().numpy(), rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(base.bias.detach().cpu().numpy(), donor.bias.detach().cpu().numpy(), rtol=1e-7, atol=1e-7)
