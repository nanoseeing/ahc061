from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from reinforce.ppo_discrete.pipeline.model_checkpoint_service import load_agent_checkpoint, save_agent_checkpoint
from reinforce.ppo_discrete.models.nets.discrete_board import DiscreteBoardAgent
from reinforce.ppo_discrete.models import StudentMBoardAgent, TeacherP0V1BoardAgent, build_agent


class TestPPOCoreModelAndCheckpoint:
    def test_discrete_agent_deterministic_action_with_mask(self) -> None:
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

    def test_checkpoint_roundtrip(self) -> None:
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
