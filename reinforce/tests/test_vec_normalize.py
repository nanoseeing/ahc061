from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from reinforce.ppo_discrete.env.vec_normalize import RunningMeanStd, VecNormalize, normalize_obs_with_state


class _ToyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self._t = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._t = 0
        return np.asarray([1.0, -1.0], dtype=np.float32), {}

    def step(self, action):
        self._t += 1
        obs = np.asarray([1.0, -1.0], dtype=np.float32)
        reward = 2.0
        terminated = False
        truncated = self._t >= 1
        info = {}
        return obs, reward, terminated, truncated, info


class TestVecNormalize:
    def test_running_mean_std_update_and_restore(self) -> None:
        rms = RunningMeanStd(shape=(2,))
        rms.update(np.asarray([[1.0, 3.0], [3.0, 5.0]], dtype=np.float64))
        state = rms.state_dict()

        rms2 = RunningMeanStd(shape=(2,))
        rms2.load_state_dict(state)
        np.testing.assert_allclose(rms2.mean, rms.mean)
        np.testing.assert_allclose(rms2.var, rms.var)
        assert rms2.count == pytest.approx(rms.count, abs=1e-12)

    def test_vecnormalize_step_and_returns_reset_on_done(self) -> None:
        envs = gym.vector.SyncVectorEnv([lambda: _ToyEnv(), lambda: _ToyEnv()])
        vec = VecNormalize(envs, norm_obs=True, norm_reward=True, gamma=0.99, training=True)
        try:
            obs, _info = vec.reset(seed=0)
            assert obs.shape == (2, 2)
            assert np.all(np.isfinite(obs))

            _next_obs, reward, term, trunc, _infos = vec.step(np.asarray([0, 1], dtype=np.int64))
            assert reward.shape == (2,)
            assert np.all(np.isfinite(reward))
            assert np.all(np.logical_or(term, trunc))
            assert np.allclose(vec.returns, 0.0)
        finally:
            vec.close()

    def test_normalize_obs_with_state_single_and_batch(self) -> None:
        state = {
            "epsilon": 1e-8,
            "clip_obs": 10.0,
            "obs_rms": {
                "mean": np.asarray([1.0, -1.0], dtype=np.float64),
                "var": np.asarray([4.0, 1.0], dtype=np.float64),
                "count": 10.0,
            },
        }
        single = np.asarray([3.0, -2.0], dtype=np.float32)
        batch = np.asarray([[3.0, -2.0], [1.0, -1.0]], dtype=np.float32)
        out_single = normalize_obs_with_state(single, state)
        out_batch = normalize_obs_with_state(batch, state)
        np.testing.assert_allclose(out_single, np.asarray([1.0, -1.0], dtype=np.float32), atol=1e-5)
        np.testing.assert_allclose(
            out_batch,
            np.asarray([[1.0, -1.0], [0.0, 0.0]], dtype=np.float32),
            atol=1e-5,
        )

    def test_normalize_infos_respects_mask(self) -> None:
        envs = gym.vector.SyncVectorEnv([lambda: _ToyEnv(), lambda: _ToyEnv()])
        vec = VecNormalize(envs, norm_obs=True, norm_reward=False, training=False)
        try:
            vec.obs_rms.mean = np.asarray([1.0, -1.0], dtype=np.float64)
            vec.obs_rms.var = np.asarray([1.0, 1.0], dtype=np.float64)
            vec.epsilon = 1e-8
            infos = {
                "final_observation": np.asarray([[1.0, -1.0], [3.0, -3.0]], dtype=np.float32),
                "_final_observation": np.asarray([True, False], dtype=np.bool_),
            }
            out = vec._normalize_infos(infos)
            got = np.asarray(out["final_observation"], dtype=np.float32)
            np.testing.assert_allclose(got[0], np.asarray([0.0, 0.0], dtype=np.float32), atol=1e-5)
            np.testing.assert_allclose(got[1], np.asarray([3.0, -3.0], dtype=np.float32), atol=1e-5)
        finally:
            vec.close()
