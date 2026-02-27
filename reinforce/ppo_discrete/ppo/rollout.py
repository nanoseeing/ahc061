from __future__ import annotations

from dataclasses import dataclass

import torch

from .vecnorm import VecNormalize
from ..env.interface import BatchEnvProtocol, ensure_batch_env
from ..game_constants import OPP_PARAM_TRUE_TAIL_SHAPE, OPP_VALID_TAIL_SHAPE


@dataclass
class Rollout:
    obs: torch.Tensor  # [T, B, C, N, N] (CPU, float32)
    masks: torch.Tensor  # [T, B, A] (CPU, uint8)
    actions: torch.Tensor  # [T, B] (CPU, int64)
    logprobs: torch.Tensor  # [T, B] (CPU, float32)
    values: torch.Tensor  # [T, B] (CPU, float32)
    rewards: torch.Tensor  # [T, B] (CPU, float32)
    dones: torch.Tensor  # [T, B] (CPU, uint8); done-before-action
    last_done: torch.Tensor  # [B] (CPU, uint8)
    last_value: torch.Tensor  # [B] (CPU, float32)
    aux_opp_param_true: torch.Tensor | None = None  # [T, B, 7, 5] (CPU, float32)
    aux_opp_valid: torch.Tensor | None = None  # [T, B, 7] (CPU, uint8)


@dataclass
class RolloutWorkspace:
    obs: torch.Tensor
    masks: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    next_obs: torch.Tensor
    next_mask: torch.Tensor
    last_done: torch.Tensor
    last_value: torch.Tensor
    aux_opp_param_true: torch.Tensor | None = None
    aux_opp_valid: torch.Tensor | None = None
    aux_move_dist_tmp: torch.Tensor | None = None
    aux_opp_param_tmp: torch.Tensor | None = None
    aux_opp_valid_tmp: torch.Tensor | None = None
    board_dev: torch.Tensor | None = None
    mask_dev: torch.Tensor | None = None
    logp_dev: torch.Tensor | None = None
    values_dev: torch.Tensor | None = None


def _empty_cpu(shape, *, dtype: torch.dtype, pin_memory: bool) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=pin_memory)


def create_rollout_workspace(
    env: BatchEnvProtocol,
    *,
    num_steps: int,
    device: torch.device,
    channels_last: bool = False,
    pin_memory: bool = False,
    collect_aux_targets: bool = False,
) -> RolloutWorkspace:
    env = ensure_batch_env(env)
    bsz = int(env.batch_size)
    c = int(env.feature_channels)
    n = int(env.board_size)
    a = int(env.action_dim)
    t_max = int(num_steps)
    use_cuda = device.type == "cuda"
    use_channels_last = bool(use_cuda and channels_last)
    use_pinned = bool(use_cuda and pin_memory)
    aux_opp_slots = int(OPP_PARAM_TRUE_TAIL_SHAPE[0])
    aux_param_dim = int(OPP_PARAM_TRUE_TAIL_SHAPE[1])
    aux_valid_slots = int(OPP_VALID_TAIL_SHAPE[0])
    if bool(collect_aux_targets) and aux_valid_slots != aux_opp_slots:
        raise ValueError(
            f"aux opp slot mismatch: opp_param_true slots={aux_opp_slots}, opp_valid slots={aux_valid_slots}"
        )
    m_max = int(env.spec.m_max)
    if bool(collect_aux_targets) and m_max < aux_opp_slots + 1:
        raise ValueError(f"env.spec.m_max is too small for aux targets: m_max={m_max}, need>={aux_opp_slots + 1}")

    def _alloc_cpu(pin: bool) -> tuple[torch.Tensor, ...]:
        return (
            _empty_cpu((t_max, bsz, c, n, n), dtype=torch.float32, pin_memory=pin),
            _empty_cpu((t_max, bsz, a), dtype=torch.uint8, pin_memory=pin),
            _empty_cpu((t_max, bsz), dtype=torch.int64, pin_memory=pin),
            _empty_cpu((t_max, bsz), dtype=torch.float32, pin_memory=pin),
            _empty_cpu((t_max, bsz), dtype=torch.float32, pin_memory=pin),
            _empty_cpu((t_max, bsz), dtype=torch.float32, pin_memory=pin),
            _empty_cpu((t_max, bsz), dtype=torch.uint8, pin_memory=pin),
            _empty_cpu((bsz, c, n, n), dtype=torch.float32, pin_memory=pin),
            _empty_cpu((bsz, a), dtype=torch.uint8, pin_memory=pin),
            _empty_cpu((bsz,), dtype=torch.uint8, pin_memory=pin),
            _empty_cpu((bsz,), dtype=torch.float32, pin_memory=pin),
        )

    def _alloc_aux_cpu(pin: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            _empty_cpu((t_max, bsz, aux_opp_slots, aux_param_dim), dtype=torch.float32, pin_memory=pin),
            _empty_cpu((t_max, bsz, aux_opp_slots), dtype=torch.uint8, pin_memory=pin),
            _empty_cpu((bsz, m_max, a), dtype=torch.float32, pin_memory=pin),
            _empty_cpu((bsz, m_max, aux_param_dim), dtype=torch.float32, pin_memory=pin),
            _empty_cpu((bsz, m_max), dtype=torch.uint8, pin_memory=pin),
        )

    def _do_alloc(pin: bool) -> tuple[tuple[torch.Tensor, ...], tuple]:
        main = _alloc_cpu(pin)
        aux = _alloc_aux_cpu(pin) if bool(collect_aux_targets) else (None, None, None, None, None)
        return main, aux

    if use_pinned:
        try:
            main_tensors, aux_tensors = _do_alloc(True)
        except RuntimeError:
            main_tensors, aux_tensors = _do_alloc(False)
    else:
        main_tensors, aux_tensors = _do_alloc(False)

    (obs, masks, actions, logprobs, values, rewards, dones, next_obs, next_mask, last_done, last_value) = main_tensors
    (aux_opp_param_true, aux_opp_valid, aux_move_dist_tmp, aux_opp_param_tmp, aux_opp_valid_tmp) = aux_tensors

    board_dev = None
    mask_dev = None
    logp_dev = None
    values_dev = None
    if use_cuda:
        if use_channels_last:
            board_dev = torch.empty(
                (bsz, c, n, n),
                dtype=torch.float32,
                device=device,
                memory_format=torch.channels_last,
            )
        else:
            board_dev = torch.empty((bsz, c, n, n), dtype=torch.float32, device=device)
        mask_dev = torch.empty((bsz, a), dtype=torch.uint8, device=device)
        # Avoid many small D2H copies. Write per-step logits/values on device and copy [T, B] once.
        logp_dev = torch.empty((t_max, bsz), dtype=torch.float32, device=device)
        values_dev = torch.empty((t_max, bsz), dtype=torch.float32, device=device)

    return RolloutWorkspace(
        obs=obs,
        masks=masks,
        actions=actions,
        logprobs=logprobs,
        values=values,
        rewards=rewards,
        dones=dones,
        next_obs=next_obs,
        next_mask=next_mask,
        last_done=last_done,
        last_value=last_value,
        aux_opp_param_true=aux_opp_param_true,
        aux_opp_valid=aux_opp_valid,
        aux_move_dist_tmp=aux_move_dist_tmp,
        aux_opp_param_tmp=aux_opp_param_tmp,
        aux_opp_valid_tmp=aux_opp_valid_tmp,
        board_dev=board_dev,
        mask_dev=mask_dev,
        logp_dev=logp_dev,
        values_dev=values_dev,
    )


@torch.inference_mode()
def collect_rollout(
    env: BatchEnvProtocol,
    agent: torch.nn.Module,
    device: torch.device,
    num_steps: int,
    *,
    use_action_mask: bool,
    sample: bool,
    amp: bool = False,
    channels_last: bool = False,
    workspace: RolloutWorkspace | None = None,
    pin_memory: bool = False,
    vecnorm: VecNormalize | None = None,
    collect_aux_targets: bool = False,
) -> Rollout:
    env = ensure_batch_env(env)
    bsz = int(env.batch_size)
    c = int(env.feature_channels)
    n = int(env.board_size)
    a = int(env.action_dim)
    t_max = int(num_steps)
    use_cuda = device.type == "cuda"
    use_amp = bool(use_cuda and amp)

    if workspace is None:
        workspace = create_rollout_workspace(
            env,
            num_steps=t_max,
            device=device,
            channels_last=bool(channels_last),
            pin_memory=bool(pin_memory),
            collect_aux_targets=bool(collect_aux_targets),
        )

    obs = workspace.obs
    masks = workspace.masks
    actions = workspace.actions
    logprobs = workspace.logprobs
    values = workspace.values
    rewards = workspace.rewards
    dones = workspace.dones
    next_obs = workspace.next_obs
    next_mask = workspace.next_mask
    last_done = workspace.last_done
    last_value = workspace.last_value
    aux_opp_param_true = workspace.aux_opp_param_true
    aux_opp_valid = workspace.aux_opp_valid
    aux_move_dist_tmp = workspace.aux_move_dist_tmp
    aux_opp_param_tmp = workspace.aux_opp_param_tmp
    aux_opp_valid_tmp = workspace.aux_opp_valid_tmp
    aux_opp_slots = int(OPP_PARAM_TRUE_TAIL_SHAPE[0])
    aux_param_dim = int(OPP_PARAM_TRUE_TAIL_SHAPE[1])
    aux_active = bool(collect_aux_targets)

    expected_obs_shape = (t_max, bsz, c, n, n)
    if tuple(obs.shape) != tuple(expected_obs_shape):
        raise ValueError(f"workspace.obs shape mismatch: expected={expected_obs_shape}, got={tuple(obs.shape)}")
    if tuple(masks.shape) != (t_max, bsz, a):
        raise ValueError(f"workspace.masks shape mismatch: expected={(t_max, bsz, a)}, got={tuple(masks.shape)}")
    if aux_active:
        if (
            aux_opp_param_true is None
            or aux_opp_valid is None
            or aux_move_dist_tmp is None
            or aux_opp_param_tmp is None
            or aux_opp_valid_tmp is None
        ):
            raise RuntimeError("collect_aux_targets=True requires aux_* tensors in workspace")
        if tuple(aux_opp_param_true.shape) != (t_max, bsz, aux_opp_slots, aux_param_dim):
            raise ValueError(
                "workspace.aux_opp_param_true shape mismatch: "
                f"expected={(t_max, bsz, aux_opp_slots, aux_param_dim)}, got={tuple(aux_opp_param_true.shape)}"
            )
        if tuple(aux_opp_valid.shape) != (t_max, bsz, aux_opp_slots):
            raise ValueError(
                f"workspace.aux_opp_valid shape mismatch: expected={(t_max, bsz, aux_opp_slots)}, got={tuple(aux_opp_valid.shape)}"
            )

    board_dev = workspace.board_dev
    mask_dev = workspace.mask_dev
    logp_dev = workspace.logp_dev
    values_dev = workspace.values_dev
    if use_cuda and (board_dev is None or mask_dev is None or logp_dev is None or values_dev is None):
        raise RuntimeError("CUDA rollout requires board_dev/mask_dev/logp_dev/values_dev in workspace")

    last_done.zero_()
    was_training = agent.training
    agent.eval()
    env.observe_into(obs[0], masks[0])
    if vecnorm is not None:
        vecnorm.normalize_obs_inplace(obs[0])
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp)

    for t in range(t_max):
        dones[t].copy_(last_done)

        if use_cuda:
            assert board_dev is not None
            assert mask_dev is not None
            board_dev.copy_(obs[t], non_blocking=True)
            board = board_dev
            mask_model = None
            if use_action_mask:
                mask_dev.copy_(masks[t], non_blocking=True)
                mask_model = mask_dev
        else:
            board = obs[t]
            mask_model = masks[t] if use_action_mask else None

        with autocast:
            if sample:
                action_t, logp_t, _ent_t, value_t = agent.get_action_and_value(
                    board,
                    action_mask=mask_model,
                )
            else:
                action_t = agent.act(board, action_mask=mask_model, deterministic=True)
                _a, logp_t, _ent_t, value_t = agent.get_action_and_value(
                    board,
                    action=action_t,
                    action_mask=mask_model,
                )

        action_cpu = action_t.to(dtype=torch.int64, device="cpu").contiguous()
        # actions are consumed immediately by env.step_observe_into(), so keep this copy synchronous.
        actions[t].copy_(action_cpu, non_blocking=False)

        if use_cuda:
            assert logp_dev is not None
            assert values_dev is not None
            logp_dev[t].copy_(logp_t.float(), non_blocking=True)
            values_dev[t].copy_(value_t.view(-1).float(), non_blocking=True)
        else:
            logprobs[t].copy_(logp_t.float(), non_blocking=True)
            values[t].copy_(value_t.view(-1).float(), non_blocking=True)

        if t + 1 < t_max:
            if aux_active:
                assert aux_move_dist_tmp is not None
                assert aux_opp_param_tmp is not None
                assert aux_opp_valid_tmp is not None
                assert aux_opp_param_true is not None
                assert aux_opp_valid is not None
                env.step_observe_aux_into(
                    action_cpu,
                    obs[t + 1],
                    masks[t + 1],
                    rewards[t],
                    last_done,
                    aux_move_dist_tmp,
                    aux_opp_param_tmp,
                    aux_opp_valid_tmp,
                )
                aux_opp_param_true[t].copy_(
                    aux_opp_param_tmp[:, 1 : 1 + aux_opp_slots, :],
                    non_blocking=False,
                )
                aux_opp_valid[t].copy_(
                    aux_opp_valid_tmp[:, 1 : 1 + aux_opp_slots],
                    non_blocking=False,
                )
            else:
                env.step_observe_into(action_cpu, obs[t + 1], masks[t + 1], rewards[t], last_done)
            if vecnorm is not None:
                vecnorm.normalize_reward_inplace(rewards[t], last_done)
                vecnorm.normalize_obs_inplace(obs[t + 1])
        else:
            if aux_active:
                assert aux_move_dist_tmp is not None
                assert aux_opp_param_tmp is not None
                assert aux_opp_valid_tmp is not None
                assert aux_opp_param_true is not None
                assert aux_opp_valid is not None
                env.step_observe_aux_into(
                    action_cpu,
                    next_obs,
                    next_mask,
                    rewards[t],
                    last_done,
                    aux_move_dist_tmp,
                    aux_opp_param_tmp,
                    aux_opp_valid_tmp,
                )
                aux_opp_param_true[t].copy_(
                    aux_opp_param_tmp[:, 1 : 1 + aux_opp_slots, :],
                    non_blocking=False,
                )
                aux_opp_valid[t].copy_(
                    aux_opp_valid_tmp[:, 1 : 1 + aux_opp_slots],
                    non_blocking=False,
                )
            else:
                env.step_observe_into(action_cpu, next_obs, next_mask, rewards[t], last_done)
            if vecnorm is not None:
                vecnorm.normalize_reward_inplace(rewards[t], last_done)
                vecnorm.normalize_obs_inplace(next_obs)

    if use_cuda:
        assert logp_dev is not None
        assert values_dev is not None
        logprobs.copy_(logp_dev, non_blocking=True)
        values.copy_(values_dev, non_blocking=True)

    if use_cuda:
        assert board_dev is not None
        board_dev.copy_(next_obs, non_blocking=True)
        board = board_dev
    else:
        board = next_obs

    with autocast:
        last_v = agent.get_value(board).view(-1).float()
    if use_cuda:
        last_value.copy_(last_v, non_blocking=True)
        # Ensure non_blocking D2H copies complete before the caller consumes CPU tensors.
        torch.cuda.current_stream(device).synchronize()
    else:
        last_value.copy_(last_v, non_blocking=False)

    if was_training:
        agent.train()

    return Rollout(
        obs=obs,
        masks=masks,
        actions=actions,
        logprobs=logprobs,
        values=values,
        rewards=rewards,
        dones=dones,
        last_done=last_done,
        last_value=last_value,
        aux_opp_param_true=aux_opp_param_true,
        aux_opp_valid=aux_opp_valid,
    )
