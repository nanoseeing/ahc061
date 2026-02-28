"""`trainer` に関するモジュール。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .config import PPOConfig
from .rollout_buffer import RolloutBuffer


@dataclass
class UpdateStats:
    """`UpdateStats` を表すデータクラス。

    Attributes:
        policy_loss (float): フィールド値。
        value_loss (float): フィールド値。
        entropy (float): フィールド値。
        approx_kl (float): フィールド値。
        clipfrac (float): フィールド値。
        value_clipfrac (float): フィールド値。
        update_epochs_used (int): フィールド値。
        early_stop_by_kl (bool): フィールド値。
        target_kl_threshold (float | None): フィールド値。
        aux_opp_param_loss (float): フィールド値。
    """
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clipfrac: float
    value_clipfrac: float
    update_epochs_used: int
    early_stop_by_kl: bool
    target_kl_threshold: float | None
    aux_opp_param_loss: float


def aux_opp_param_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    use_valid_mask: bool = True,
) -> torch.Tensor:
    """`aux_opp_param_mse` を実行する。

    Args:
        pred (torch.Tensor): pred の値。
        target (torch.Tensor): target の値。
        valid (torch.Tensor): valid の値。
        use_valid_mask (bool): 有効化フラグ。

    Returns:
        torch.Tensor: 計算結果。
    """
    if pred.ndim != 3 or target.ndim != 3:
        raise ValueError(f"aux opp tensors must be rank-3: pred={tuple(pred.shape)} target={tuple(target.shape)}")
    if tuple(pred.shape) != tuple(target.shape):
        raise ValueError(f"aux opp target shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}")
    if valid.ndim != 2 or tuple(valid.shape) != tuple(pred.shape[:2]):
        raise ValueError(f"aux opp valid shape mismatch: valid={tuple(valid.shape)} expected={tuple(pred.shape[:2])}")
    diff2 = (pred - target) ** 2
    if bool(use_valid_mask):
        w = valid.to(dtype=pred.dtype).unsqueeze(-1)
        den = torch.clamp(w.sum(), min=1.0)
        return (diff2 * w).sum() / den
    return diff2.mean()


class PPOTrainer:
    """`PPOTrainer` を表すクラス。"""

    def __init__(
        self,
        cfg: PPOConfig,
        agent: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        use_channels_last: bool = False,
    ):
        """インスタンスを初期化する。

        Args:
            cfg (PPOConfig): 設定オブジェクト。
            agent (nn.Module): agent の値。
            optimizer (torch.optim.Optimizer): optimizer の値。
            use_channels_last (bool): 有効化フラグ。
        """
        self.cfg = cfg
        self.agent = agent
        self.optimizer = optimizer
        self.runtime_ent_coef: float | None = None
        self.runtime_clip_coef: float | None = None
        self.runtime_clip_range_vf: float | None = None
        self.runtime_aux_opp_param_loss_coef: float | None = None
        self.runtime_aux_opp_param_use_valid_mask: bool | None = None
        self.use_channels_last = bool(use_channels_last)

    def _model_device(self) -> torch.device:
        """内部ヘルパー: `model_device` を実行する。

        Returns:
            torch.device: 計算結果。
        """
        try:
            p = next(self.agent.parameters())
            return p.device
        except StopIteration:
            return torch.device("cpu")

    def _target_kl_threshold(self) -> float | None:
        """内部ヘルパー: `target_kl_threshold` を実行する。

        Returns:
            float | None: 計算結果。
        """
        if self.cfg.target_kl is None:
            return None
        # SB3-compatible semantics: early-stop when approx_kl > 1.5 * target_kl.
        return float(self.cfg.target_kl) * 1.5

    def set_runtime_coefficients(
        self,
        *,
        ent_coef: float | None = None,
        clip_coef: float | None = None,
        clip_range_vf: float | None = None,
        aux_opp_param_loss_coef: float | None = None,
        aux_opp_param_use_valid_mask: bool | None = None,
    ) -> None:
        """`runtime_coefficients`を設定する。

        Args:
            ent_coef (float | None): ent_coef の値。
            clip_coef (float | None): clip_coef の値。
            clip_range_vf (float | None): clip_range_vf の値。
            aux_opp_param_loss_coef (float | None): aux_opp_param_loss_coef の値。
            aux_opp_param_use_valid_mask (bool | None): 有効化フラグ。
        """
        self.runtime_ent_coef = None if ent_coef is None else float(ent_coef)
        self.runtime_clip_coef = None if clip_coef is None else float(clip_coef)
        self.runtime_clip_range_vf = None if clip_range_vf is None else float(clip_range_vf)
        self.runtime_aux_opp_param_loss_coef = (
            None if aux_opp_param_loss_coef is None else float(aux_opp_param_loss_coef)
        )
        self.runtime_aux_opp_param_use_valid_mask = (
            None if aux_opp_param_use_valid_mask is None else bool(aux_opp_param_use_valid_mask)
        )

    def update(self, buffer: RolloutBuffer) -> UpdateStats:
        """`update` を実行する。

        Args:
            buffer (RolloutBuffer): buffer の値。

        Returns:
            UpdateStats: 計算結果。
        """
        batch = buffer.flatten()
        b_inds = buffer.shuffled_indices()
        model_device = self._model_device()
        use_non_blocking = model_device.type == "cuda"
        clip_coef = float(self.runtime_clip_coef if self.runtime_clip_coef is not None else self.cfg.clip_coef)
        ent_coef = float(self.runtime_ent_coef if self.runtime_ent_coef is not None else self.cfg.ent_coef)
        aux_coef_cfg = getattr(self.cfg, "aux_opp_param_loss_coef", 0.0)
        aux_coef = float(
            max(
                0.0,
                self.runtime_aux_opp_param_loss_coef
                if self.runtime_aux_opp_param_loss_coef is not None
                else float(aux_coef_cfg),
            )
        )
        aux_use_valid_mask = (
            bool(self.runtime_aux_opp_param_use_valid_mask)
            if self.runtime_aux_opp_param_use_valid_mask is not None
            else bool(getattr(self.cfg, "aux_opp_param_use_valid_mask", True))
        )
        aux_active = bool(aux_coef > 0.0)
        if aux_active and (batch.aux_opp_param_true is None or batch.aux_opp_valid is None):
            raise RuntimeError("aux_opp_param_loss_coef > 0 requires rollout buffer aux_opp_param targets")
        target_kl_threshold = self._target_kl_threshold()
        update_epochs_used = 0
        early_stop_by_kl = False

        last_pg_loss = torch.tensor(0.0, device=model_device)
        last_v_loss = torch.tensor(0.0, device=model_device)
        last_entropy = torch.tensor(0.0, device=model_device)

        # Tensor accumulators — avoid .item() in the hot loop; materialize once after all epochs.
        pg_loss_sum = torch.tensor(0.0, device=model_device)
        v_loss_sum = torch.tensor(0.0, device=model_device)
        entropy_sum = torch.tensor(0.0, device=model_device)
        clipfrac_sum = torch.tensor(0.0, device=model_device)
        value_clipfrac_sum = torch.tensor(0.0, device=model_device)
        aux_loss_sum = torch.tensor(0.0, device=model_device)
        total_mb = 0
        value_clipfrac_count = 0
        aux_loss_count = 0
        last_epoch_kl = torch.tensor(0.0, device=model_device)

        # Normalize advantages over the full batch once (before epoch loop), matching exp002.
        # Uses population variance: var = E[adv²] - E[adv]²
        if self.cfg.norm_adv:
            adv = batch.advantages.float()
            adv_n = float(adv.numel())
            adv_mean = adv.sum() / adv_n
            adv_var = (adv.pow(2).sum() / adv_n - adv_mean.pow(2)).clamp(min=0.0)
            batch.advantages = ((adv - adv_mean) / (adv_var.sqrt() + 1e-8)).to(batch.advantages.dtype)

        for _ in range(self.cfg.update_epochs):
            update_epochs_used += 1
            epoch_kl_sum = torch.tensor(0.0, device=model_device)
            epoch_mb_count = 0
            np.random.shuffle(b_inds)
            for start in range(0, self.cfg.batch_size, self.cfg.minibatch_size):
                end = start + self.cfg.minibatch_size
                mb_inds = b_inds[start:end]

                obs_mb = batch.obs[mb_inds]
                actions_mb = batch.actions.long()[mb_inds].view(-1)
                old_logprobs_mb = batch.logprobs[mb_inds]
                old_values_mb = batch.values[mb_inds]
                advantages_mb = batch.advantages[mb_inds]
                returns_mb = batch.returns[mb_inds]

                action_mask_mb = None
                if batch.action_masks is not None:
                    action_mask_mb = batch.action_masks[mb_inds]

                if obs_mb.device != model_device:
                    obs_mb = obs_mb.to(device=model_device, non_blocking=use_non_blocking)
                if self.use_channels_last and obs_mb.ndim == 4 and model_device.type == "cuda":
                    obs_mb = obs_mb.contiguous(memory_format=torch.channels_last)

                if actions_mb.device != model_device:
                    actions_mb = actions_mb.to(device=model_device, non_blocking=use_non_blocking)
                if old_logprobs_mb.device != model_device:
                    old_logprobs_mb = old_logprobs_mb.to(device=model_device, non_blocking=use_non_blocking)
                if old_values_mb.device != model_device:
                    old_values_mb = old_values_mb.to(device=model_device, non_blocking=use_non_blocking)
                if advantages_mb.device != model_device:
                    advantages_mb = advantages_mb.to(device=model_device, non_blocking=use_non_blocking)
                if returns_mb.device != model_device:
                    returns_mb = returns_mb.to(device=model_device, non_blocking=use_non_blocking)
                if action_mask_mb is not None and action_mask_mb.device != model_device:
                    action_mask_mb = action_mask_mb.to(device=model_device, non_blocking=use_non_blocking)

                aux_pred_mb: torch.Tensor | None = None
                try:
                    forward_out = self.agent(
                        obs_mb,
                        action=actions_mb,
                        action_mask=action_mask_mb,
                        return_aux_opp_param=bool(aux_active),
                    )
                except TypeError as e:
                    raise RuntimeError(
                        "agent.forward must support (obs, action=..., action_mask=..., "
                        "return_aux_opp_param=...)"
                    ) from e

                if isinstance(forward_out, (tuple, list)):
                    if len(forward_out) == 5:
                        _, newlogprob, entropy, newvalue, aux_pred_mb = forward_out
                    elif len(forward_out) == 4:
                        _, newlogprob, entropy, newvalue = forward_out
                    else:
                        raise RuntimeError(f"agent forward returned unexpected tuple length: {len(forward_out)}")
                else:
                    raise RuntimeError("agent forward must return tuple(action, logprob, entropy, value[, aux])")

                if aux_active and aux_pred_mb is None:
                    aux_fn = getattr(self.agent, "get_aux_opp_param", None)
                    if not callable(aux_fn):
                        raise RuntimeError(
                            "aux_opp_param_loss_coef > 0 requires model to implement "
                            "get_aux_opp_param(obs) or forward(..., return_aux_opp_param=True)"
                        )
                    aux_pred_mb = aux_fn(obs_mb)
                logratio = newlogprob - old_logprobs_mb
                ratio = logratio.exp()

                with torch.no_grad():
                    last_approx_kl = ((ratio - 1.0) - logratio).mean()
                    epoch_kl_sum += last_approx_kl
                    epoch_mb_count += 1
                    clipfrac_sum += ((ratio - 1.0).abs() > clip_coef).float().mean()

                mb_advantages = advantages_mb

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                last_pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                pg_loss_sum += last_pg_loss.detach()

                newvalue = newvalue.view(-1)
                vf_clip_cfg = self.runtime_clip_range_vf
                if vf_clip_cfg is None:
                    vf_clip_cfg = self.cfg.clip_range_vf
                if self.cfg.clip_vloss and vf_clip_cfg is not None:
                    vf_clip = float(vf_clip_cfg)
                    with torch.no_grad():
                        value_clipfrac_sum += ((newvalue - old_values_mb).abs() > vf_clip).float().mean()
                        value_clipfrac_count += 1
                    v_loss_unclipped = (newvalue - returns_mb) ** 2
                    v_clipped = old_values_mb + torch.clamp(
                        newvalue - old_values_mb,
                        -vf_clip,
                        vf_clip,
                    )
                    v_loss_clipped = (v_clipped - returns_mb) ** 2
                    last_v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    last_v_loss = 0.5 * ((newvalue - returns_mb) ** 2).mean()
                v_loss_sum += last_v_loss.detach()

                last_entropy = entropy.mean()
                entropy_sum += last_entropy.detach()
                total_mb += 1

                aux_loss = torch.tensor(0.0, device=model_device)
                if aux_active:
                    assert batch.aux_opp_param_true is not None
                    assert batch.aux_opp_valid is not None
                    assert aux_pred_mb is not None
                    aux_true_mb = batch.aux_opp_param_true[mb_inds]
                    aux_valid_mb = batch.aux_opp_valid[mb_inds]
                    if aux_true_mb.device != model_device:
                        aux_true_mb = aux_true_mb.to(device=model_device, non_blocking=use_non_blocking)
                    if aux_valid_mb.device != model_device:
                        aux_valid_mb = aux_valid_mb.to(device=model_device, non_blocking=use_non_blocking)
                    aux_loss = aux_opp_param_mse(
                        aux_pred_mb.float(),
                        aux_true_mb.float(),
                        aux_valid_mb,
                        use_valid_mask=bool(aux_use_valid_mask),
                    )
                    aux_loss_sum += aux_loss.detach()
                    aux_loss_count += 1
                loss = last_pg_loss - ent_coef * last_entropy + self.cfg.vf_coef * last_v_loss + aux_coef * aux_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

            # One .item() per epoch at most — only when early stopping is configured.
            if epoch_mb_count > 0:
                last_epoch_kl = epoch_kl_sum / epoch_mb_count
            if target_kl_threshold is not None and float(last_epoch_kl.item()) > target_kl_threshold:
                early_stop_by_kl = True
                break

        n = max(total_mb, 1)
        return UpdateStats(
            policy_loss=float((pg_loss_sum / n).item()),
            value_loss=float((v_loss_sum / n).item()),
            entropy=float((entropy_sum / n).item()),
            approx_kl=float(last_epoch_kl.item()),
            clipfrac=float((clipfrac_sum / n).item()),
            value_clipfrac=(
                float((value_clipfrac_sum / value_clipfrac_count).item())
                if value_clipfrac_count > 0
                else float("nan")
            ),
            update_epochs_used=int(update_epochs_used),
            early_stop_by_kl=bool(early_stop_by_kl),
            target_kl_threshold=(float(target_kl_threshold) if target_kl_threshold is not None else None),
            aux_opp_param_loss=(
                float((aux_loss_sum / aux_loss_count).item())
                if aux_loss_count > 0
                else float("nan")
            ),
        )
