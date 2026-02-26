from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np
import torch

from ..domains.ahc061.env import AHC061LocalEnv, CaseData, RuntimeState
from ..domains.ahc061.opponent_bayes import create_opponent_bayes_estimator
from ..env.vec_normalize import normalize_obs_with_state
from ..runtime.checkpoint import load_agent_checkpoint


@dataclass
class IOConfig:
    n: int
    m: int
    t: int
    u: int
    values: np.ndarray  # [n,n]
    start_xy: list[tuple[int, int]]


class TokenReader:
    def __init__(self):
        self.buf: list[str] = []

    def _fill(self) -> None:
        line = sys.stdin.buffer.readline()
        if not line:
            raise EOFError("unexpected EOF")
        toks = line.decode("utf-8").strip().split()
        self.buf.extend(toks)

    def next(self) -> str:
        while not self.buf:
            self._fill()
        return self.buf.pop(0)

    def next_int(self) -> int:
        return int(self.next())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AHC061 stdin/stdout agent using ppo_discrete checkpoint.")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--use-action-mask", action="store_true", default=True)
    p.add_argument("--bayes-num-particles", type=int, default=128)
    p.add_argument("--bayes-seed", type=int, default=0)
    p.add_argument("--bayes-backend", choices=["auto", "python", "cpp"], default="auto")
    p.add_argument("--vecnorm-mode", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--debug-stderr", action="store_true")
    return p.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def read_initial(reader: TokenReader) -> IOConfig:
    n = reader.next_int()
    m = reader.next_int()
    t = reader.next_int()
    u = reader.next_int()
    values = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(n):
            values[i, j] = reader.next_int()
    start_xy: list[tuple[int, int]] = []
    for _ in range(m):
        x = reader.next_int()
        y = reader.next_int()
        start_xy.append((x, y))
    return IOConfig(n=n, m=m, t=t, u=u, values=values, start_xy=start_xy)


def to_case(cfg: IOConfig) -> CaseData:
    m = cfg.m
    t = cfg.t
    return CaseData(
        n=cfg.n,
        m=cfg.m,
        t=cfg.t,
        u=cfg.u,
        values=cfg.values,
        start_xy=cfg.start_xy,
        wa=np.zeros((max(0, m - 1),), dtype=np.float64),
        wb=np.zeros((max(0, m - 1),), dtype=np.float64),
        wc=np.zeros((max(0, m - 1),), dtype=np.float64),
        wd=np.zeros((max(0, m - 1),), dtype=np.float64),
        eps=np.zeros((max(0, m - 1),), dtype=np.float64),
        r=np.zeros((max(0, m - 1), 2 * max(1, t)), dtype=np.float64),
    )


def encode_obs(case: CaseData, st: RuntimeState, turn: int, scores: np.ndarray, bayes_vec: np.ndarray) -> np.ndarray:
    # Must match AHC061LocalEnv._encode_obs for checkpoint compatibility.
    n = case.n
    value_scale = 1000.0

    vals = np.zeros((10, 10), dtype=np.float32)
    owner_self = np.zeros((10, 10), dtype=np.float32)
    owner_enemy = np.zeros((10, 10), dtype=np.float32)
    owner_neutral = np.zeros((10, 10), dtype=np.float32)
    level_norm = np.zeros((10, 10), dtype=np.float32)
    piece_self = np.zeros((10, 10), dtype=np.float32)
    piece_enemy = np.zeros((10, 10), dtype=np.float32)

    for i in range(n):
        for j in range(n):
            vals[i, j] = float(case.values[i, j] / value_scale)
            o = int(st.owner[i, j])
            if o == 0:
                owner_self[i, j] = 1.0
            elif o == -1:
                owner_neutral[i, j] = 1.0
            else:
                owner_enemy[i, j] = 1.0
            level_norm[i, j] = float(st.level[i, j] / max(1, case.u))

    piece_self[st.px[0], st.py[0]] = 1.0
    for p in range(1, case.m):
        piece_enemy[st.px[p], st.py[p]] = 1.0

    board_vec = np.concatenate(
        [
            vals.reshape(-1),
            owner_self.reshape(-1),
            owner_enemy.reshape(-1),
            owner_neutral.reshape(-1),
            level_norm.reshape(-1),
            piece_self.reshape(-1),
            piece_enemy.reshape(-1),
        ]
    )

    globals_base = np.zeros((21,), dtype=np.float32)
    globals_base[0] = float(turn / max(1, case.t))
    globals_base[1] = float((case.m - 2) / 6.0)
    globals_base[2] = float((case.u - 1) / 4.0)
    globals_base[3] = float(st.px[0] / max(1, n - 1))
    globals_base[4] = float(st.py[0] / max(1, n - 1))

    off = 5
    for ei in range(7):
        if ei + 1 < case.m:
            p = ei + 1
            globals_base[off + 2 * ei] = float(st.px[p] / max(1, n - 1))
            globals_base[off + 2 * ei + 1] = float(st.py[p] / max(1, n - 1))
        else:
            globals_base[off + 2 * ei] = 0.0
            globals_base[off + 2 * ei + 1] = 0.0

    s_cap = float(max(1, case.u * np.sum(case.values)))
    s0 = float(scores[0])
    sa = float(np.max(scores[1:])) if case.m > 1 else 1.0
    globals_base[19] = float(np.clip(s0 / s_cap, 0.0, 1.0))
    globals_base[20] = float(np.clip(sa / s_cap, 0.0, 1.0))

    if bayes_vec.shape != (28,):
        raise ValueError(f"unexpected bayes feature shape: {bayes_vec.shape}")
    return np.concatenate([board_vec, globals_base, bayes_vec.astype(np.float32)]).astype(np.float32)


def decode_action(a: int) -> tuple[int, int]:
    x = int(a // 10)
    y = int(a % 10)
    return x, y


def main() -> int:
    args = parse_args()
    device = choose_device(args.device)
    agent, meta = load_agent_checkpoint(args.model_path, device=device)
    agent.eval()

    vec_state = meta.get("vecnormalize_state") if isinstance(meta, dict) else None
    vecnorm_mode = str(args.vecnorm_mode).lower().strip()
    use_vecnorm = bool(vecnorm_mode == "on" or (vecnorm_mode == "auto" and isinstance(vec_state, dict)))
    if vecnorm_mode == "on" and not isinstance(vec_state, dict):
        print("[warn] vecnorm-mode=on but checkpoint has no vecnormalize_state; proceeding without vecnorm", file=sys.stderr)
        use_vecnorm = False

    reader = TokenReader()
    cfg = read_initial(reader)
    case = to_case(cfg)
    st = AHC061LocalEnv._init_state(case)

    bayes = create_opponent_bayes_estimator(
        n=case.n,
        m=case.m,
        u=case.u,
        num_particles=args.bayes_num_particles,
        seed=args.bayes_seed,
        backend=args.bayes_backend,
    )

    expected_obs_dim = 7 * 100 + 21 + 28
    if tuple(agent.obs_shape) != (expected_obs_dim,):
        print(
            f"[warn] unexpected obs_shape={agent.obs_shape}; expected ({expected_obs_dim},)",
            file=sys.stderr,
        )
    if int(agent.action_dim) != 100:
        print(f"[warn] unexpected action_dim={agent.action_dim}; expected 100", file=sys.stderr)

    for turn in range(cfg.t):
        scores = AHC061LocalEnv._score_all(st, case)
        candidates_all: list[list[tuple[int, int]]] = [AHC061LocalEnv._get_candidates(st, case, p) for p in range(case.m)]
        bayes_vec = bayes.posterior_feature_vector(max_enemies=7, normalize=True)
        obs = encode_obs(case, st, turn, scores, bayes_vec)
        if use_vecnorm and isinstance(vec_state, dict):
            obs = normalize_obs_with_state(obs, vec_state)
        obs_t = torch.as_tensor(obs[None, :], dtype=torch.float32, device=device)

        mask_t = None
        mask_np = None
        if args.use_action_mask:
            mask_np = np.zeros((100,), dtype=np.bool_)
            for x_c, y_c in candidates_all[0]:
                mask_np[x_c * 10 + y_c] = True
            mask_t = torch.as_tensor(mask_np[None, :], dtype=torch.bool, device=device)

        with torch.no_grad():
            act_t = agent.act(obs_t, action_mask=mask_t, deterministic=args.deterministic)
        action = int(act_t.detach().cpu().numpy().reshape(-1)[0])
        x, y = decode_action(action)

        if mask_np is not None and not mask_np[action]:
            cands = candidates_all[0]
            if cands:
                x, y = cands[0]
            else:
                x, y = st.px[0], st.py[0]

        sys.stdout.write(f"{x} {y}\n")
        sys.stdout.flush()

        # Read tester feedback for this turn.
        # 1) selected moves of all players
        selected_moves: list[tuple[int, int]] = []
        for _ in range(cfg.m):
            sx = reader.next_int()
            sy = reader.next_int()
            selected_moves.append((sx, sy))

        # Update bayes from turn-start state and observed selected destinations.
        bayes.update(
            values=case.values,
            owner_before=st.owner,
            level_before=st.level,
            observed_selected=selected_moves,
            observed_candidates=candidates_all,
        )

        # 2) current piece positions
        px = [0] * cfg.m
        py = [0] * cfg.m
        for p in range(cfg.m):
            px[p] = reader.next_int()
            py[p] = reader.next_int()
        # 3) owner grid
        owner = np.zeros((cfg.n, cfg.n), dtype=np.int16)
        for i in range(cfg.n):
            for j in range(cfg.n):
                owner[i, j] = reader.next_int()
        # 4) level grid
        level = np.zeros((cfg.n, cfg.n), dtype=np.int16)
        for i in range(cfg.n):
            for j in range(cfg.n):
                level[i, j] = reader.next_int()

        st = RuntimeState(owner=owner, level=level, px=px, py=py)
        if args.debug_stderr:
            obj = AHC061LocalEnv._objective(AHC061LocalEnv._score_all(st, case))
            print(f"[turn={turn}] action=({x},{y}) obj={obj:.6f}", file=sys.stderr, flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
