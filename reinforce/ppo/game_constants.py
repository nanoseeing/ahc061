"""`game_constants` に関するモジュール。"""
from __future__ import annotations

BOARD_SIZE: int = 10
"""Grid side length (N)."""

MAX_PLAYERS: int = 8
"""Total player count including self (M_MAX)."""

OPP_SLOT_COUNT: int = 7
"""Number of enemy slots (MAX_PLAYERS - 1)."""

OPP_PARAM_DIM: int = 5
"""Parameters per enemy slot (wa, wb, wc, wd, eps)."""

AUX_OPP_PARAM_TOTAL: int = OPP_SLOT_COUNT * OPP_PARAM_DIM
"""Total auxiliary opponent-parameter outputs (7 * 5 = 35)."""

BAYES_PARAM_DIM: int = OPP_PARAM_DIM * OPP_SLOT_COUNT
"""Flat Bayes posterior parameter dimension (5 params × 7 enemies = 35)."""

BAYES_TAIL_SHAPE: tuple[int, ...] = (BAYES_PARAM_DIM,)
"""Shape of per-step Bayes parameter vector, excluding batch dimension."""

OPP_PARAM_TRUE_TAIL_SHAPE: tuple[int, ...] = (OPP_SLOT_COUNT, OPP_PARAM_DIM)
"""Shape of opp_param_true aux array (excluding batch dim): (7, 5)."""

OPP_VALID_TAIL_SHAPE: tuple[int, ...] = (OPP_SLOT_COUNT,)
"""Shape of opp_valid aux array (excluding batch dim): (7,)."""
