"""`log_utils` に関するユーティリティ。"""
from __future__ import annotations

import os
import logging
import sys


_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_YELLOW = "\033[33m"
_RESET = "\033[0m"


class _WarningYellowFormatter(logging.Formatter):
    """`_WarningYellowFormatter` を表すクラス。"""

    def format(self, record: logging.LogRecord) -> str:
        """`format` を実行する。

        Args:
            record (logging.LogRecord): record の値。

        Returns:
            str: 計算結果。
        """
        rendered = super().format(record)
        if int(record.levelno) >= int(logging.WARNING):
            return f"{_YELLOW}{rendered}{_RESET}"
        return rendered


def _use_color() -> bool:
    """内部ヘルパー: `use_color` を実行する。

    Returns:
        bool: 計算結果。
    """
    if str(os.environ.get("NO_COLOR", "")).strip():
        return False
    if str(os.environ.get("FORCE_COLOR", "")).strip():
        return True
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False


def get_logger(name: str) -> logging.Logger:
    """`logger`を取得する。

    Args:
        name (str): name の値。

    Returns:
        logging.Logger: 計算結果。
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(stream=sys.stdout)
    if _use_color():
        handler.setFormatter(_WarningYellowFormatter(_FMT))
    else:
        handler.setFormatter(logging.Formatter(_FMT))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
