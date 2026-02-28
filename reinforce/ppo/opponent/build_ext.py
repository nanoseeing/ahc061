from __future__ import annotations

import sysconfig
from pathlib import Path

from setuptools import Distribution, Extension
from setuptools.command.build_ext import build_ext


def _module_candidates(pkg_dir: Path) -> list[Path]:
    suffix = str(sysconfig.get_config_var("EXT_SUFFIX") or "").strip()
    cands: list[Path] = []
    if suffix:
        cands.extend(sorted(pkg_dir.glob(f"_opponent_bayes_cpp*{suffix}")))
    cands.extend(sorted(pkg_dir.glob("_opponent_bayes_cpp*.so")))
    cands.extend(sorted(pkg_dir.glob("_opponent_bayes_cpp*.pyd")))
    uniq: list[Path] = []
    seen: set[str] = set()
    for p in cands:
        k = str(p.resolve())
        if k in seen:
            continue
        seen.add(k)
        uniq.append(p)
    return uniq


def build_extension(*, force: bool = False, verbose: bool = False) -> Path:
    pkg_dir = Path(__file__).resolve().parent
    # opponent/ -> ppo/ -> reinforce/ -> ppo_cpp_ext/src/
    reinforce_dir = Path(__file__).resolve().parents[2]
    src = reinforce_dir / "ppo_cpp_ext" / "src" / "opponent_bayes_ext.cpp"
    if not src.exists():
        raise FileNotFoundError(f"cpp source not found: {src}")

    if not force:
        existing = _module_candidates(pkg_dir)
        if existing:
            return existing[0]

    try:
        import pybind11
    except ImportError as e:
        raise RuntimeError("pybind11 is required to build opponent_bayes_cpp extension") from e

    ext = Extension(
        "reinforce.ppo.opponent._opponent_bayes_cpp",
        sources=[str(src)],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"],
    )

    dist = Distribution(
        {
            "name": "reinforce_ppo_cpp",
            "version": "0.0.0",
            "ext_modules": [ext],
        }
    )
    cmd = build_ext(dist)
    cmd.inplace = True
    cmd.force = bool(force)
    if verbose:
        cmd.verbose = 2
    cmd.ensure_finalized()
    cmd.run()

    built = _module_candidates(pkg_dir)
    if not built:
        raise FileNotFoundError("build finished but _opponent_bayes_cpp artifact was not found")
    return built[0]


if __name__ == "__main__":
    out = build_extension(force=False, verbose=True)
    print(out)
