from __future__ import annotations

from dataclasses import dataclass

from .cpp_ext import load_ext


@dataclass(frozen=True)
class FeatureSpec:
    feature_id: str
    channels: int
    submit_supported: bool


def list_feature_specs(*, verbose_build: bool = False) -> list[FeatureSpec]:
    ext = load_ext(verbose=bool(verbose_build))
    ids = [str(x) for x in ext.feature_ids()]
    out: list[FeatureSpec] = []
    for fid in ids:
        out.append(
            FeatureSpec(
                feature_id=fid,
                channels=int(ext.feature_channels(fid)),
                submit_supported=bool(ext.feature_submit_supported(fid)),
            )
        )
    out.sort(key=lambda x: x.feature_id)
    return out


def get_feature_spec(feature_id: str, *, verbose_build: bool = False) -> FeatureSpec:
    key = str(feature_id).strip()
    if not key:
        raise ValueError("feature_id must not be empty")
    for fs in list_feature_specs(verbose_build=bool(verbose_build)):
        if fs.feature_id == key:
            return fs
    names = ", ".join(fs.feature_id for fs in list_feature_specs(verbose_build=bool(verbose_build)))
    raise ValueError(f"unknown native feature_id={feature_id!r}; available={names}")

