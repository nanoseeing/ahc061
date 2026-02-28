# Submission Export (Single main.cpp)

`StudentMBoardAgent` / `TeacherP0V1BoardAgent` / `Exp002ResNetBoardAgent` の checkpoint から、
重み埋め込み済みの単一 `main.cpp` を生成します。

## Command

```bash
python -m reinforce.ppo.entrypoints.export_submission_main_cpp \
  --checkpoint /path/to/models/last.pt \
  --output reinforce/outputs/submission/main.cpp
```

`make_submit_compact` 互換のエントリ（推奨）:

```bash
python -m reinforce.ppo.entrypoints.make_submit_compact \
  --checkpoint /path/to/models/last.pt \
  --output reinforce/outputs/submission/main.cpp \
  --payload-encoding huff91 \
  --compact-layout \
  --tta-mode 0
```

exp002フル機能（`ppconcat` 量子化プリセット群）を使う場合:

```bash
python -m reinforce.ppo.entrypoints.make_submit_compact \
  --exp002-full \
  --checkpoint /path/to/exp002_ckpt.pt \
  --output reinforce/outputs/submission/main.cpp \
  --payload-encoding base91 \
  --ppconcat-preset c7
```

TorchScript 経路（exp002互換）:

```bash
python -m reinforce.ppo.entrypoints.make_submit_torchscript \
  --ckpt /path/to/exp002_ckpt.pt \
  --out-dir reinforce/outputs/submission/exp002_ts
```

## Notes

- 対応モデル:
  - `model_config.type == "StudentMBoardAgent"` (compact path)
  - `model_config.type == "TeacherP0V1BoardAgent"` (compact path)
  - `model_config.type == "Exp002ResNetBoardAgent"` (compact path)
- 生成コードは policy + 簡易Bayes 推定込みの単一 C++ 実装です。
- `--compact-layout` は exp002 版由来のトークン/マクロ圧縮を適用します（デフォルト有効）。
- `--tta-mode/--tta-k/--tta-auto-off-ms` は `Exp002ResNetBoardAgent` のみ対応です。
- `--exp002-full` を付けると、`make_submit_compact_exp002`（vendored実装）へ委譲し、
  `--ppconcat-*` を含む exp002 由来のフル機能を利用できます。
- `state_dict` の `_orig_mod.` / `module.` 接頭辞は exporter 側で吸収します。
- 目安コンパイル:

```bash
g++ -std=gnu++20 -O2 reinforce/outputs/submission/main.cpp -o /tmp/submission_main
```
