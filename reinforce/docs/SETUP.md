# Setup (pyenv + uv / Docker)

このプロジェクトは `reinforce` をトップレベル名前空間として実行します。
実行例: `python -m reinforce.ppo.entrypoints.train_ppo`

## 1. ローカル環境 (pyenv + uv)

前提:
- `pyenv`
- `uv` (`https://docs.astral.sh/uv/`)

```bash
# repository root で実行
cd /path/to/AHC061

# .python-version は 3.11.4
pyenv install -s 3.11.4
pyenv local 3.11.4

# 仮想環境作成 + 依存同期
uv venv
uv sync --group dev

# MLflow を使う場合のみ
uv sync --group mlflow

# テスト
uv run pytest -q reinforce/tests
```

## 2. requirements.txt ベースで入れる場合

```bash
cd /path/to/AHC061
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r reinforce/requirements/dev.txt
```

MLflow を使う場合:

```bash
pip install -r reinforce/requirements/docker.txt
```

## 3. Docker

```bash
cd /path/to/AHC061
docker compose -f reinforce/docker/docker-compose.yml build
docker compose -f reinforce/docker/docker-compose.yml run --rm reinforce
```

任意コマンド実行例:

```bash
docker compose -f reinforce/docker/docker-compose.yml run --rm \
  reinforce python -m reinforce.ppo.entrypoints.run_pipeline --help
```

## 4. エントリポイント

- `python -m reinforce.ppo.entrypoints.run_pipeline`
- `python -m reinforce.ppo.entrypoints.train_ppo`
- `python -m reinforce.ppo.entrypoints.train_bc`
- `python -m reinforce.ppo.entrypoints.eval_policy`
