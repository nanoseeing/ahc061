FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_SYSTEM_PYTHON=1 \
    PYTHONPATH=/workspace/AHC061

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -sf /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /workspace/AHC061

COPY pyproject.toml ./
COPY reinforce/requirements ./reinforce/requirements
RUN uv pip install --system --index-url https://download.pytorch.org/whl/cpu "torch>=2.4,<2.7"
RUN uv pip install --system -r reinforce/requirements/docker.txt

COPY . .
RUN chmod +x reinforce/docker/entrypoint.sh

ENTRYPOINT ["bash", "reinforce/docker/entrypoint.sh"]
CMD ["python", "-m", "reinforce.ppo.entrypoints.run_pipeline", "--help"]
