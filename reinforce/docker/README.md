# Docker

```bash
docker compose -f reinforce/docker/docker-compose.yml build
docker compose -f reinforce/docker/docker-compose.yml run --rm reinforce
```

任意コマンド:

```bash
docker compose -f reinforce/docker/docker-compose.yml run --rm \
  reinforce python -m reinforce.ppo.entrypoints.run_pipeline --help
```
