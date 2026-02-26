# AHC061 PPO Throughput Bench (2026-02-25)

## Setup
- Environment: `AHC061Local-v0`
- Backend: `bayes_backend=cpp`, `bayes_num_particles=128`
- Model: `model.ahc061_student_m.toml`, init from strong checkpoint (`1771964092980`)
- Device: CUDA
- Common args:
  - `total_timesteps=131072`
  - `vector_env=async`
  - `use_action_mask=true`
  - `vecnorm=false`
  - `eval disabled` (`eval_interval_steps=0`, `eval_at_start=false`)

## Results

| case | num_envs | num_steps | num_minibatches | update_epochs | SPS | elapsed(s) | GPU util mean(%) | GPU mem mean(MB) | CPU busy mean(%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bench_e16_s128_m8_u4 | 16 | 128 | 8 | 4 | 2145 | 64 | 14.53 | 1500.22 | 21.18 |
| bench_e24_s128_m12_u4 | 24 | 128 | 12 | 4 | 2550 | 54 | 18.16 | 1516.67 | 23.66 |
| bench_e32_s128_m16_u4 | 32 | 128 | 16 | 4 | 2720 | 51 | 17.86 | 1509.57 | 25.52 |
| bench_e16_s256_m16_u4 | 16 | 256 | 16 | 4 | 2119 | 64 | 17.22 | 1510.43 | 20.03 |
| bench_e32_s128_m16_u3 | 32 | 128 | 16 | 3 | 2986 | 47 | 16.38 | 1501.72 | 26.71 |
| bench_e40_s128_m20_u4 | 40 | 128 | 20 | 4 | 2909 | 48 | 19.12 | 1507.22 | 26.08 |

## Interpretation
- Throughput-optimal observed setting: `num_envs=32`, `num_steps=128`, `num_minibatches=16`, `update_epochs=3`.
- GPU utilization stayed low (~15-19%) while CPU busy was moderate (~20-27%).
- This indicates the primary bottleneck is not GPU compute; rollout/env side (including case generation + Bayes update) dominates.
- Increasing `num_steps` from 128 to 256 did not help throughput on this setup.
- Increasing envs beyond 32 to 40 gave only marginal gain vs 32/u3 and is less stable for run-to-run CPU load.

### After candidate-sharing optimization (env step + Bayes update)
- Change: compute candidates once per player/turn in env step, reuse for
  - legality check
  - enemy move decision
  - Bayes likelihood update (`observed_candidates`)
- Same benchmark condition (`e32,s128,m16,u3`) improved:
  - before: `sps=2986`
  - after:  `sps=3068~3091`
  - gain: `+2.7%~+3.5%`

### Re-run (2026-02-25, same condition `e32,s128,m16,u3`)

| run | SPS | elapsed(s) |
|---|---:|---:|
| bench_e32_s128_m16_u3_r1 | 3017 | 46.20 |
| bench_e32_s128_m16_u3_r2 | 2944 | 47.31 |
| bench_e32_s128_m16_u3_r3 | 3005 | 46.38 |

- Mean SPS: `2988.67`
- Stddev: `31.96` (about `1.07%` of mean)
- Min/Max: `2944 / 3017`
- Note: in this rerun, throughput is effectively flat vs baseline (`2986`), and lower than the previously observed `3068~3091`. This suggests run-to-run/system-load sensitivity; repeated measurement is necessary when deciding final throughput tuning.

### After observation/scoring vectorization (2026-02-25)
- Change:
  - `_encode_obs` vectorized with NumPy mask/assign operations.
  - `_score_all` vectorized with `np.bincount`.
- Functional equivalence:
  - matched old logic on 300-step random rollout (`allclose`).

#### Micro benchmark
- `_encode_obs`: `189.11us -> 31.08us` (about `6.09x`)
- `_score_all`: `16.61us -> 8.96us` (about `1.85x`)

#### End-to-end PPO throughput re-run (`e32,s128,m16,u3`)

| run | SPS | elapsed(s) |
|---|---:|---:|
| bench_e32_s128_m16_u3_obsvec_r1 | 3119 | 45.10 |
| bench_e32_s128_m16_u3_obsvec_r2 | 3250 | 43.14 |
| bench_e32_s128_m16_u3_obsvec_r3 | 3116 | 45.23 |

- Mean SPS: `3161.67`
- Stddev: `62.47`
- Min/Max: `3116 / 3250`
- Compared to previous re-run mean `2988.67`, gain is about `+5.79%`.

## Recommended configs
- Fast tuning:
  - `num_envs=32`, `num_steps=128`, `num_minibatches=16`, `update_epochs=3`
- Stability-priority fine-tune:
  - `num_envs=32`, `num_steps=128`, `num_minibatches=16`, `update_epochs=4`
