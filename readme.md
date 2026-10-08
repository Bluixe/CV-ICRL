# Context Value Informed In-Context Reinforcement Learning (CV-ICRL)

Research code for CV-ICRL on Dark Room, MiniGrid, and Procgen. The camera-ready update adds the Reinformer and twin-Q IC-CQL comparisons, feedback interventions, PPO-state target construction, and portable training/evaluation commands. It retains the original public implementation and vendored environments rather than copying the full research working tree.

## Install

Use Python **3.9**. The exercised Linux environment uses Python 3.9.15, PyTorch 2.6.0, NumPy 1.23.5, Transformers 4.50.3, Gymnasium 1.1.1, and the vendored packages below. The previous PyTorch 1.13 recipe is incompatible with the vendored SB3, which requires PyTorch >=2.3.

```bash
conda create -n cv-icrl python=3.9.15 -y
conda activate cv-icrl
# CPU; use the corresponding PyTorch CUDA wheel for GPU training.
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -r requirements-test.txt
pip install -e ./Minigrid -e ./stable-baselines3 -e ./stable-baselines3-contrib
```

The vendored SB3 reports `2.6.1a0`; use the vendored source for these commands, including the project's PPO modifications. We verified execution in an existing Linux environment with these source directories on `PYTHONPATH`; a fresh environment installation and full paper retraining have not been validated.

W&B logging defaults to **disabled** for the ICL release entry points. Enable it with `WANDB_MODE=online` or the baseline `--wandb-mode online` flag after configuring your own account. Original MiniGrid/Dark Room PPO trainers retain their historical logging behavior; set `WANDB_MODE=disabled` when running them without an account. No credentials, datasets, checkpoints, or experiment logs are included in Git.

### Linux Procgen extra

Procgen requires its native simulator; it is optional for MiniGrid/Dark Room. The tested server package is `procgen 0.10.7+5e1dbf3`. Install the official Procgen release for your Linux/Python platform:

```bash
pip install procgen==0.10.7
```

The standard PyPI wheel is not asserted to be bitwise identical to that server build. Procgen installation on macOS has not been tested. Headless smoke runs use `SDL_VIDEODRIVER=dummy`.

## Quick verification

From the repository root:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m pytest -q tests
python tests/smoke_release.py --output-dir outputs/smoke
# With the Linux Procgen extra installed:
python tests/smoke_release.py --output-dir outputs/smoke-procgen --procgen
```

The smoke command uses tiny synthetic histories, trains for one batch, saves/reloads checkpoints, and runs short real MiniGrid/Procgen rollouts. Its returns are **not paper results**. The tests cover causal alignment, episode boundaries, feedback write-back, target moments and data identity, twin-Q loss, Reinformer RTG masks, and equal-task aggregation. The core-regression test compares against commit `fddde4d`; use a Git clone retaining that public commit.

## Data and checkpoint inputs

The code release does not bundle trained weights or the large trajectory corpus. Existing data must be copied into the paths below or supplied explicitly with `--train-data`, `--val-data`, and `--checkpoint`. Dataset pickles and checkpoints are intended to be loaded from trusted sources.

| Data | Required fields / shape | Default location |
| --- | --- | --- |
| MiniGrid original histories | `observations [N,H,3,7,7]`, `actions [N,H]`, `rewards [N,H]`, `dones [N,H]` | `datasets/MiniGrid/ENV/train_traj-more.pkl`, `test_traj.pkl` |
| MiniGrid CV histories | same observations/actions; `rewards` replaced by evaluated source-checkpoint return J | `datasets/MiniGrid/ENV/train_traj-relabel.pkl` |
| Procgen AD | `.npz` with observations, actions, rewards, dones | `datasets/Procgen/ENV/{train,test}_traj-official.npz` |
| Procgen CV | same fields plus checkpoint-return `values` | `datasets/Procgen/ENV/{train,test}_traj-relabel-new.npz` |

The added baseline datasets require **17 adjacent rows per stream** for the archived 400-step MiniGrid corpus. Do not use 17 for a different collector layout. Episode-local Reinformer RTG uses gamma=1, masks incomplete suffixes, and reuses a fixed training-corpus scale in validation.

Original data collection entry points remain `train_minigrid_ppo.py`, `collect_minigrid_data.py`, `train_darkroom_ppo.py`, and `collect_darkroom_data.py`. Run their `--help` for source-policy and collection options. The corrected `collect_minigrid_data.sh` targets MiniGrid rather than the unrelated Dark Room command previously in that file.

## MiniGrid AD and CV-ICRL

The compact release trainer uses the existing CNN/GPT-2 backbone and sum-reduced action CE + scalar MSE. It takes explicit data and output paths, keeps checkpoint parameter names compatible, and records input hashes. These portable entry points support new runs; they do not claim to recreate archived sampler order or checkpoint bytes.

```bash
ENV=MiniGrid-SimpleCrossingS9N3-v0
python train_cv_minigrid.py --env "$ENV" --method cv \
  --train-data "datasets/MiniGrid/$ENV/train_traj-relabel.pkl" \
  --val-data "datasets/MiniGrid/$ENV/test_traj.pkl" \
  --output-dir "outputs/$ENV/cv" --H 400 --embd 256 --layer 4 --head 4 \
  --num-epochs 15 --batch-size 32 --lr 0.001 --seed 0
python eval_cv_minigrid.py --checkpoint "outputs/$ENV/cv/epoch_15.pt" \
  --eval-env "$ENV" --method cv --output-json "outputs/$ENV/cv/eval.json" \
  --num-envs 20 --num-steps 8000 --seed-start 0
```

For AD, use `--method ad` and the original `train_traj-more.pkl` rewards. Its history uses the original trailing-zero action/reward alignment. The CV feedback evaluator preserves the reported intervention suite's **leading zero followed by the model's causal shift**; it retains FIFO context and running-max feedback across episode resets. These alignment conventions are deliberately explicit. The original `train_icl_minigrid.py`, `eval_icl_minigrid.py`, and Dark Room entry points are retained for historical workflows.

### Feedback interventions

```bash
bash scripts/run_minigrid_feedback.sh CHECKPOINT ENV outputs/feedback
```

This evaluates normal, zero, and frozen write-back with the same model and task seeds. Normal writes `max(0, previous_max, prediction)`. Frozen fixes the first written value, and zero writes zero. The intervention modifies subsequent context, so trajectories can diverge. This evaluates behavioral reliance on feedback, not value calibration.

### Reinformer and IC-CQL

For each reported task type (`SimpleCrossingS9N3`, `SimpleCrossingS11N5`, `LavaCrossingS9N3`, `BlockedUnlockPickup`), use the full `MiniGrid-...-v0` identifier:

```bash
bash scripts/run_minigrid_baseline.sh reinformer ENV datasets/MiniGrid/ENV outputs/ENV/reinformer
bash scripts/run_minigrid_baseline.sh ic_cql ENV datasets/MiniGrid/ENV outputs/ENV/ic_cql
```

The wrapper fixes H400, a 4-layer/4-head/256-dimensional Transformer, training seed 0, 15 epochs, and effective batch size 32. Reinformer uses the **single-pass linear v3** action head and evaluates epoch **10**. IC-CQL uses **twin Q**, the paper tuple, CQL weight 0.01, label smoothing 0.3, and evaluates epoch **5**, with greedy Q1 actions. Both evaluate 8,000 steps on 20 tasks with seeds **20000–20019**. The retained CV table references use **0–19**; those comparisons are not seed-paired. Earlier Reinformer head variants are retained only for checkpoint compatibility.

### PPO-state targets

```bash
python build_ppo_state_targets.py \
  --train-data datasets/MiniGrid/ENV/train_traj-relabel.pkl \
  --val-data datasets/MiniGrid/ENV/test_traj.pkl \
  --eval-results PPO_CHECKPOINT_DIR/eval_results.txt --checkpoint-dir PPO_CHECKPOINT_DIR \
  --output-dir outputs/ENV/ppo_targets --device cuda
python train_cv_minigrid.py --env ENV \
  --train-data datasets/MiniGrid/ENV/train_traj-relabel.pkl \
  --val-data datasets/MiniGrid/ENV/test_traj.pkl \
  --target-npy outputs/ENV/ppo_targets/train_ppo_state.npy \
  --val-target-npy outputs/ENV/ppo_targets/val_ppo_state.npy \
  --output-dir outputs/ENV/ppo_state --num-epochs 10
```

The builder reconstructs the archived 40-checkpoint selection and 17-row/eight-block mapping from sorted `ppo_<steps>_steps.zip` files and evaluated returns. It verifies every training J label against that mapping, queries each block's **source PPO critic**, and constructs `mu_J + sigma_J * (V_PPO(s) - mu_V) / sigma_V`. Both splits use **training moments**. A manifest binds each sidecar to its dataset and target hashes. Training J labels verify the source mapping. Validation rows are required to retain the same archived collector order; their environment rewards cannot authenticate checkpoint identity. A different collection layout requires an explicit mapping adaptation. It does not infer checkpoint identity from a plausible value shape.

For H200, prepare the separately collected 200-step corpus and train a model with `--H 200`; changing evaluation context alone is not the reported H200 comparison. The manuscript's H400 references are retained main-table results and should not be labeled a matched H200 retraining.

## Single-task Procgen

The recovered Procgen source supports the paper's single-task pipeline for Bigfish, Starpilot, and Miner, with an IMPALA-style encoder and a separate reward/value input. It excludes the subsequent scheduled competence, held-out multi-env, posthoc critic, and experiment-controller work.

```bash
# Train source PPO, then collect original and checkpoint-return histories.
python train_procgen_ppo_official.py --env_name bigfish --save_path models/Procgen/bigfish/official
python collect_procgen_official.py --env bigfish --train --save_path models/Procgen/bigfish/official
python collect_procgen_official.py --env bigfish --save_path models/Procgen/bigfish/official
python collect_procgen_official.py --env bigfish --train --relabel_reward --save_path models/Procgen/bigfish/official
python collect_procgen_official.py --env bigfish --relabel_reward --save_path models/Procgen/bigfish/official
# Example new CV run; select a training budget explicitly for your experiment.
python train_icl_procgen.py --env bigfish --auto_relabel --H 400 --embd 256 --layer 4 --head 4 \
  --num_epochs 10 --sample_stride 4 --output-dir outputs/bigfish/cv
python eval_icl_procgen.py --env bigfish --eval_env bigfish --auto_relabel \
  --checkpoint outputs/bigfish/cv/epoch10_auto_relabel.pt --output-json outputs/bigfish/cv/eval.json \
  --H 400 --embd 256 --layer 4 --head 4 --num_envs 20 --num_steps 8000 --start_level 10000
```

Omit `--auto_relabel` for AD. Explicit `--train-data`/`--val-data`, `--batch-size`, and `--num-workers` are supported. Multi-env launch requests are rejected in this release. Other legacy ablation options are retained but have not been validated by this release's smoke. The recovered sampling stride is 4; this source recovery alone does not authenticate the complete provenance of every original Procgen curve. The default 50M-step PPO run and full collection are expensive and are not part of the smoke command.

Procgen CV preserves the source implementation's per-episode feedback reset and cross-episode context retention; it does **not** inherit MiniGrid E1's running-max rule. Evaluation defaults to levels 10000–10019, and the formerly ignored `--start_level` now controls that range.

## Metrics, sources, and remaining release inputs

MiniGrid baseline and feedback JSON exports include per-task completed episode returns. AER is the mean completed-episode return within each task, LER is its last completed return, and IF counts adjacent drops to <=95% of the preceding return divided by the number of completed episodes. Aggregate means give tasks equal weight; reported deviations are population SD across tasks, not training-seed uncertainty. The dedicated feedback/Procgen JSON utility marks tasks without completed episodes as missing; baseline compatibility exports retain their historical zero convention. Check that all **20 tasks have completed episodes** before comparing paper scores. Partial final episodes are excluded from these JSON metrics; legacy console totals may include them.

See [source_manifest.json](docs/source_manifest.json) for recovered component commits and source hashes, and [release_notes.md](docs/release_notes.md) for adaptations and validation. Third-party licenses remain in their original directories. The project itself previously had no top-level license; a project license is pending the author's choice.

The separate MetaWorld pipeline is described below. The large original datasets, published checkpoint bundle, exact source of older curves, and a fresh-install/full-training check remain separate release inputs. This repository update does not assert that every paper number can already be reproduced from Git alone.


## Optional MetaWorld module

The supplementary continuous-control code was recovered from a separate source repository. Use a **separate Python 3.10 environment**, without the root NumPy pins or vendored SB3 on `PYTHONPATH`:

```bash
conda create -n cv-icrl-metaworld python=3.10 -y
conda activate cv-icrl-metaworld
pip install torch==2.11.0
pip install -r requirements-metaworld.txt
```

Run `python tests/smoke_metaworld.py --output-dir outputs/meta-smoke` to exercise the optional module.

The inspected runtime uses MetaWorld 3.0.0, MuJoCo 3.6.0, NumPy 2.2.6, Gymnasium 1.2.3, Transformers 4.40.0, and SB3 2.8.0. Fresh installation has not been validated. The module adds explicit `--env-name reach-v3|push-v3` and output paths; the recovered source hardcoded reach-v3. This generalization does not authenticate the provenance of the archived push result.

```bash
python train_ppo_ml1_all_goals.py --env-name reach-v3 --output_dir outputs/reach/ppo
python collect_metaworld_data.py --env-name reach-v3 --ckpt_dir outputs/reach/ppo \
  --output datasets/metaworld/reach-v3/train.pkl --mode ad --raw-icl-observations
python collect_metaworld_data.py --env-name reach-v3 --ckpt_dir outputs/reach/ppo \
  --output datasets/metaworld/reach-v3/train_relabel.pkl --mode relabel_reward --raw-icl-observations
python train_icl_metaworld.py --env-name reach-v3 --train_data datasets/metaworld/reach-v3/train_relabel.pkl \
  --algorithm auto_relabel --value-token-input --output-dir outputs/reach/cv --horizon 400 --num_epochs 50
python eval_icl_metaworld.py --env-name reach-v3 --model_dir outputs/reach/cv --epoch final \
  --algorithm auto_relabel --split test --num_tasks 10 --num_steps 5000 --output-json outputs/reach/cv/eval.json
```

For new runs, use `--raw-icl-observations`: PPO continues to consume its goal-specific normalized states, while ICL histories store raw states matching the bare-environment evaluator. The legacy collector stored normalized states, whereas the legacy ICL evaluator used raw states; archived checkpoints from that path require a separate normalization audit and must not be treated as validated reproduction. The default collection behavior is retained for compatibility, and a side manifest records the chosen mode. Use the original reward corpus and `--algorithm ad` for AD; use `--env-name push-v3` consistently for a new push run. MetaWorld retains its original continuous-action loss and evaluator semantics, using completed-episode means and zero for tasks without completed episodes; JSON calls that metric `legacy_aer_per_task`. Do not mix it with MiniGrid's completed-episode metrics. The source PPO trainer had uncommitted server changes; both its base commit and recovered working-file hash are recorded in the manifest. The recovered `auto_relabel` trainer inputs environment rewards and supervises an auxiliary J head, while its evaluator writes predicted J into the reward channel. The explicit `--value-token-input` option used above teacher-forces J into that channel for new runs, retaining the architecture and loss. This option and raw-state collection change the new-run input protocol; they do not reinterpret archived checkpoint results. Full MetaWorld collection/training and archived-score reproduction are not part of this update's checks.
