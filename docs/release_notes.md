# Camera-ready code update

This update extends the original public repository rather than replacing its history. The author selected the existing `Bluixe/CV-ICRL` repository. Research snapshots are selectively imported; datasets, models, logs, machine-specific controllers, and later multi-env experimentation are excluded.

## Source recovery

`source_manifest.json` records each imported source component's research commit and SHA256 before adaptation, plus its released file hash. Those research commits are not commits in the public repository. The public baseline is `fddde4d`.

| Component | Recovered source | Release treatment |
| --- | --- | --- |
| Original Dark Room/MiniGrid | public `fddde4d` | Retained; fix missing evaluator imports, unused CQL eager imports, default W&B mode, and input-device padding |
| MiniGrid multihead | `28e317d` | Port only this class; retain default logits and state-dict keys |
| Reinformer | `f356f05` | v3 single-pass head and source dataset/train/eval/tests; older heads retained for checkpoint compatibility |
| IC-CQL | `1b6c35b` | Corrected twin-Q implementation; default 15 epochs and fixed epoch-5 wrapper |
| Feedback | `842a924` | Pure intervention/metric helpers, explicit portable evaluator and normal/zero/frozen wrapper |
| PPO-state targets | `afa0ce0` | Extract mapping/critic helpers; replace server-specific history/identity locks with data and target hashes and explicit files |
| Procgen | `bc2eaa1` | Earliest recovered single-task pipeline, before later competence/self-feed controllers; portable inputs/outputs, collector and JSON metrics |
| MetaWorld | `63cb386` working tree | Six experiment/model/dataset files; separate environment, explicit task/output options; source PPO working-file modification recorded |

## Implementation and protocol decisions

- Existing public files and third-party license notices remain. No broad replacement of `dataset.py` or `nets/net.py` is performed.
- MiniGrid portable training uses the original backbone and sum-reduced CE + MSE. It records data hashes and saves model configuration. It is a new-run implementation, not a claim of identical archived sampler order or checkpoint bytes.
- AD retains the original trailing-zero buffer alignment. CV feedback retains the reported E1 leading-zero input plus the model shift. Feedback and FIFO context persist across episode resets.
- Reinformer/IC-CQL wrappers encode H400, training seed0, effective batch32, 15 training epochs, evaluation epochs10/5 respectively, and 20 tasks ×8,000 steps with evaluator seeds20000–20019. CV table-reference seeds0–19 are not relabeled as paired.
- PPO-state targets use the source checkpoint associated with each 50-step block. Training J labels verify the 40-checkpoint/17-row mapping. Validation collector order remains an input prerequisite; reward values cannot authenticate its source identity. Train and validation use training-corpus normalization moments and hash-bound sidecars. Collector score text accepts both commas and newlines.
- H200 requires its own 200-step corpus/model. Existing H400 table references are retained, not asserted to be a matched retraining.
- Procgen preserves its reward/value channels and per-episode feedback reset. It does not use the MiniGrid running-max deployment rule. `--start_level` now controls the default10000–10019 range. Multi-env requests are rejected. Other retained legacy ablation switches are not covered by the release smoke.
- MetaWorld compatibility mode retains the recovered continuous-action loss and feedback evaluator. New collection can store raw ICL states while PPO consumes normalized inputs (`--raw-icl-observations`). New training can teacher-force J into the reward channel (`--value-token-input`). Both are explicit input-protocol adaptations and do not reinterpret archived checkpoint results. The legacy normalized/raw mismatch and reward/J channel mismatch are documented in README.
- Per-task completed-episode JSON metrics use equal task weight and population SD. Missing-episode conventions differ in legacy baseline and dedicated feedback exporters; compare paper runs only after confirming all20 tasks completed episodes. MetaWorld exports preserve completed-episode means and zero for empty tasks.

## Validation

Validation is conducted in isolated staged release directories with CPU-only synthetic runs. Original research repositories, running training jobs, and datasets are not modified.

- Unit tests cover model parameter keys/default fixed-input logits against the public baseline, episode-local RTG and masks, train/validation scales, predecessor alignment, twin-Q TD/CQL and target updates, feedback/FIFO behavior, equal-task statistics, source-critic checkpoint loading, and hash-bound PPO-state sidecars through a backward pass.
- The core smoke exercises17 public CLI help commands and14 train/eval commands: tiny AD/CV/Reinformer/IC-CQL training, save/load, real short MiniGrid feedback rollouts, and single-task Procgen AD/CV training plus real short Procgen rollouts.
- The optional MetaWorld smoke checks CLI imports, observation storage without changing PPO normalization, scalar channel selection, tiny legacy AD/auto-relabel and explicit value-token training, save/load, and short reach/push rollouts.
- These checks validate execution and information flow. Their synthetic scores are not paper results.

Results on 2026-10-08:

| Check | Result | Runtime |
| --- | --- | --- |
| Core unit tests (`python -m pytest -q tests`) | **62 passed**, no skips, 6.17 seconds | Python 3.9.15 / PyTorch 2.6.0 / NumPy 1.23.5 / Transformers 4.50.3; vendored SB3 |
| Core smoke (`python tests/smoke_release.py --output-dir outputs/smoke --procgen`) | **31 commands passed** | Same core environment; Procgen 0.10.7+5e1dbf3 |
| MetaWorld smoke (`python tests/smoke_metaworld.py --output-dir outputs/meta-smoke`) | **13 commands passed**, including legacy and explicit value-token inputs | Python 3.10.0 / PyTorch 2.11.0 / NumPy 2.2.6 / Transformers 4.40.0 / MetaWorld 3.0.0 / MuJoCo 3.6.0 / installed SB3 2.8.0 |

The core regression test was supplied an exported copy of public baseline `fddde4d` in the staged non-Git directory. A normal clone resolves that baseline from Git history. CPU-only smoke settings include disabled W&B logging and one OpenMP thread. The original source directories are not imported by these smoke commands.

Fresh environment installation, large data collection, full training, 8,000-step score reproduction, and original-curve provenance are not certified by these checks.

## Remaining release inputs

- Author choice of top-level project license; no license is inferred from vendored third-party code.
- A separately published dataset/checkpoint bundle and verified download instructions.
- Exact provenance of older Procgen curves and archived MetaWorld push results; recovered source and successful short execution alone do not establish those identities.
- Fresh installation and full reproduction checks if an end-to-end score-reproduction release is desired.
