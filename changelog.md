# Changelog: `littlebit-mps` branch

Goal: run LittleBit-2 (Joint-ITQ init → QAT with knowledge distillation → `eval.py`) on an
Apple M1 MacBook Air with 8 GB unified RAM, with as few code changes as possible.
The CUDA / DeepSpeed path behaves the same as on `main`.

## 2026-11-25: MPS support (first session)

### Repository setup
- Branch `littlebit-mps` was created from `main` (`0ceadcf`) and pushed to the fork `origin` (`andreagaffuririva-ctrl/LittleBit`).
- Safety against accidental upstream PRs:
  - `gh repo set-default` points to the fork.
  - `git remote set-url --push upstream DISABLED`: pushes to SamsungLabs fail.
  - Removed `branch.littlebit-mps.github-pr-base-branch` (VS Code had set it to SamsungLabs).
  - The GitHub "Create PR" link printed after `git push` targets upstream by default. Use `gh pr create` or change the base repo by hand.

### Environment
- System Python is 3.15 (too new for torch). A `uv` venv with Python 3.12 is in `.venv/` (ignored by git).
- Installed: `torch==2.8.0` (MPS works), `transformers==4.51.3`, `datasets==3.4.1`, `lm-eval==0.4.8`, `accelerate`, `sentencepiece`, `tqdm`, `safetensors`, `huggingface_hub`, `numpy==2.2.4`.
- Not installed (not needed on this path): `deepspeed`, `bitsandbytes`, `mpi4py`.

### Code changes
| File | Change |
|---|---|
| `main.py` | `import deepspeed` is now lazy: it is imported only in the DeepSpeed teacher path and in the ZeRO-3 branch of `save_artifacts`. |
| `main.py` | `get_device_config()` returns `(1, "mps")` when there is no CUDA and MPS is available. |
| `main.py` | `load_teacher_model()`: without CUDA and without `--ds_config_path`, the teacher is a plain PyTorch model (frozen, eval mode) on the training device. DeepSpeed is not used. |
| `main.py` | Model dtype is bf16 on CUDA (as before) and fp32 elsewhere. `TrainingArguments(bf16=...)` follows CUDA availability. |
| `main.py` | New args: `--max_steps` (default -1, which keeps epoch-based training), `--train_seqlen` (default 2048, as before). `--report` also accepts `none`. |
| `utils/datautils.py` | `--train_seqlen` is passed through `get_qat_dataset` to the existing `get_*_train(seqlen=...)` functions. A non-default seqlen gets its own cache directory (`<hash>_seq<N>`). The default cache path is unchanged. |
| `quantization/utils/quant_util.py` | `load_quantized_model(device="auto")` picks `cuda`, then `mps`, then `cpu`. |
| `eval.py` | Moves the model to `mps` when there is no CUDA (needed for `--fp_eval`). |
| `.gitignore` | Ignores the `wikitext2/` and `c4/` dataset caches (as `c4_wiki` already was). |

LittleBit-2 itself needed no change. The rotation (`_compute_itq_rotation` in
`quantization/modules/littlebit.py`) already exists and is enabled with `--use_itq True`.
SVD and ITQ run on CPU on Mac (the existing fallback), which is fast for these matrix sizes.

### Experiments so far (model `HuggingFaceTB/SmolLM2-135M`)
1. **Smoke train at 1.0 bpw, seqlen 1024**: stopped by hand. Python used 6.6 GB and swap was full
   (Chrome and VS Code held about 4 GB). The memory estimate was too low, because the binarized fp32 copies
   (`Vq`, `Uq`) saved for backward add to the params, grads and Adam states.
2. **Smoke train at 0.1 bpw, seqlen 512, 2 steps**: finished OK on MPS. The loss was finite (very large at init, as expected),
   with no CPU-fallback ops. It took 30 to 77 s/step because swap was already under heavy pressure. Output: `outputs/smoke/`
   (local only, ignored by git), with `littlebit_config.json` showing `use_itq: true`.
3. **Smoke eval** (`eval.py --model_id outputs/smoke --seqlen 512 --limit 20`): the packed checkpoint loaded,
   was binarized and moved to MPS. Results: wikitext2 PPL ≈ 1.96e6, piqa acc 0.50 (20 samples). That is garbage, as expected after 2 steps.

### Finding: real bit budget on a small model
For a hidden size of 576, the fixed cost of the fp16 scale vectors is larger than 0.1 bpw. Every layer falls to
`min_split_dim = 8`. With `--residual True`:

| `--eff_bit` | real bpw | rank q/o, kv, MLP | latent params |
|---|---|---|---|
| 0.1 | 0.139 | 8, 8, 8 | 4.9M |
| 0.3 | 0.275 | 24, 8, 40 | 19.3M |
| 0.55 | 0.526 | 56, 16, 96 | 45.7M |
| 0.7 | 0.688 | 80, 32, 128 | 62.7M |
| 1.0 | 0.986 | 120, 48, 192 | 94.0M |

## 2026-11-25: session 2 (run from phone via Remote Control)

### Fixes
- **Bug (session 1): `--train_seqlen` had no effect.** `get_wikitext2_train`, `get_c4_train` and `get_c4_wiki_train`
  accepted `seqlen` but hardcoded `block_size = 2048`. They now use `block_size = seqlen` (the default 2048 is unchanged).
  All session 1 runs and the first probe trained on 2048-token sequences, which explains the memory blow-up and the slow steps.
  The mislabeled caches `wikitext2/d98c9be_seq512` and `_seq1024` were deleted and rebuilt (5015 × 512 tokens).
- `quantization/functions/binary.py`: `y[y == 0] = 1` → `torch.where(y == 0, 1, y)` in `STEBinary` and `SmoothSign`.
  The output is identical (checked on CPU and MPS, including ±0). The masked assignment forces a device sync on MPS (0.71 ms vs 0.17 ms per call).
- `.claude/settings.json`: read-only Bash allowlist (from `/fewer-permission-prompts`).

### Speed (SmolLM2-135M, 0.55 bpw, residual, batch 1 × grad-accum 4, seqlen 512, MPS fp32)
- Before the fix (really 2048 tokens): 106 s/step. After the fix: **11 s/step** (about 2.8 s per 512-token micro-batch).

### Results
| Model | wikitext2 PPL (seqlen 2048) | piqa | arc_e | arc_c | winogrande | boolq | hellaswag |
|---|---|---|---|---|---|---|---|
| SmolLM2-135M FP32 (baseline) | **15.43** | 0.694 | 0.606 | 0.234 | 0.520 | 0.618 | 0.394 |
| LittleBit-2 0.55 bpw, 1 epoch | _pending_ | | | | | | |

Zero-shot: `acc`, 0-shot, `--limit 500` per task (acc_norm: piqa 0.700, arc_e 0.570, arc_c 0.272, hellaswag 0.442).

### Observation
SmolLM2 has very large hidden-state outliers (max |h| ≈ 2e4), so the layer-to-layer MSE term (`l2l_loss`, ×10)
is about 5× larger than the KD term and dominates the loss. This is the same as the paper recipe, kept unchanged.

### Known quirks (not changed)
- Importing `quantization.modules` directly causes a circular import. Import `quantization.utils` first.
- `get_eval_loaders` forces `bos_token_id=1, eos_token_id=2` on the tokenizer (inherited from the Llama code).
- `kd_utils.mse_loss` uses CPU autocast when there is no CUDA. It has no effect on MPS tensors.
