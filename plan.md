# Plan: LittleBit-2 on Apple M1 (8 GB)

See `changelog.md` for what is already done.

**Status (session 2, 2026-11-25):** `--eff_bit 0.55` was chosen. The FP baseline is done (PPL 15.43). The seqlen bug is fixed (11 s/step).
A 1-epoch QAT run (1254 steps, ~4 h) followed by eval is running in the background via `nohup outputs/run_0p55.sh`:
- progress: `outputs/train_0p55.log`; completion: `outputs/run_0p55.status`; eval results: `outputs/lb2_0p55_eval.log`
- checkpoint: `outputs/smollm2-135m-lb2-0p55/`

Steps 0 to 5 below are done. **Result:** wikitext2 PPL 2588 (FP: 15.43), zero-shot at chance. See `changelog.md`.

## Experiments done (300 steps each unless noted; details in `changelog.md`)
| setting | PPL |
|---|---|
| default (l2l 10, lr 4e-5), full epoch | 2588 |
| l2l 0.1, lr 4e-5 | 2177 |
| l2l 0.1, lr 2e-4 | 1136 |
| l2l 0.1, lr 2e-4, **1.0 bpw** | **1036** |

Best recipe so far: `--l2l_loss_scale 0.1 --lr 2e-4`. KD is still falling at 300 steps, so the training budget is the main limit.

## Next experiments (ranked)
1. **Full epoch with the best recipe** (0.55 bpw, ~3.5 h): `outputs/run_exp.sh lb2_0p55_best --num_train_epochs 1 --l2l_loss_scale 0.1 --lr 2e-4`.
   Use 1.0 bpw instead (`--eff_bit 1.0`, ~4 h) if the goal is the lowest PPL. 0.55 bpw is the more interesting sub-1-bit point.
2. **Higher LR probe:** `--lr 5e-4`, 300 steps (~1 h). Check whether 2e-4 is still too low.
3. **More data:** `--dataset c4_wiki` (one C4 shard) and/or 2 to 3 epochs. Run it overnight.
4. **Ablation:** `--use_itq False` with the best setting, to measure what LittleBit-2 adds.

## Goal
Reproduce the LittleBit-2 evaluation (wikitext2 PPL + zero-shot) on a small model:
1. FP baseline of the original model.
2. LittleBit-2 checkpoint after a short QAT run.
3. Compare the two.

## Decisions taken
- **Model:** `HuggingFaceTB/SmolLM2-135M` (Llama architecture). Qwen2.5-0.5B does not fit: at about 1 bpw,
  its ~358M latent values × 16 B of training state come to about 5.7 GB.
- **Method:** LittleBit-2 only (`--use_itq True`, `--quant_func SmoothSign`, `--residual True`).
- **Precision:** fp32 on MPS (bf16 on M1 is unreliable; fp16 can overflow).
- **Training data:** `wikitext2` (small, cached locally in `wikitext2/<hash>_seq512`).

## Open decision (do this first tomorrow)
Pick the bit budget. 0.1 bpw is meaningless at this size (it becomes 0.139 real bpw, with rank 8 in every layer).
- **Recommended:** `--eff_bit 0.55` (0.53 real bpw, ~46M latents, about half the memory of 1.0).
- Alternative: `--eff_bit 1.0` (0.99 real bpw, ~94M latents). This needs Chrome closed and probably `--train_seqlen 512`.

## Next steps
0. Quit Chrome (and other heavy apps) before any training run. Check swap: `sysctl vm.swapusage`.
1. Activate the environment:
   ```bash
   cd ~/Documents/projects/LLMs/quantization/LittleBit && source .venv/bin/activate
   ```
2. FP baseline (no training, about 0.5 GB):
   ```bash
   python eval.py --model_id HuggingFaceTB/SmolLM2-135M --fp_eval True --seqlen 2048 \
     --ppl_task wikitext2 --zeroshot_task piqa,arc_easy,arc_challenge,winogrande,boolq,hellaswag --limit 500 \
     | tee outputs/fp_eval.log
   ```
3. Timing probe: run the command in step 4 with `--max_steps 10 --f_name probe`. Note the s/step and the peak memory
   (Activity Monitor). Expect a few s/step without swap. If it is above 20 s/step, use `--train_seqlen 256` or a lower `--eff_bit`.
4. Real QAT run (adjust `--max_steps` from the probe; ~200 steps is a first target):
   ```bash
   PYTORCH_ENABLE_MPS_FALLBACK=1 python -u main.py --model_id HuggingFaceTB/SmolLM2-135M \
     --dataset wikitext2 --train_seqlen 512 --save_dir outputs --f_name smollm2-135m-lb2-0p55 \
     --per_device_train_batch_size 1 --gradient_accumulation_steps 4 --max_steps 200 \
     --lr 4e-5 --warmup_ratio 0.02 --report none \
     --quant_func SmoothSign --quant_mod LittleBitLinear --residual True --eff_bit 0.55 \
     --use_itq True --itq_n_iter 50 > outputs/train_0p55.log 2>&1
   ```
5. Evaluate the checkpoint with the same tasks as step 2:
   ```bash
   python eval.py --model_id outputs/smollm2-135m-lb2-0p55 --seqlen 2048 \
     --ppl_task wikitext2 --zeroshot_task piqa,arc_easy,arc_challenge,winogrande,boolq,hellaswag --limit 500 \
     | tee outputs/lb2_0p55_eval.log
   ```
6. Record the FP vs LittleBit-2 results in `changelog.md`, commit, and push to `origin` (the fork only).

## Success criteria
- The QAT loss goes down clearly over the run (`kd_loss` and `l2l_loss` in the log).
- LittleBit-2 PPL is far below the smoke value (1.96e6). It is expected to be worse than the FP baseline.
- Zero-shot accuracies are above chance on at least some tasks.

## If something goes wrong
- **Swap / very slow steps:** close apps, then lower `--train_seqlen` (512 → 256), then lower `--eff_bit`.
- **NaN loss:** lower `--lr` (e.g. 1e-5). Keep fp32.
- **Optional later:** a `--eff_bit 1.0` run for comparison; `--use_itq False` to measure what ITQ adds.

## Out of scope
DeepSpeed / multi-GPU, bigger models, and the full (no `--limit`) zero-shot suites. These are possible later on CUDA hardware.
