# ShakGPT Run 1 — Postmortem

**Date:** 2026-05-04
**Run duration:** ~10 hours before termination
**Cost incurred:** ~$7.40
**Outcome:** Aborted at step ~35,000 / 106,808 due to data ordering bug

## Summary

The first full training run of ShakGPT was terminated partway through after identifying that the training data was being consumed in source-block order rather than as an interleaved mixture. The model was being trained almost exclusively on FineWeb-Edu for the first ~60% of training, with Wikipedia, Gutenberg, and StarCoder concentrated in the back portions of the dataset. This violates the intended data mix and would produce a model with confused training dynamics and degraded final quality.

A separate methodological issue with validation evaluation was also identified: each `val_loss` measurement used a different val batch, making the val loss curve unreliable as a learning signal.

## What worked

- CUDA Graphs implementation via `torch.compile(mode="reduce-overhead")` succeeded. Steady-state throughput hit 60,500 tokens/sec, ~2.7× the pre-optimization baseline of 22,500 tok/s.
- GPU utilization reached 97% (vs ~12.6% MFU before), confirming the bottleneck was kernel launch overhead and that the optimization addressed it.
- Loss decreased monotonically and matched expected trajectory through the FineWeb-only phase: step 1K loss ~5.0, step 8K loss ~3.5, step 25K loss ~3.1.
- No instabilities, NaN values, or numerical issues during the run.
- Checkpoint rotation logic worked correctly after a process-restart fix.

## What went wrong

### Bug 1 (critical): Sequential data ordering in `prepare.py`

The data preparation script processes datasets sequentially:

```python
for dataset in DATASETS:
    process_source(dataset, train_bf, val_bf, tokenizer)
```

Each source's tokens are written contiguously to `train.bin`. The resulting file structure is:

```
[FineWeb-Edu ~4.2B] [Wikipedia ~1.4B] [Gutenberg ~1.05B] [StarCoder ~0.35B]
```

Combined with the data loader reading sequentially from offset 0, this means:
- Steps 0 to ~64,000 see only FineWeb-Edu
- Steps 64,000 to ~85,000 see only Wikipedia
- Steps 85,000 to ~101,000 see only Gutenberg
- Steps 101,000 to 106,808 see only StarCoder

The intended uniform mix at every step was never realized. The model would have experienced sharp distribution shifts at each source boundary, with associated catastrophic forgetting of earlier-source patterns.

**Impact:** Final model quality would have been substantially worse than a properly-interleaved run. Training dynamics would have shown unexplainable discontinuities at source boundaries, complicating any interpretability research.

### Bug 2 (significant): Non-fixed validation set

The `evaluate_step` function pulls a single fresh batch from the val loader on each call. The val loader advances sequentially, so:

- Step 500 measures on val tokens [0:16K]
- Step 1000 measures on val tokens [16K:32K]
- Step 1500 measures on val tokens [32K:48K]

Each `val_loss` measurement is on a different test set, mixing "model improved" with "this batch is easier/harder." Single-batch evaluation also has high variance (±0.1–0.2 nats from data variance alone).

**Impact:** The val loss curve in `training.log` is directionally informative but not a clean measurement. Specific values cannot be trusted; trends across many points are roughly meaningful.

### Bug 3 (minor): Outdated process running on instance

When the run was first launched, the running Python process was using a pre-fix version of `train.py` because the local edits had not been pushed to GitHub before pulling on the instance. This caused `step_0.pt` to be saved (the `step > 0` guard wasn't active) and rotation logic was missing. Discovered and corrected by pushing fixes and restarting.

**Impact:** Wasted ~50 minutes of training time on the first attempt. Process was correct on second attempt.

## Root causes

These bugs had different root causes worth distinguishing:

1. **Bug 1 (data ordering)** was a process gap. `prepare.py` was written without reference to how production LLM training pipelines handle multi-source data mixing. No one (including code review) explicitly asked "are sources interleaved?" because no checklist required it.

2. **Bug 2 (val methodology)** was a similar gap — validation methodology wasn't designed up front, just thrown together. The same checklist gap applies.

3. **Bug 3 (stale process)** was a workflow issue: a manual edit-then-launch sequence with no verification step that the running code matches the on-disk code.

## Lessons learned

- **Pre-launch checklist is required for any run costing more than $5 or running more than 2 hours.** A 30-minute checklist would have caught all three bugs.
- **Inspect the actual data, not just the manifest.** Decoding a few hundred sequences from `train.bin` would have made Bug 1 obvious in 30 seconds.
- **Read papers for the boring parts, not just the architecture.** GPT-3, Pythia, and Chinchilla papers all describe data mixing explicitly. Reading these for *implementation*, not just architecture, would have prevented Bug 1.
- **Smoke-test before committing to a long run.** A 200-step test run at $0.50 of compute would have produced loss curves that, while not catching Bug 1 directly, would have built the habit of validating runs before committing to them.
- **Plot, don't just read numbers.** Loss curves are visual. Reading numbers in a terminal is not how loss should be analyzed.
- **Verify the running code is the latest code.** `git status` and `git log` on the instance immediately before launch.

## Artifacts kept from this run

- `shakgpt_run1.log`: full training log for reference and as a comparison baseline for the corrected run
- `shakgpt_run1_manifest.json`: data manifest documenting what data was prepared (and how)

## Artifacts discarded

- All checkpoints (`checkpoints/*.pt`): trained on incorrectly-ordered data, no research value
- `train.bin` and `val.bin`: will be regenerated with the corrected pipeline
