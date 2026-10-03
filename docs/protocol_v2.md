# Experiment protocol v2 (paper_0to40_v2)

LSTM state-of-charge (SOC) estimation on the Kollmeyer LG HG2 data set, 0, 10, 25 and 40 °C.
Target deployment: an ESP32-S3-class microcontroller, so accuracy is weighed against deployment cost.

This file was written before any v2 training. It is fixed by the git tag `v2-protocol`. After any
v2 result has been seen, the rules below are not changed; if a change is unavoidable, the author is
asked first and the change is recorded in the changelog at the end (what changed and why).

Machine-readable parts of the protocol:

| File | Content |
|---|---|
| `protocol/v2/norm_stats.json` | normalisation statistics (training set only); sha256 `00c97f004a5f80a4144dda3e4fe5bf8400e36462c36cef5dba67a12666156a5b` |
| `protocol/v2/data_sha256.json` | sha256 of every data file used (train, validation, test, unseen test, manifest) |
| `src/v2_common.py` | grid, seeds, evaluation start, pre-flight checks, run-folder layout |
| `shards/*.txt` | which runs each platform trains (does not change any training setting) |

## 1. History: what was done before v2 (v1)

- v1 (experiment `paper_0to40`, code at tag `paper-v1-w300` = commit
  `b5f9fdd5fd23e10bddc53d92b9e20167dd4c6c9d`) ran an 18-config sweep (W <= 300), a multi-seed step
  and **one test-set evaluation of W300_H64_L1 (in the v2 naming L1_W300_H64), scored on rows >= 299**.
- The test set has therefore been seen once, in v1. v2 is a complete, re-designed experiment:
  every v2 model is trained from scratch, no v1 run, weight or metric is reused, and every v2
  result comes from this protocol. The v1 outputs (`results/paper_0to40`, `models/paper_0to40`)
  are read only and backed up; no v2 script writes there (the scripts refuse to).
- v1 training times are used only to estimate the v2 run time.

## 2. Data and evaluation (unchanged from v1 except the evaluation start)

- Data: `data/processed/kollmeyer_clean` (tracked in the repository). Only drive cycles and Mixed
  cycles are used, selected by the manifest test type.
- Split by whole files (parts `_part1/_part2` follow their file), per temperature:
  validation = the highest-numbered Mixed file; training = the other Mixed files;
  test = UDDS, HWFET, LA92, US06 (25 °C has no HWFET). Each file is one independent experiment.

| Set | 0 °C | 10 °C | 25 °C | 40 °C |
|---|---|---|---|---|
| Train | 589_Mixed1-2, 590_Mixed4-7 | 567_Mixed1-2, 571_Mixed4-7 (571_Mixed6 in two parts) | 551_Mixed1-2, 552_Mixed3-7 | 556_Mixed1-2 |
| Validation | 590_Mixed8 | 571_Mixed8 | 552_Mixed8 | 557_Mixed3 |
| Test | 589_UDDS, 589_HWFET, 589_LA92, 589_US06 | 576_UDDS, 576_HWFET, 582_LA92, 567_US06 | 551_UDDS, 551_LA92, 551_US06 | 556_UDDS, 556_HWFET, 556_LA92, 556_US06 |

  Unseen-temperature test (only at the very end, never used for any choice): the four cycles at
  −20 °C (610) and −10 °C (596, 601).
- Inputs: voltage, current, temperature (the `ah` column is never an input). Output: the SOC of
  the last row of the window. Labels: the `soc` column of the processed files.
- Resampling: every file to 1 Hz (f_s = 1 Hz) by `floor(time_s)`; voltage, current and temperature
  are averaged within the second, SOC is the last row of the second.
- Normalisation: mean and standard deviation of the three inputs over all training rows after
  resampling, computed once from the training set only. Every run recomputes it from the training
  files and stops unless the result is byte-identical to `protocol/v2/norm_stats.json`, whose
  sha256 must equal the v1 file (`00c97f00...6156a5b`). A copy is written to
  `results/paper_0to40_v2/norm_stats.json`.
- **Evaluation start EVAL_START = 359** (longest window 360 − 1). The validation loss used for
  early stopping and every reported validation and test metric use only rows with in-sequence index
  >= 359 of each validation / test file, so every window length is scored on the same points
  (29,645 validation points: 7,225 / 7,783 / 8,233 / 6,404 at 0 / 10 / 25 / 40 °C).
- The training set is not affected by EVAL_START: a model with window W uses every training
  window that ends at row W − 1 or later (the 39-row file 571_Mixed6_part1 is shorter than every
  window and is skipped). Training windows: 150,726 (W = 60), 149,466 (120), 148,206 (180),
  146,946 (240), 145,686 (300), 144,426 (360).
- Metrics: RMSE and MAE in SOC percentage points, overall (all validation points pooled) and per
  temperature. Standard deviations over seeds are sample standard deviations (ddof = 1).
- No statistical significance test is done. Wording such as "no significant improvement" is not
  used; only descriptions such as "the improvement was small relative to the observed
  seed-to-seed variation".

## 3. Training settings (read from the existing code, `src/sweep_lstm.py`, `src/lstm_model.py`)

| Setting | Value |
|---|---|
| Model | LSTM (input size 3, hidden size H, L layers, inter-layer dropout 0) -> output of the last time step -> Linear(H, 1) |
| Loss / optimiser | MSE / Adam, learning rate 0.001, default betas, no weight decay, no LR schedule |
| Batch size | 256 (training); 2048 for evaluation (memory only, no effect on results) |
| Epochs | at most 200; early stopping when the validation loss (rows >= 359) has not reached a new minimum for 15 epochs; the weights of the best epoch are kept |
| Shuffling | training windows reshuffled every epoch with `torch.Generator().manual_seed(seed)` |
| Seeding | `random`, `numpy`, `torch`, `torch.cuda` seeded with the run seed before the model is built; PyTorch default weight initialisation |
| cuDNN | `deterministic = True`, `benchmark = False` |

Trainable parameters: L1: H32 4,769, H64 17,729, H128 68,225; L2: H32 13,217, H64 51,009,
H128 200,321 (independent of W). Initial delay before the first estimate: W / f_s seconds.

## 4. Step 2: grid search (seed 42, 30 configs)

- L = 1: W in {60, 120, 180, 240, 300, 360} × H in {32, 64, 128} (18 configs)
- L = 2: W in {180, 240, 300, 360} × H in {32, 64, 128} (12 configs)
- Config names: `L1_W300_H64`; run names: `L1_W300_H64_seed42`.
- After step 2 the runs are merged, the step-2 heatmaps are drawn and a ranked text summary is
  produced (`tables/step2_summary.txt`).

## 5. Choice of the six step-3 configs (by the author; no code chooses)

- Claude recommends six configs with reasons, using mainly the overall validation RMSE (seed 42,
  rows >= 359), pointing out which gaps are too small to separate with a single seed, and, where
  gaps are small, the per-temperature results (especially each config's worst temperature), the
  number of parameters, the inference cost and the initial delay W / f_s. The plain top six by
  overall RMSE are listed next to the recommendation, with the differences explained.
- The author chooses the final six. They are committed in `shards/step3_candidates.txt`, and the
  step-2 heatmaps are redrawn with `--highlight` on these six.

## 6. Step 3: multiple seeds

- Each of the six configs is trained again with seeds 43, 44, 45 and 46 (24 new runs); with seed
  42 from step 2 every config has five seeds.
- Reported per config: overall and per-temperature validation RMSE as mean ± sample std (ddof = 1),
  every seed's result, parameters, CPU inference time, initial delay W / f_s.
- CPU inference time (method of v1 `benchmark_inference.py`): on the laptop CPU (Intel
  i7-12700H), seed-42 weights, `torch.set_num_threads(1)`, batch 1, one window of length W,
  100 warm-up passes then 1000 timed passes, mean and sample std in ms. Before timing: other
  programs closed, Google Drive / OneDrive sync paused, charger connected, Windows power mode
  "Best performance", no training on the laptop, the laptop not touched during the measurement.
  These times show the relative trend only; they are not the absolute time on the MCU.
- The outputs are neutral tables and figures; no code picks or labels a "best" or "recommended"
  config. The author chooses the final config.

## 7. After the final choice: test set (once)

- The test set is evaluated once (`src/evaluate_v2.py`, which refuses a second run): the final
  config, seeds 42-46, rows >= 359, plus the unseen temperatures −20 and −10 °C (`--unseen`,
  reported separately). Results are written to `results/paper_0to40_v2/test/<config>/`.
- Test files are hashed against `protocol/v2/data_sha256.json` before they are read; during
  training the test files are only checked for presence and never opened (a guard raises an
  error on any attempt).
- "Average" = mean over the test files of each seed's per-file metric, then mean ± std over the
  seeds; the RMSE pooled over all test points is reported as well.

## 8. Figures and tables

Every figure is written as PNG (300 dpi) and PDF, with English text only, and described in
`results/paper_0to40_v2/figure_notes.md`.

1. Step-2 heatmap (30 configs): rows = configs (L1 then L2, each by W then H; parameters next to
   the label), columns = Overall, 0 °C, 10 °C, 25 °C, 40 °C, cell text = validation RMSE with three
   decimals. Two versions: colour normalised inside each column (rank inside the column; numbers
   stay absolute) and one colour scale for the whole figure. `--highlight` outlines given configs.
   The caption states that these are single training runs with seed 42.
2. `step2_ranking.csv` and a copyable text summary: overall rank, per-temperature RMSE,
   parameters, W.
3. Step-3 heatmap (6 configs, same columns): colour = mean over the five seeds, cell text
   "mean ± std".
4. Step-3 summary table: overall and per-temperature mean and std, every seed's result,
   parameters, CPU inference time (ms), initial delay W / f_s (s).
5. Step-3 dot plot: 6 configs × 5 seeds, one panel per column (Overall and the four
   temperatures); point = mean, bar = ± std, plus every seed as a small dot.
6. Window saturation: for step-3 configs that share L and H but differ in W, the absolute and
   relative improvement between adjacent windows, compared with the seed-to-seed standard
   deviation (no significance test; wording as in section 2).
7. Final paper figures (after the final choice): window length vs validation RMSE (L1 and L2
   lines with different line styles and markers; single seed 42, stated in the caption);
   parameters vs validation RMSE (five seeds); test results of the final config per temperature;
   US06 SOC traces (the RMSE printed in a panel comes from one seed, stated in the caption).
   Inference times are labelled as measured on the laptop CPU (i7-12700H, single thread,
   batch 1), relative trend only, not the absolute MCU time.

## 9. Platforms and run records

- Platforms: laptop (RTX 4050 Laptop GPU, 6 GB), Kaggle (two T4, one process per GPU), Google
  Colab free tier (T4). Default step-2 split (adjusted only to balance time, after the timing probe):

| Platform | Runs |
|---|---|
| Colab T4 | L1 × W60/120/180 × H32/64/128 (9) |
| Kaggle T4 #0 | L1_W360 × H32/64/128, L2_W240 × H32/64/128 (6) |
| Kaggle T4 #1 | L1_W240 × H, L1_W300 × H, L2_W180 × H (9) |
| Laptop | L2 × W300/360 × H32/64/128 (6) |

- The same seed does not give bit-identical results on different GPUs or torch versions. Every
  run records its platform, GPU, torch / CUDA / cuDNN / Python versions, and the merged tables
  list which platform trained which run.
- Pre-flight checks before any training (the run does not start if one fails): code = tag
  `v2-protocol` or no difference in the training files (`src/lstm_data.py`, `src/lstm_model.py`,
  `src/sweep_lstm.py`, `src/v2_common.py`, `src/run_shard.py`, `protocol/v2/*.json`); data files
  present and their sha256 correct; normalisation sha256 correct; GPU available.
- One folder per run, files written atomically (temporary name, then rename), `COMPLETE.json`
  (sha256 of every file) last: weights (`models/paper_0to40_v2/<run>/model.pt`), per-epoch training
  log, validation predictions (npz), metrics (overall and per temperature, rows >= 359) and
  `run_meta.json` (git commit, platform, GPU, versions, normalisation sha256, data sha256 summary,
  code fingerprint, hyperparameters, seed, start and end time, epochs trained, best epoch, seconds
  per epoch, training time).
- An interrupted run is discarded and trained again from epoch 1 (`--skip-existing`).
- Merging stops on missing runs, duplicate runs, incomplete runs, or any difference in code
  fingerprint, normalisation sha256, data sha256 or hyperparameters.

## 10. Language

Everything added or changed for v2 (code, comments, file and folder names, CSV columns, JSON,
logs, figure text, notes, protocol, commit messages, pull requests) is in English; the only
non-ASCII symbols allowed are ° ± − × µ. `src/check_english.py` checks the tracked files and the
experiment outputs; it is run before every pull request and after every stage with new outputs.

## Changelog

- 2026-10-03: first version (before any v2 training).
