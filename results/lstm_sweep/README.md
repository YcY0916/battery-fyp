# LSTM hyperparameter comparison: how to run it

(Old 6-temperature experiment. The scripts `combine_lstm_sweep.py` and `evaluate_lstm.py` were later moved to `src/archive/`.)

The experiment has two batches:

| Batch | Grid | Configs | Validation / early-stopping evaluation start | Results folder | Weights folder |
|---|---|---|---|---|---|
| First batch (finished) | W in {30, 60, 120} × H in {32, 64, 128} × L in {1, 2} | 18 | row 119 onwards | `results/lstm_sweep/` | `models/lstm_sweep/` |
| Extension | W in {180, 240} × H in {32, 128} × L = 1 | 4 | row 239 onwards | `results/lstm_sweep_ext/` | `models/lstm_sweep_ext/` |

The best of the first 18 configs was at the W = 120 edge, so longer windows were added. To merge the two batches, the 18 trained models are loaded (no retraining), their errors are recomputed on the validation points "from row 239 onwards" and compared with the 4 new configs in one 22-config table.

Code in `src/`:

| File | Purpose |
|---|---|
| `src/lstm_data.py` | reading, filtering, split, 1 s resampling, normalisation, window Dataset |
| `src/lstm_model.py` | model: `LSTM(3, H, L)` -> last time step -> `Linear(H, 1)` |
| `src/sweep_lstm.py` | training of the comparison (training and validation sets only; reading a test file raises an error); `--ext` runs the 4 extended configs |
| `src/combine_lstm_sweep.py` | merges the 18 + 4 configs into one table, recomputing the validation error from row 239 (validation set only) |
| `src/evaluate_lstm.py` | after the config is chosen, evaluates it on the test set |

## Settings

- Inputs `voltage, current, temperature`; output the `soc` of the last time step of the window; the `ah` column is not read.
- Only drive cycles (UDDS, HWFET, LA92, US06) and Mixed, selected by the manifest `test_type`.
- Split by file:
  - validation (6): 611_Mixed8, 604_Mixed8, 590_Mixed8, 571_Mixed8, 552_Mixed8, 557_Mixed3;
  - training (37 sequences): the other Mixed, including the `_part1`, `_part2` of 602_Mixed4 and 571_Mixed6;
  - test (23): every UDDS, HWFET, LA92, US06 (25 °C has no HWFET).
- Every sequence is resampled to 1 s by `floor(time_s)`: voltage, current and temperature averaged, soc the last row of the second.
- Normalisation from the training set only, stored as `norm_stats.json` (identical in both batches; the merge script checks it).
- Sliding windows with stride 1, cut per sequence; sequences shorter than W are skipped (for W >= 60 the 39-row `571_Mixed6_part1`).
- **Evaluation start**: the validation error and the validation loss used for early stopping use only the points "from in-sequence row N onwards", where N is the first point the longest window can cover (W_max − 1), so every config is scored on the same points.
  - first 18 configs: N = 119, 41,777 validation points each;
  - the 4 extended configs and the merged 22-config table: N = 239, 41,057 validation points each (120 fewer points at the start of every validation file).
  - `sweep_lstm.py`, `evaluate_lstm.py`, `combine_lstm_sweep.py` all accept `--eval-start`, default 239.
- MSE loss, Adam (lr = 0.001), batch 256, reshuffled every epoch, at most 200 epochs, stop when the validation loss has not reached a new minimum for 15 epochs, keep the weights of the epoch with the lowest validation loss, random seed 42, dropout 0.

| W | Training windows | Validation points (start 239) |
|---|---|---|
| 30 | 223,591 | 41,057 |
| 60 | 222,501 | 41,057 |
| 120 | 220,341 | 41,057 |
| 180 | 218,181 | 41,057 |
| 240 | 216,021 | 41,057 |

## Running on Windows (extension)

Type the commands below in **Anaconda Prompt**. Replace `D:\battery-fyp` with the location of the repository on your computer and `your_env_name` with your conda environment. **Do not delete or move `results\lstm_sweep\` and `models\lstm_sweep\`**; the merge needs the results and weights of the first 18 configs.

### Step 1: activate the environment and switch to the new code

```bat
conda activate your_env_name
cd /d D:\battery-fyp
git fetch origin
git checkout lstm-sweep-ext
git pull origin lstm-sweep-ext
```

(After the pull request is merged, use `git checkout master` and `git pull origin master` instead.)

### Step 2: check that PyTorch can use the GPU

```bat
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

It should print `True` and the GPU name. If it prints `False`, the script falls back to the CPU (much slower).

### Step 3: smoke test (about 3 minutes)

```bat
python src\sweep_lstm.py --smoke --eval-start 119
python src\sweep_lstm.py --ext --smoke
python src\combine_lstm_sweep.py --smoke
```

- The first line is the smoke test of the first batch (W30_H32_L1, W60_H64_L2, evaluation start 119); if it was run before, it prints "already finished, skipped".
- The second line is the smoke test of the extension: 25 °C data only, W180_H32_L1 and W240_H32_L1, 2 epochs each.
- The third line merges the two smoke folders into one table, to check that the merge script runs.
- Smoke outputs go to `*_smoke` folders, separate from the real results and never committed.

### Step 4: run the 4 extended configs

```bat
python src\sweep_lstm.py --ext
```

- Runs W180_H32_L1, W180_H128_L1, W240_H32_L1, W240_H128_L1 with evaluation start 239, output to `results\lstm_sweep_ext\` and `models\lstm_sweep_ext\`.
- `Ctrl+C` can stop it at any time; typing **the same command** again continues: finished configs are skipped, the config that was running starts again from epoch 1.
- To split it in two sessions:

  ```bat
  python src\sweep_lstm.py --ext --windows 180
  python src\sweep_lstm.py --ext --windows 240
  ```

### Step 5: merge into the 22-config table

```bat
python src\combine_lstm_sweep.py
```

- Loads the 18 models of `models\lstm_sweep\` and the 4 of `models\lstm_sweep_ext\`, **does not retrain**, only recomputes the errors on the validation set; with a GPU it takes a minute or two.
- Writes `results\lstm_sweep_ext\sweep_results_22.csv`. **`results\lstm_sweep\sweep_results.csv` (the old 18-config table) is not changed.**
- Every config is also recomputed at the evaluation start it was trained with and compared with the validation RMSE recorded then ("recomputed vs recorded" in the log). The two should be almost equal; a difference above 0.01 percentage points is flagged.
- If some of the 4 configs have not finished, the script stops and lists them; add `--allow-missing` to see the finished ones first.

### Step 6: choose the config

Open `results\lstm_sweep_ext\sweep_results_22.csv` (already sorted by the validation RMSE from start 239) and choose one config, using the training curves in `results\lstm_sweep\curves\` and `results\lstm_sweep_ext\curves\` as well. **The choice is based on the validation set only.**

### Step 7: evaluate the chosen config on the test set

For example W=240, H=128, L=1:

```bat
python src\evaluate_lstm.py --window 240 --hidden 128 --layers 1
```

- The weights are looked up in `models\lstm_sweep_ext\` and `models\lstm_sweep\`; no need to say which batch.
- By default only points "from in-sequence row 239 onwards" are scored, as in the validation set of the merged table; add `--eval-start <row>` to change it (not below W−1).
- The result folder name includes the evaluation start (e.g. `test_eval\W240_H128_L1_start239\`), so re-evaluating with another start never overwrites earlier results.

### (Optional) check the data split

```bat
python src\lstm_data.py
```

Prints which files are in the training, validation and test sets, and the window counts for W = 30, 60, 120, 180, 240. To count the rows this step reads the test files (no training, no evaluation of a model); skip it to leave the test set untouched.

### Appendix: commands of the first 18 configs (finished, normally not needed)

The first 18 configs were run with evaluation start 119; to continue or reproduce them, `--eval-start 119` is required:

```bat
python src\sweep_lstm.py --eval-start 119
```

Without it, the script sees that the existing results in `results\lstm_sweep\` used other settings, stops with an error and trains nothing.

## Output files

### Outputs of `python src\sweep_lstm.py --ext` (the first 18 configs have the same layout, in the `lstm_sweep` folders)

| Path | Content |
|---|---|
| `results/lstm_sweep_ext/sweep_results.csv` | summary of these 4 configs (start 239), sorted by validation RMSE |
| `results/lstm_sweep_ext/norm_stats.json` | normalisation stats (training mean, std, and which sequences were used) |
| `results/lstm_sweep_ext/sweep_log.txt` | run log (loss of every epoch, skipped short sequences, times), appended by every run |
| `results/lstm_sweep_ext/runs/W*_H*_L*.json` | full result and settings of every config; its existence means the config has finished |
| `results/lstm_sweep_ext/runs/W*_H*_L*_history.csv` | training loss, validation loss, validation RMSE and time of every epoch |
| `results/lstm_sweep_ext/curves/W*_H*_L*.png` | training curves of every config (training / validation MSE, log scale, dashed line at the best epoch) |
| `models/lstm_sweep_ext/W*_H*_L*.pt` | weights of the epoch with the lowest validation loss, stored together with the config, normalisation stats and training settings |

Columns of `sweep_results.csv`:

| Column | Meaning |
|---|---|
| `rank` | rank by validation RMSE |
| `config`, `window`, `hidden_size`, `num_layers` | the config |
| `val_rmse_pct`, `val_mae_pct` | validation RMSE, MAE (SOC percentage points, i.e. error × 100) |
| `val_rmse_pct_n20degC` ... `val_rmse_pct_40degC` | validation RMSE per temperature (each temperature is one validation file) |
| `best_epoch` | the epoch with the lowest validation loss (its weights are saved) |
| `epochs_trained` | total epochs trained |
| `stopped_early` | whether training ended by early stopping |
| `train_time_s` | total training time (seconds, including the validation of every epoch) |
| `n_params` | number of model parameters |
| `n_train_windows`, `n_val_points` | training windows, validation points |
| `device_name` | device used for training |

### Outputs of `python src\combine_lstm_sweep.py`

| Path | Content |
|---|---|
| `results/lstm_sweep_ext/sweep_results_22.csv` | summary of the 22 configs, all computed on the validation points "from row 239 onwards", sorted by validation RMSE |
| `results/lstm_sweep_ext/combine_log.txt` | run log (consistency checks, "recomputed vs recorded" of every config) |

`sweep_results_22.csv` has these columns in addition to the ones above:

| Column | Meaning |
|---|---|
| `sweep` | which batch the config belongs to (`lstm_sweep` or `lstm_sweep_ext`) |
| `val_rmse_pct`, `val_mae_pct`, `val_rmse_pct_<temperature>` | validation errors recomputed **from row 239 onwards for every config** (used for sorting) |
| `early_stop_eval_start` | evaluation start used for early stopping when the config was trained (119 for the first 18, 239 for the 4 extended) |
| `val_rmse_pct_recorded` | validation RMSE recorded during training (evaluation start in the previous column), only for comparison, not comparable with `val_rmse_pct` |
| `recheck_diff_pct` | RMSE recomputed at the training start minus the recorded value, should be close to 0 |

Note: the saved "best epoch" of the first 18 configs was chosen with the validation loss from start 119, that of the 4 extended configs from start 239. The two sets of validation points differ only by the first 120 points of every file (41,777 vs 41,057), which has little effect on the chosen epoch, but the report should say so.

### Outputs of `python src\evaluate_lstm.py ...`

In `results/<batch>/test_eval/<config>_start<evaluation start>/`:

| Path | Content |
|---|---|
| `rmse_table.csv` | RMSE per temperature × cycle (SOC percentage points); the `all` row / column pools the corresponding test points |
| `mae_table.csv` | the same for MAE |
| `per_file_metrics.csv` | points, RMSE, MAE and maximum absolute error of every test file |
| `summary.json` | RMSE and MAE over all test points together, and the evaluation settings |
| `predictions/<file>.csv` | `time_s, soc_true, soc_pred` of every test file, second by second |
| `plots/<file>.png` | "true SOC vs predicted SOC" curve of every test file, with the error curve below |

## Expected run time (rough)

On a GPU the time per epoch of the LSTM is roughly proportional to the window length, so it can be estimated from the finished W=120 results (`train_time_s` in `results\lstm_sweep\sweep_results.csv`, or `sec_per_epoch` in `runs\*.json`):

| New config | Time per epoch, about |
|---|---|
| W180_H32_L1 | 1.5 × W120_H32_L1 |
| W240_H32_L1 | 2 × W120_H32_L1 |
| W180_H128_L1 | 1.5 × W120_H128_L1 |
| W240_H128_L1 | 2 × W120_H128_L1 |

If early stopping takes about as many epochs as at W=120, the 4 extended configs take about **3.5 × (`train_time_s` of W120_H32_L1 + `train_time_s` of W120_H128_L1)** in total. The merge script does not train and needs only a minute or two.

(On a 4-core cloud CPU, W240_H128_L1 takes about 20 minutes per epoch; running on a CPU is not recommended.)
