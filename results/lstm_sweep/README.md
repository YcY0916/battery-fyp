# LSTM 超参数对比实验：运行说明

本目录存放 LSTM 超参数对比（18 组）的结果。代码在 `src/`：

| 文件 | 作用 |
|---|---|
| `src/lstm_data.py` | 读取、筛选、划分、1 s 重采样、标准化、窗口 Dataset |
| `src/lstm_model.py` | 模型：`LSTM(3, H, L)` → 最后一个时间步 → `Linear(H, 1)` |
| `src/sweep_lstm.py` | 跑 18 组对比（只用训练集和验证集，读到测试文件会直接报错） |
| `src/evaluate_lstm.py` | 选定配置之后，在测试集上评估这一组 |

## 实验设置（固定不变）

- 输入 `voltage, current, temperature`，输出窗口最后一个时间点的 `soc`；`ah` 列不读入。
- 只用驾驶工况（UDDS、HWFET、LA92、US06）和 Mixed，按 manifest 的 `test_type` 筛选。
- 按文件划分：
  - 验证集（6 个）：611_Mixed8、604_Mixed8、590_Mixed8、571_Mixed8、552_Mixed8、557_Mixed3；
  - 训练集（37 条序列）：其余 Mixed，包括 602_Mixed4 和 571_Mixed6 的 `_part1`、`_part2`；
  - 测试集（23 个）：全部 UDDS、HWFET、LA92、US06（25 °C 没有 HWFET）。
- 每条序列各自按 `floor(time_s)` 重采样到 1 s：电压、电流、温度取平均，soc 取该秒最后一行。
- 标准化参数只用训练集计算，存为 `norm_stats.json`。
- 滑动窗口步长 1，每条序列各自切，比 W 短的序列跳过（日志里会列出：W=60、120 时跳过 `571_Mixed6_part1`，只有 39 行）。
- 验证误差只在"序列内第 119 行（第 119 秒）及之后"的点上计算，三种窗口长度评估的是同一批点（41,777 个）。早停用的验证 loss 也只算这些点。
- MSE loss，Adam（lr = 0.001），batch 256，每轮打乱，最多 200 轮，验证 loss 连续 15 轮没有创新低就停止，保存验证 loss 最低那一轮的权重，随机种子 42，dropout 0。
- 网格：W ∈ {30, 60, 120} × H ∈ {32, 64, 128} × L ∈ {1, 2}。

## 在 Windows 上运行

下面的命令都在 **Anaconda Prompt** 里输入。`D:\battery-fyp` 换成你电脑上仓库的实际位置，`你的环境名` 换成你的 conda 环境名。

### 第 1 步：进入环境和仓库

```bat
conda activate 你的环境名
cd /d D:\battery-fyp
git fetch origin
git checkout lstm-sweep
git pull origin lstm-sweep
```

（Pull Request 合并以后，改为 `git checkout master` 和 `git pull origin master`。）

### 第 2 步：确认 PyTorch 能用 GPU

```bat
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

应当输出 `True` 和 `NVIDIA GeForce RTX 4050 Laptop GPU`。如果是 `False`，脚本会自动改用 CPU（会慢很多）。

### 第 3 步（可选）：核对数据划分

```bat
python src\lstm_data.py
```

打印训练集、验证集、测试集各有哪些文件，以及三种窗口长度下的窗口数。这一步为了数行数会读取测试文件，但不训练、不评估任何模型；不想碰测试集可以跳过。

### 第 4 步：smoke 测试（约 1 分钟）

```bat
python src\sweep_lstm.py --smoke
```

只用 25 °C 的数据，跑 W30_H32_L1 和 W60_H64_L2 两组，每组 2 轮。结果写到 `results\lstm_sweep_smoke\` 和 `models\lstm_sweep_smoke\`，与正式结果分开，不会影响正式实验的"断点续跑"，也已经在 `.gitignore` 里，不会被提交。看到最后打印出汇总表就说明代码能跑通。

### 第 5 步：正式跑 18 组

```bat
python src\sweep_lstm.py
```

- 可以随时按 `Ctrl+C` 中断。之后再输入**同一条命令**就会继续：已经跑完的组合（`runs\` 里有结果 JSON 且 `models\lstm_sweep\` 里有权重）直接跳过；中断时正在跑的那一组会从第 1 轮重新开始。
- 想分几次跑，可以只跑一部分，例如先跑 W=30 的 6 组：

  ```bat
  python src\sweep_lstm.py --windows 30
  python src\sweep_lstm.py --windows 60
  python src\sweep_lstm.py --windows 120
  ```

  也可以用 `--hidden`、`--layers` 进一步限定。
- 每跑完一组，`sweep_results.csv` 都会重新生成。
- 如果想从头重跑，先删除旧结果：

  ```bat
  rmdir /s /q results\lstm_sweep\runs results\lstm_sweep\curves
  del results\lstm_sweep\norm_stats.json results\lstm_sweep\sweep_results.csv results\lstm_sweep\sweep_log.txt
  rmdir /s /q models\lstm_sweep
  ```

### 第 6 步：选定配置

打开 `results\lstm_sweep\sweep_results.csv`（已按验证 RMSE 从小到大排好），结合 `curves\` 里的训练曲线选定一组配置。**选择只依据验证集。**

### 第 7 步：在测试集上评估选定的配置

例如选定 W=60、H=64、L=2：

```bat
python src\evaluate_lstm.py --window 60 --hidden 64 --layers 2
```

测试集也只评估"序列内第 119 行及之后"的点，与验证集一致；需要改的话加 `--eval-start 行号`（不能小于 W−1）。

## 输出文件

### `python src\sweep_lstm.py` 的输出

| 路径 | 内容 |
|---|---|
| `results/lstm_sweep/sweep_results.csv` | 汇总表，按验证 RMSE 从小到大排序，每组一行 |
| `results/lstm_sweep/norm_stats.json` | 标准化参数（训练集的 mean、std，以及用了哪些序列） |
| `results/lstm_sweep/sweep_log.txt` | 运行日志（每轮的 loss、跳过的短序列、用时等），多次运行会接着往后写 |
| `results/lstm_sweep/runs/W*_H*_L*.json` | 每组的完整结果和设置；它存在就表示这一组已经跑完 |
| `results/lstm_sweep/runs/W*_H*_L*_history.csv` | 每组每一轮的训练 loss、验证 loss、验证 RMSE、用时 |
| `results/lstm_sweep/curves/W*_H*_L*.png` | 每组的训练曲线（训练/验证 MSE，对数坐标，虚线为最佳轮） |
| `models/lstm_sweep/W*_H*_L*.pt` | 验证 loss 最低那一轮的权重，里面同时存了配置和标准化参数 |

`sweep_results.csv` 各列：

| 列 | 含义 |
|---|---|
| `rank` | 按验证 RMSE 的名次 |
| `config`、`window`、`hidden_size`、`num_layers` | 配置 |
| `val_rmse_pct`、`val_mae_pct` | 验证集 RMSE、MAE（SOC 百分点，即误差 × 100） |
| `val_rmse_pct_n20degC` … `val_rmse_pct_40degC` | 验证集按温度分开的 RMSE（每个温度就是一个验证文件） |
| `best_epoch` | 验证 loss 最低的一轮（保存的就是这一轮的权重） |
| `epochs_trained` | 实际训练的总轮数 |
| `stopped_early` | 是否因为早停而提前结束 |
| `train_time_s` | 训练总用时（秒，含每轮的验证） |
| `n_params` | 模型参数量 |
| `n_train_windows`、`n_val_points` | 训练窗口数、验证点数 |
| `device_name` | 训练用的设备 |

### `python src\evaluate_lstm.py ...` 的输出

输出在 `results/lstm_sweep/test_eval/<配置>/`：

| 路径 | 内容 |
|---|---|
| `rmse_table.csv` | 温度 × 工况的 RMSE（SOC 百分点）；`all` 行/列是把相应测试点合在一起算的 |
| `mae_table.csv` | 同上，MAE |
| `per_file_metrics.csv` | 每个测试文件的点数、RMSE、MAE、最大绝对误差 |
| `summary.json` | 全部测试点合在一起的 RMSE、MAE，以及评估设置 |
| `predictions/<文件>.csv` | 每个测试文件逐秒的 `time_s, soc_true, soc_pred` |
| `plots/<文件>.png` | 每个测试文件的"真实 SOC vs 预测 SOC"曲线，下方是误差曲线 |

## 预计运行时间（粗略）

云端测试机（4 核 CPU）实测：最大的一组 W120_H128_L2 每轮 **329 s**（训练 220,341 个窗口 + 验证 41,777 个点）。按每批用时换算，18 组各跑一轮在 CPU 上合计约 1,330 s，其中 W120_H128_L2 一组就占四分之一；最小的 W30_H32_L1 每轮约 7 s。

RTX 4050 上无法实测，按经验粗略估计：大配置每轮约 5-12 s，小配置每轮约 2-4 s（小配置主要受每批的固定开销限制，GPU 加速不明显）。每组实际跑多少轮取决于早停，一般在 50-120 轮之间，最多 200 轮。据此：

- 18 组全部跑完：**约 2-5 小时**；
- 最坏情况（每组都跑满 200 轮）：约 6-8 小时。

比较准的做法：先跑一组，例如 `python src\sweep_lstm.py --windows 120 --hidden 128 --layers 2`，从日志里看每轮用时（`sweep_log.txt` 每行最后的秒数，或 `runs\*.json` 的 `sec_per_epoch`），再按上面的比例推算。**不建议在 CPU 上跑完整对比**（估计要一天以上）。
