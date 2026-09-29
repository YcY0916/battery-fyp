# LSTM 超参数对比实验：运行说明

实验分两批：

| 批次 | 网格 | 组数 | 验证 / 早停评估起点 | 结果目录 | 权重目录 |
|---|---|---|---|---|---|
| 最初一批（已跑完） | W ∈ {30, 60, 120} × H ∈ {32, 64, 128} × L ∈ {1, 2} | 18 | 第 119 行及之后 | `results/lstm_sweep/` | `models/lstm_sweep/` |
| 扩展一批 | W ∈ {180, 240} × H ∈ {32, 128} × L = 1 | 4 | 第 239 行及之后 | `results/lstm_sweep_ext/` | `models/lstm_sweep_ext/` |

最初 18 组的最佳配置落在 W = 120 这个边界上，所以补跑更长的窗口。两批合并时，把 18 个已训练好的模型读进来（不重新训练），在"第 239 行及之后"的验证点上重新计算误差，与新的 4 组放在同一张 22 组的表里比较。

代码在 `src/`：

| 文件 | 作用 |
|---|---|
| `src/lstm_data.py` | 读取、筛选、划分、1 s 重采样、标准化、窗口 Dataset |
| `src/lstm_model.py` | 模型：`LSTM(3, H, L)` → 最后一个时间步 → `Linear(H, 1)` |
| `src/sweep_lstm.py` | 训练对比实验（只用训练集和验证集，读到测试文件会直接报错）；`--ext` 跑扩展的 4 组 |
| `src/combine_lstm_sweep.py` | 把 18 + 4 组合并成一张表，统一在第 239 行之后重算验证误差（只读验证集） |
| `src/evaluate_lstm.py` | 选定配置之后，在测试集上评估这一组 |

## 实验设置

- 输入 `voltage, current, temperature`，输出窗口最后一个时间点的 `soc`；`ah` 列不读入。
- 只用驾驶工况（UDDS、HWFET、LA92、US06）和 Mixed，按 manifest 的 `test_type` 筛选。
- 按文件划分：
  - 验证集（6 个）：611_Mixed8、604_Mixed8、590_Mixed8、571_Mixed8、552_Mixed8、557_Mixed3；
  - 训练集（37 条序列）：其余 Mixed，包括 602_Mixed4 和 571_Mixed6 的 `_part1`、`_part2`；
  - 测试集（23 个）：全部 UDDS、HWFET、LA92、US06（25 °C 没有 HWFET）。
- 每条序列各自按 `floor(time_s)` 重采样到 1 s：电压、电流、温度取平均，soc 取该秒最后一行。
- 标准化参数只用训练集计算，存为 `norm_stats.json`（两批完全相同，合并脚本会检查）。
- 滑动窗口步长 1，每条序列各自切，比 W 短的序列跳过（W ≥ 60 时跳过只有 39 行的 `571_Mixed6_part1`）。
- **评估起点**：验证误差和早停用的验证 loss 只在"序列内第 N 行及之后"的点上计算，N 取最长窗口能覆盖的第一个点（W_max − 1），这样所有配置比较的是同一批点。
  - 最初 18 组：N = 119，每组 41,777 个验证点；
  - 扩展 4 组和 22 组合并表：N = 239，每组 41,057 个验证点（每个验证文件少了开头 120 个点）。
  - `sweep_lstm.py`、`evaluate_lstm.py`、`combine_lstm_sweep.py` 都可以用 `--eval-start` 指定，默认 239。
- MSE loss，Adam（lr = 0.001），batch 256，每轮打乱，最多 200 轮，验证 loss 连续 15 轮没有创新低就停止，保存验证 loss 最低那一轮的权重，随机种子 42，dropout 0。

| W | 训练窗口数 | 验证点数（起点 239） |
|---|---|---|
| 30 | 223,591 | 41,057 |
| 60 | 222,501 | 41,057 |
| 120 | 220,341 | 41,057 |
| 180 | 218,181 | 41,057 |
| 240 | 216,021 | 41,057 |

## 在 Windows 上运行（扩展实验）

下面的命令都在 **Anaconda Prompt** 里输入。`D:\battery-fyp` 换成你电脑上仓库的实际位置，`你的环境名` 换成你的 conda 环境名。**不要删除或移动 `results\lstm_sweep\` 和 `models\lstm_sweep\`**，合并时要用到里面最初 18 组的结果和权重。

### 第 1 步：进入环境，切换到新代码

```bat
conda activate 你的环境名
cd /d D:\battery-fyp
git fetch origin
git checkout lstm-sweep-ext
git pull origin lstm-sweep-ext
```

（Pull Request 合并以后，改为 `git checkout master` 和 `git pull origin master`。）

### 第 2 步：确认 PyTorch 能用 GPU

```bat
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

应当输出 `True` 和显卡名。如果是 `False`，脚本会自动改用 CPU（会慢很多）。

### 第 3 步：smoke 测试（约 3 分钟）

```bat
python src\sweep_lstm.py --smoke --eval-start 119
python src\sweep_lstm.py --ext --smoke
python src\combine_lstm_sweep.py --smoke
```

- 第一条是最初那批的 smoke（W30_H32_L1、W60_H64_L2，评估起点 119）；如果之前已经跑过，会直接显示"已完成，跳过"。
- 第二条是扩展的 smoke：只用 25 °C 的数据，跑 W180_H32_L1 和 W240_H32_L1，每组 2 轮。
- 第三条把两个 smoke 目录合并成一张表，检查合并脚本能否跑通。
- smoke 的输出都在 `*_smoke` 目录里，与正式结果分开，也不会被提交。

### 第 4 步：正式跑扩展的 4 组

```bat
python src\sweep_lstm.py --ext
```

- 跑 W180_H32_L1、W180_H128_L1、W240_H32_L1、W240_H128_L1，评估起点 239，输出到 `results\lstm_sweep_ext\` 和 `models\lstm_sweep_ext\`。
- 可以随时按 `Ctrl+C` 中断，之后再输入**同一条命令**就会继续：已经跑完的组合直接跳过；中断时正在跑的那一组从第 1 轮重新开始。
- 想分两次跑：

  ```bat
  python src\sweep_lstm.py --ext --windows 180
  python src\sweep_lstm.py --ext --windows 240
  ```

### 第 5 步：合并成 22 组的结果表

```bat
python src\combine_lstm_sweep.py
```

- 读入 `models\lstm_sweep\` 的 18 个模型和 `models\lstm_sweep_ext\` 的 4 个模型，**不重新训练**，只在验证集上重新计算误差，用 GPU 一般一两分钟就完成。
- 写出 `results\lstm_sweep_ext\sweep_results_22.csv`。**`results\lstm_sweep\sweep_results.csv`（旧的 18 组表）不会被改动。**
- 每一组还会在它训练时用的评估起点上再算一次，与当时记录的验证 RMSE 对照（日志里的"重算 vs 记录"）。两者应当几乎相同，相差超过 0.01 个百分点时会标出来提醒。
- 4 组没跑完时会报错并列出缺哪几组；只想先看已完成的组合，可以加 `--allow-missing`。

### 第 6 步：选定配置

打开 `results\lstm_sweep_ext\sweep_results_22.csv`（已按起点 239 的验证 RMSE 从小到大排好），结合 `results\lstm_sweep\curves\` 和 `results\lstm_sweep_ext\curves\` 里的训练曲线选定一组。**选择只依据验证集。**

### 第 7 步：在测试集上评估选定的配置

例如选定 W=240、H=128、L=1：

```bat
python src\evaluate_lstm.py --window 240 --hidden 128 --layers 1
```

- 权重会自动在 `models\lstm_sweep_ext\` 和 `models\lstm_sweep\` 里找，不需要指定是哪一批。
- 默认只评估"序列内第 239 行及之后"的点，与合并表的验证集一致；需要改的话加 `--eval-start 行号`（不能小于 W−1）。
- 结果目录名里带着评估起点（例如 `test_eval\W240_H128_L1_start239\`），换起点重评不会覆盖之前的结果。

### （可选）核对数据划分

```bat
python src\lstm_data.py
```

打印训练集、验证集、测试集各有哪些文件，以及 W = 30、60、120、180、240 时的窗口数。这一步为了数行数会读取测试文件（不训练、不评估模型）；不想碰测试集可以跳过。

### 附：最初 18 组的命令（已跑完，一般不需要再运行）

最初 18 组是用评估起点 119 跑的，续跑或复现时必须加 `--eval-start 119`：

```bat
python src\sweep_lstm.py --eval-start 119
```

不加这个参数的话，脚本会发现 `results\lstm_sweep\` 里已有结果的设置不同，直接报错停止，不会训练任何东西。

## 输出文件

### `python src\sweep_lstm.py --ext` 的输出（最初 18 组的输出结构相同，在 `lstm_sweep` 目录里）

| 路径 | 内容 |
|---|---|
| `results/lstm_sweep_ext/sweep_results.csv` | 这 4 组自己的汇总表（起点 239），按验证 RMSE 排序 |
| `results/lstm_sweep_ext/norm_stats.json` | 标准化参数（训练集的 mean、std，以及用了哪些序列） |
| `results/lstm_sweep_ext/sweep_log.txt` | 运行日志（每轮的 loss、跳过的短序列、用时等），多次运行会接着往后写 |
| `results/lstm_sweep_ext/runs/W*_H*_L*.json` | 每组的完整结果和设置；它存在就表示这一组已经跑完 |
| `results/lstm_sweep_ext/runs/W*_H*_L*_history.csv` | 每组每一轮的训练 loss、验证 loss、验证 RMSE、用时 |
| `results/lstm_sweep_ext/curves/W*_H*_L*.png` | 每组的训练曲线（训练/验证 MSE，对数坐标，虚线为最佳轮） |
| `models/lstm_sweep_ext/W*_H*_L*.pt` | 验证 loss 最低那一轮的权重，里面同时存了配置、标准化参数和训练设置 |

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

### `python src\combine_lstm_sweep.py` 的输出

| 路径 | 内容 |
|---|---|
| `results/lstm_sweep_ext/sweep_results_22.csv` | 22 组汇总，全部在"第 239 行及之后"的验证点上计算，按验证 RMSE 从小到大排序 |
| `results/lstm_sweep_ext/combine_log.txt` | 运行日志（一致性检查、每组"重算 vs 记录"的对照） |

`sweep_results_22.csv` 在上表各列之外还有：

| 列 | 含义 |
|---|---|
| `sweep` | 这一组属于哪一批（`lstm_sweep` 或 `lstm_sweep_ext`） |
| `val_rmse_pct`、`val_mae_pct`、`val_rmse_pct_<温度>` | **统一在第 239 行及之后**重新计算的验证误差（排序依据） |
| `early_stop_eval_start` | 这一组训练时早停用的评估起点（最初 18 组为 119，扩展 4 组为 239） |
| `val_rmse_pct_recorded` | 这一组训练时记录的验证 RMSE（评估起点见上一列），只作对照，不能与 `val_rmse_pct` 直接比较 |
| `recheck_diff_pct` | 在训练时的起点上重算的 RMSE 减去记录值，应当接近 0 |

说明：最初 18 组保存的"最佳轮"是按起点 119 的验证 loss 选的，扩展 4 组是按起点 239 选的。两批验证点只差每个文件开头的 120 个点（41,777 与 41,057），对选出的轮数影响很小，但写报告时应当说明。

### `python src\evaluate_lstm.py ...` 的输出

输出在 `results/<批次>/test_eval/<配置>_start<评估起点>/`：

| 路径 | 内容 |
|---|---|
| `rmse_table.csv` | 温度 × 工况的 RMSE（SOC 百分点）；`all` 行/列是把相应测试点合在一起算的 |
| `mae_table.csv` | 同上，MAE |
| `per_file_metrics.csv` | 每个测试文件的点数、RMSE、MAE、最大绝对误差 |
| `summary.json` | 全部测试点合在一起的 RMSE、MAE，以及评估设置 |
| `predictions/<文件>.csv` | 每个测试文件逐秒的 `time_s, soc_true, soc_pred` |
| `plots/<文件>.png` | 每个测试文件的"真实 SOC vs 预测 SOC"曲线，下方是误差曲线 |

## 预计运行时间（粗略）

GPU 上 LSTM 每轮用时大致与窗口长度成正比，所以可以用你已经跑完的 W=120 结果来推算（看 `results\lstm_sweep\sweep_results.csv` 的 `train_time_s`，或 `runs\*.json` 的 `sec_per_epoch`）：

| 新配置 | 每轮用时 ≈ |
|---|---|
| W180_H32_L1 | 1.5 × W120_H32_L1 |
| W240_H32_L1 | 2 × W120_H32_L1 |
| W180_H128_L1 | 1.5 × W120_H128_L1 |
| W240_H128_L1 | 2 × W120_H128_L1 |

如果早停的轮数和 W=120 差不多，扩展 4 组的总时间约为 **3.5 × (W120_H32_L1 的 `train_time_s` + W120_H128_L1 的 `train_time_s`)**。合并脚本不训练，只需一两分钟。

（云端 4 核 CPU 上，W240_H128_L1 每轮约 20 分钟，不建议在 CPU 上跑。）
