# GRU、1D-CNN、Transformer 与 LSTM 的对比实验

在 LSTM 对比实验（见 [`results/lstm_sweep/README.md`](../results/lstm_sweep/README.md)）的基础上，用**完全相同的数据流程和训练设置**，给 GRU、1D-CNN、Transformer 做同样的超参数搜索，最后四个模型各取最佳配置、用 5 个种子重训，在测试集上公平比较。

## 新增和修改的文件

| 文件 | 作用 |
|---|---|
| `src/models.py`（新增） | GRU、1D-CNN、Transformer 三个模型，四个模型的注册表和搜索网格。LSTM 直接用原来的 `LSTMSOC`，没有改动 |
| `src/sweep_models.py`（新增） | 统一的超参数搜索脚本，`--model lstm / gru / cnn / transformer`。只用训练集和验证集，测试集保护打开 |
| `src/import_lstm_sweep.py`（新增） | ① `--verify`：用新代码重跑一组旧 LSTM 配置，对照记录值；② 把已跑完的 22 组 LSTM 结果导入统一的搜索表（不重新训练） |
| `src/final_eval.py`（新增） | 最终评估：四个模型各取验证 RMSE 最低的配置，用 seed 42–46 重训，按驾驶工况和总体评估测试集，输出汇总表。**只有这个脚本会读测试集** |
| `src/smoke_test_models.py`（新增） | smoke test：随机张量检查前向形状；用假数据跑通搜索、续跑、导入、最终评估的全流程；检查 LSTM 结果与原来逐位相同 |
| `src/sweep_lstm.py`（修改） | 只把 `train_one` 里的训练循环抽成通用的 `fit()`、验证指标抽成 `val_metrics()`，供新脚本复用。随机数调用顺序没变，LSTM 的结果与原来逐位相同（smoke test 第 3 项会检查） |
| `docs/models.md`（新增） | 本文档 |

没有修改的：`src/lstm_data.py`（读取、划分、重采样、标准化、窗口）、`src/lstm_model.py`、`src/evaluate_lstm.py`、`src/combine_lstm_sweep.py`，以及所有 LSTM 的结果和权重。

## 模型

四个模型的接口相同：输入 `(batch, W, 3)`（标准化后的 voltage、current、temperature），输出 `(batch,)`（窗口最后一个时间点的 SOC，0–1）。输出头都是 `Linear(·, 1)`。

| 模型 | 结构 | 搜索网格 | 组数 | 配置名示例 |
|---|---|---|---|---|
| LSTM | `LSTM(3, H, L)` → 最后一个时间步 → `Linear(H, 1)` | W × H{32, 64, 128} × L{1, 2} | 30 | `W120_H64_L2` |
| GRU | `GRU(3, H, L, batch_first)` → 最后一个时间步 → `Linear(H, 1)`（只把 LSTM 换成 GRU） | W × H{32, 64, 128} × L{1, 2} | 30 | `W120_H64_L2` |
| 1D-CNN | 转置成 `(batch, 3, W)` → N × [`Conv1d(k=3, p=1)` → `BatchNorm1d` → `ReLU`]（每层 F 个滤波器）→ `AdaptiveAvgPool1d(1)` → `Flatten` → `Linear(F, 1)` | W × F{32, 64, 128} × N{2, 3} | 30 | `W120_F64_L3` |
| Transformer | `Linear(3, d_model)` → 正弦位置编码（max_len = 512）→ `TransformerEncoder`（nhead 4，dim_feedforward 4·d_model，dropout 0.1，batch_first）→ 最后一个时间步 → `Linear(d_model, 1)` | W × d_model{32, 64} × L{1, 2} | 20 | `W120_D64_L2` |

窗口长度 W ∈ {30, 60, 120, 180, 240}，即 LSTM 实验最终用到的全部窗口。

LSTM 原来只跑了 22 组（W 为 180、240 时只试了 H{32, 128} × L1）。为了与其他模型的搜索空间一致，LSTM 也补齐到 30 组：已有的 22 组直接导入（不重新训练），只补跑缺少的 8 组。

## 固定设置（四个模型完全相同，与 LSTM 一致）

- 数据：`lstm_data.py` 的同一套流程。验证集是每个温度编号最大的 Mixed 文件（6 个）；训练集是其余 Mixed（37 条序列）；测试集是 UDDS、HWFET、LA92、US06（23 个）。按 1 s 重采样，标准化只用训练集计算。搜索脚本会把算出的标准化参数与 LSTM 实验保存的 `norm_stats.json` 逐一比对，不一致就停止。
- 训练：Adam，lr 0.001（`--lr` 可覆盖），batch 256，MSE loss（SOC 按 0–1），每轮打乱，最多 200 轮。验证 loss 连续 15 轮没有创新低就停止，保存验证 loss 最低那一轮的权重。seed 42，cudnn deterministic。
- 验证误差和早停只在"序列内第 239 行及之后"的点上计算，每个配置都是同样的 41,057 个验证点，与 LSTM 的 22 组表一致。
- RMSE、MAE 的单位是 SOC 百分点（误差 × 100）。
- 推理时间：batch = 1 的单样本延迟，先预热 50 次再计时 300 次取平均，GPU 和 CPU 各测一次（CPU 用 PyTorch 默认线程数，线程数记录在 `cpu_threads` 列）。
- 模型自身的差异（按规格）：LSTM/GRU 的 dropout 为 0，Transformer 的 encoder dropout 为 0.1，CNN 带 BatchNorm。

## 在 Windows 上运行

下面的命令都在 **Anaconda Prompt** 里输入。`D:\battery-fyp` 换成你电脑上仓库的实际位置，`你的环境名` 换成你的 conda 环境名。

> **不要删除或移动 `results\lstm_sweep*\` 和 `models\lstm_sweep*\`。** 导入 LSTM 结果时要读取里面的 22 组权重，导入后的 `sweep_lstm.csv` 里 `model_file` 列指向的也是这些旧权重。

### 第 0 步：切换到新代码，确认 GPU

```bat
conda activate 你的环境名
cd /d D:\battery-fyp
git fetch origin
git checkout feature/gru-cnn-transformer
git pull origin feature/gru-cnn-transformer
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### 第 1 步：smoke test（不到 1 分钟）

```bat
python src\smoke_test_models.py
```

最后一行应当是 `SMOKE TEST PASSED`，前面列出 40 项检查，全部为"通过"。这一步只用随机数和假数据，输出写到临时目录，跑完自动删除。

### 第 2 步：LSTM——核对、导入 22 组、补跑 8 组

```bat
python src\import_lstm_sweep.py --verify
```

- 用**新代码**把 22 组表（`results\lstm_sweep_ext\sweep_results_22.csv`）里验证 RMSE 最低的那组从头重新训练一次。设置与当初完全相同，包括当初早停用的评估起点：最初 18 组是 119，扩展 4 组是 239。
- 日志会列出"记录 vs 重跑"的验证 RMSE、最佳轮、总轮数，以及与旧权重的最大参数差。验证 RMSE 相差在 ±0.05 个百分点以内判为通过；同一台电脑、同样的 PyTorch 版本下一般完全相同，参数差为 0。
- 用时与这组当初的训练时间相同，可以在 22 组表的 `train_time_s` 列里查到。想先快速试一下，可以指定小窗口的配置，例如 `python src\import_lstm_sweep.py --verify W30_H32_L1`。
- 结果写到 `results\lstm_verify\` 和 `models\lstm_verify\`，不碰搜索表和旧结果。
- **未通过**时，先查明原因再导入。常见原因是 PyTorch 或 CUDA 的版本、显卡与当初不同；换过版本导致的微小差异可以用 `--tol` 放宽容差。

核对通过后导入：

```bat
python src\import_lstm_sweep.py
```

- 把 22 组写进 `results\tables\sweep_lstm.csv`，数值直接取自 22 组表，不重新训练。
- 每组用旧权重补测推理时间（GPU、CPU 各一次），大约一两分钟。
- 没有通过核对时会拒绝导入。可以重复运行，已导入的配置会跳过。

然后补跑缺少的 8 组（W180/W240 × H64 × L1/L2，以及 W180/W240 × H{32, 128} × L2）：

```bat
python src\sweep_models.py --model lstm
```

### 第 3 步：GRU、1D-CNN、Transformer 的搜索

```bat
python src\sweep_models.py --model gru
python src\sweep_models.py --model cnn
python src\sweep_models.py --model transformer
```

- 每跑完一组，就往 `results\tables\sweep_<模型>.csv` 追加一行。
- 可以随时按 `Ctrl+C` 中断，之后输入**同一条命令**就会续跑：CSV 里已有的配置直接跳过，中断时正在跑的那一组从第 1 轮重新开始。
- 想分几次跑，可以只跑部分窗口：

  ```bat
  python src\sweep_models.py --model gru --windows 30 60 120
  python src\sweep_models.py --model gru --windows 180 240
  ```

- 只跑某一组：`python src\sweep_models.py --model cnn --only W240_F64_L3`。

### （可选）Transformer 加试 lr = 0.0005

```bat
python src\sweep_models.py --model transformer --lr 0.0005
```

结果另存为 `results\tables\sweep_transformer_lr0.0005.csv` 和 `models\transformer_lr0.0005\`，不会和 lr = 0.001 的结果混在一起。只想试部分配置时，同样可以加 `--windows` 或 `--only`。

### 第 4 步：最终评估（会读测试集，最后一步再运行）

```bat
python src\final_eval.py
```

1. **选配置（只看验证集）**：每个模型从自己的搜索表里取验证 RMSE 最低的一组。
   - 默认要求 `sweep_<模型>.csv` 已经跑完完整网格，否则报错并列出缺哪几组。确实要提前评估可以加 `--allow-incomplete`。
   - 如果跑过 `--lr 0.0005`，`sweep_transformer_lr0.0005.csv` 也会一起参与比较，学习率算作被选出的超参数之一。只想用 lr = 0.001 的结果，加 `--pick transformer=results/tables/sweep_transformer.csv`。
2. **重训**：每个最佳配置用 seed 42、43、44、45、46 各训练一次，共 20 次。流程与搜索完全相同，早停只看验证集；这一阶段测试集保护是打开的。
3. **测试**：20 次训练全部结束之后才读测试集。每个测试文件评估第 239 行及之后的点，按 UDDS、HWFET、LA92、US06 分别计算（6 个温度的测试点合在一起），再计算全部测试点的总体结果。
4. **汇总**：每个模型测试 RMSE、MAE 的均值 ± 标准差（5 个种子，样本标准差），加上参数量和推理时间。推理时间用 seed 42 的权重，四个模型在同一次运行里连续测量。

中断后重跑同一条命令即可：已训练的（模型, 种子）直接读权重，不重新训练。只评估部分模型：`python src\final_eval.py --models gru cnn`。

用时大约是四个最佳配置在搜索表里的 `train_time_s` 之和 × 5。

## 输出文件

### 搜索：`results/tables/sweep_<模型>.csv`

每组一行，列如下：

| 列 | 含义 |
|---|---|
| `model`、`config`、`window` | 模型、配置名、窗口长度 |
| `hidden_size`、`num_layers` / `filters`、`num_layers` / `d_model`、`num_layers` | 结构超参数（LSTM、GRU / CNN / Transformer） |
| `lr`、`batch_size`、`seed`、`max_epochs`、`patience`、`loss` | 训练设置 |
| `eval_start` | 验证指标的评估起点（239） |
| `early_stop_eval_start` | 早停用的评估起点。新跑的都是 239；从最初 18 组导入的 LSTM 是 119 |
| `val_rmse_pct`、`val_mae_pct` | 验证集 RMSE、MAE（SOC 百分点），选配置的依据 |
| `val_rmse_pct_<温度>` | 每个温度（即每个验证文件）的 RMSE |
| `best_epoch`、`epochs_trained`、`stopped_early` | 最佳轮、总轮数、是否早停 |
| `train_time_s` | 训练总时间（秒，含每轮验证） |
| `n_params` | 可训练参数量 |
| `infer_ms_gpu`、`infer_ms_cpu` | 单样本（batch = 1）平均推理时间，毫秒；没有 GPU 时 `infer_ms_gpu` 为空 |
| `infer_gpu_name`、`cpu_threads` | 测推理时间用的显卡、CPU 线程数 |
| `n_train_windows`、`n_val_points` | 训练窗口数、验证点数 |
| `device_name` | 训练用的设备 |
| `source` | `sweep_models` 表示本脚本训练；`imported:lstm_sweep` 或 `imported:lstm_sweep_ext` 表示从旧 LSTM 结果导入 |
| `model_file` | 权重路径 |
| `finished_at` | 完成时间 |

其他搜索输出：

| 路径 | 内容 |
|---|---|
| `models/<模型>/<配置>.pt` | 验证 loss 最低那一轮的权重，里面同时存了配置、标准化参数和训练设置 |
| `results/search/<模型>/sweep_log.txt` | 运行日志，多次运行接着往后写 |
| `results/search/<模型>/norm_stats.json` | 标准化参数 |
| `results/search/<模型>/history/<配置>.csv` | 每轮的 train/val loss、验证 RMSE、用时 |
| `results/search/<模型>/curves/<配置>.png` | 训练曲线 |
| `results/search/lstm/import_log.txt` | LSTM 导入日志 |
| `results/lstm_verify/<配置>.json`、`verify_log.txt` | LSTM 核对结果 |

### 最终评估

| 路径 | 内容 |
|---|---|
| `results/tables/final_summary.csv` | 每个模型一行：配置、lr、参数量、推理时间（GPU/CPU），以及 `test_rmse_pct_<范围>_mean/_std`、`test_mae_pct_<范围>_mean/_std`，范围是 `all`、`UDDS`、`HWFET`、`LA92`、`US06` |
| `results/tables/final_summary_formatted.csv` | 同上，写成"均值 ± 标准差"的文字，方便贴进论文 |
| `results/tables/final_runs.csv` | 每个（模型, 种子）一行：验证指标、最佳轮、用时、各工况和总体的测试 RMSE/MAE |
| `results/final_eval/per_file_metrics.csv` | 每个（模型, 种子, 测试文件）一行，可以再按温度汇总 |
| `results/final_eval/final_log.txt` | 运行日志 |
| `results/final_eval/runs/` | 每次训练的记录（JSON）和每轮 loss |
| `models/final/<模型>/<配置>_seed<种子>.pt` | 每次训练的最佳权重 |

## 注意事项

- **测试集只在第 4 步用到。** `sweep_models.py` 和 `import_lstm_sweep.py` 在开始时就打开测试集保护，读到任何测试文件都会直接报错；smoke test 也会检查这两个脚本里没有关闭保护的代码。
- **LSTM 最初 18 组的早停起点是 119**，其余配置都是 239。这 18 组的验证误差已经在 239 上重新计算过（`combine_lstm_sweep.py`），所以放在同一张表里比较是公平的，只是保存下来的"最佳轮"是按 119 选的，影响很小（README 里已说明）。最终评估的重训统一用 239。
- **写论文时推理时间用 `final_summary.csv` 里的数字**：那是四个模型在同一次运行里连续测的。搜索表里的推理时间是各自搜索时测的，可能受当时机器负载影响。
- 脚本设置了 `cudnn.deterministic`，同一台电脑上 LSTM、GRU 的结果通常可以逐位复现。Transformer 反向传播里个别 CUDA 运算可能有极小的不确定性，所以最终评估里 seed 42 的重训与搜索阶段的结果可能有极小差异。这不影响比较，多种子的标准差已经反映了这类波动。
- `.gitignore` 忽略所有 `*.pt` 权重文件；`results/` 下的 CSV 不会被自动忽略，要不要提交由你决定。
