# LSTM SOC 估计：完整流程讲解

这份文档一步步讲 `src/` 里 LSTM SOC 估计的完整流程：从清洗好的 CSV 开始，经过数据划分、重采样、标准化、切窗口、模型、训练、合并结果，最后到测试集评估。写报告时可以直接参考。

- 代码位置统一写成 `文件名:行号`，对应的版本是 `master` 分支的 `be7b363`（PR #2 合并之后）。
- 专有名词和代码保留英文。术语第一次出现时，会先用一句大白话解释。
- 文中的"真实数字"有两个来源：
  1. `results/lstm_sweep/README.md` 里记录的数字，例如窗口数、验证点数；
  2. 我按照代码的逻辑、**只读取训练集文件**重新算出来的数字，例如 mean 和 std。

  写这份文档时，仓库里没有训练好的 `.pt` 文件和 `runs/*.json`，也没有读取任何测试集文件，也没有运行训练或 `evaluate_lstm.py`。

---

## 目录

- [阶段 0：全局地图](#阶段-0全局地图)
- [阶段 1：用哪些数据](#阶段-1用哪些数据)
- [阶段 2：划分训练集、验证集、测试集](#阶段-2划分训练集验证集测试集)
- [阶段 3：重采样到 1 秒](#阶段-3重采样到-1-秒)
- [阶段 4：标准化](#阶段-4标准化)
- [阶段 5：切窗口](#阶段-5切窗口)
- [阶段 6：模型结构](#阶段-6模型结构)
- [阶段 7：训练](#阶段-7训练sweep_lstmpy)
- [阶段 8：合并结果](#阶段-8合并结果combine_lstm_sweeppy)
- [阶段 9：测试集评估](#阶段-9测试集评估evaluate_lstmpy)
- [阶段 10：总结](#阶段-10总结)

---

## 阶段 0：全局地图

### 0.1 更早的阶段：`preprocess_kollmeyer.py`

`src/preprocess_kollmeyer.py` 负责数据清洗。它把 208 个原始 CSV 清洗成 102 条独立的时间序列，保存在 `data/processed/kollmeyer_clean/<温度>/xxx.csv`，列是 `time_s, voltage, current, temperature, soc, ah`。同时它还生成 `manifest.json`，这是一张"目录卡片"，记录了每个文件的温度、测试类型、是否被排除等信息。清洗阶段**不划分数据集、不重采样、不标准化**（`DATA_REPORT.md` 第 22 行），这几件事都留给 LSTM 阶段来做。

### 0.2 涉及的 5 个文件

| 文件 | 负责什么 | 主要内容 |
|---|---|---|
| `src/lstm_data.py` | 数据：筛选、划分、读取、重采样、标准化、切窗口、测试集保护 | `build_splits`（`:138`）、`read_raw`（`:226`）、`resample_1s`（`:240`）、`load_sequences`（`:249`）、`compute_norm_stats`（`:263`）、`normalize`（`:288`）、`WindowDataset`（`:297`）、`block_test_files`（`:78`） |
| `src/lstm_model.py` | 模型 | `LSTMSOC`（`:9`）、`count_parameters`（`:22`） |
| `src/sweep_lstm.py` | 训练多组超参数（sweep），同时也是给后面两个脚本用的工具箱 | `train_one`（`:143`）、`main`（`:287`）、`predict`（`:131`）、`metrics_pct`（`:136`） |
| `src/combine_lstm_sweep.py` | 把 18 + 4 = 22 组放在同一批验证点上重新计算并合并 | `main`（`:55`）、`per_temperature`（`:45`） |
| `src/evaluate_lstm.py` | 选定配置之后，在测试集上评估一次 | `main`（`:66`）、`plot_file`（`:45`） |

### 0.3 调用关系

```
                  lstm_data.py        lstm_model.py
              （底层，不 import 项目里的其他文件）
                    ▲    ▲    ▲         ▲    ▲    ▲
          ┌─────────┘    │    └─────┐   │    │    │
   sweep_lstm.py ────────┼──────────┼───┘    │    │
     ▲         ▲         │          │        │    │
     │   combine_lstm_sweep.py ─────┼────────┘    │
   evaluate_lstm.py ────────────────┘─────────────┘
```

- `sweep_lstm.py:48-49`：import `lstm_data`（别名 `D`）和 `lstm_model`
- `combine_lstm_sweep.py:34-38`：import `lstm_data`、`lstm_model`，并从 `sweep_lstm` 借用 `predict`、`metrics_pct`、`config_name`、`Logger`、`pick_device` 和网格常量
- `evaluate_lstm.py:31-34`：import `lstm_data`、`lstm_model`，并从 `sweep_lstm` 借用 `predict`、`metrics_pct`、`config_name`、`pick_device` 和配色常量

三个脚本共用同一个 `predict` 和 `metrics_pct`，所以误差的计算方式完全一致。

### 0.4 流程图

```
data/processed/kollmeyer_clean/
  ├─ manifest.json ─────────────┐
  └─ <温度>/*.csv（102 条序列）  │
                                ▼
 lstm_data.build_splits()          只读 manifest，不读数据
   筛选 test_type ∈ {mixed, drive} 且 excluded == False → 66 条
   划分 train 37 / val 6 / test 23（按整个文件分）
                                │
   （sweep 和 combine 在这里调用 block_test_files() 打开测试集保护）
                                ▼
 load_sequences() = read_raw() + resample_1s()
   读 CSV（不读 ah 列）→ 按 floor(time_s) 合并成每秒一行
                                ▼
 compute_norm_stats(训练集) → mean / std
   → results/<批次>/norm_stats.json，同时写进每个 .pt
                                ▼
 WindowDataset(seqs, stats, W, min_end_index)
   标准化 → 长度为 W 的滑动窗口（取样时才切）
   输入 (W, 3) = V, I, T；标签 = 窗口最后一秒的 soc
                                ▼
 sweep_lstm.py：对每组 (W, H, L) 调用 train_one()
   Adam + MSE + early stopping，保存验证 loss 最低的那一轮
   18 组 → results/lstm_sweep/、models/lstm_sweep/
   4 组（--ext）→ results/lstm_sweep_ext/、models/lstm_sweep_ext/
                                ▼
 combine_lstm_sweep.py：读回 22 个 .pt，在验证集第 239 行及以后
   （41,057 个点）上重新计算 → results/lstm_sweep_ext/sweep_results_22.csv
                                ▼
                  只根据验证集结果，人工选定一组配置
                                ▼
 evaluate_lstm.py --window W --hidden H --layers L
   第一次、也是唯一一次读取 23 个测试文件
   → results/<批次>/test_eval/<配置>_start239/
      rmse_table.csv、mae_table.csv（温度 × 工况）、per_file_metrics.csv、
      summary.json、predictions/、plots/
```

---

## 阶段 1：用哪些数据

### 1.1 `manifest.json`

`manifest.json` 里一共有 231 条记录：主数据集 102 条，`_excluded/` 里 129 条。这一步用到的字段只有几个，以训练文件 `25degC/551_Mixed1` 为例：

```json
{ "id": "25degC/551_Mixed1", "file": "25degC/551_Mixed1.csv", "excluded": false,
  "folder": "25degC", "nominal_temperature_C": 25, "test_type": "mixed",
  "test_name": "Mixed1", "part": 1, "n_parts": 1, "sampling_interval_median_s": 0.1 }
```

### 1.2 筛选代码：`lstm_data.py:148-153`

```python
entries = [
    e for e in manifest["series"]
    if not e["excluded"] and e["folder"] in folders
    and e["test_type"] in ("mixed", "drive")
]
```

一条记录必须同时满足三个条件：没有被排除、在要用的温度文件夹里、类型是 Mixed 或 drive（单一驾驶工况 UDDS / HWFET / LA92 / US06）。

- **输入**：231 条记录。
- **输出**：66 条（Mixed 43 条，drive 23 条）。

这一步**只读 manifest，不打开任何 CSV**（`lstm_data.py:141`）。

之所以按 `test_type` 筛选、不按文件名猜（`lstm_data.py:7-8`），是因为文件名靠不住。比如从 `PausCycl` 这个名字看不出它是 UDDS 的前 18-36 分钟（D26）；有些文件还会被拆成 `_part1` 和 `_part2`。manifest 里的类型在清洗阶段做过工况曲线指纹比对（D42）。另外还有两道自检：`lstm_data.py:175-176` 检查 drive 文件的名字必须是 4 种工况之一；`lstm_data.py:131-135` 检查 Mixed 文件的名字必须严格符合 `Mixed<数字>` 的格式。

### 1.3 没有使用的测试类型

主数据集里有 36 条序列没有被 LSTM 使用：

| 测试类型 | 条数 | 采样间隔 | 例子 |
|---|---|---|---|
| Cap_1C | 8 | 10 s | `25degC/551_Cap_1C`，3,921 s |
| C20DisCh | 6 | 60 s | `25degC/549_C20DisCh`，145,117 s |
| Dis_0p5C / Dis_2C | 6 + 6 | 60 s | `25degC/549_Dis_2C`，1,549 s |
| HPPC | 6 | 0.1/1/10 s 混合 | `25degC/549_HPPC` |
| PausCycl | 4 | 0.1 s | `n20degC/611_PausCycl`，1,661 s |

代码里只写了"只用 drive 和 Mixed"这条规则。下面的理由是根据 `DATA_REPORT.md` 里的事实整理出来的：

1. **任务目标不同。** 这个 LSTM 要估计的是动态驾驶工况下的 SOC。恒流测试和 HPPC 的电流形状和驾驶工况差别很大，保留它们原本是为了 EKF 辨识参数（D24）。
2. **时间尺度不同。** 10 s 或 60 s 采样的数据，重采样后仍然是 10 s 或 60 s 一行。同样是 W=240 行，在驾驶工况里代表 4 分钟，在 C/20 里却代表 4 小时。
3. **标签分布不同。** 特性测试在低温下 SOC 会是负值（D12，−20 °C 的 C/20 末尾是 −47 %）。
4. **泄露风险。** PausCycl 就是 UDDS 的前 18-36 分钟（D26），而 UDDS 属于测试集。

`_excluded/` 里另外还有 4 条 drive 记录：`551_HWFET` 和 `582_LA92_attempt1_part1-3`。它们因为 `excluded` 为真，不会进入测试集，但会被 `block_test_files()` 一起封锁（见 2.6）。

---

## 阶段 2：划分训练集、验证集、测试集

先说明三个集合的用途：

- **训练集**：模型用它来学习，也就是更新参数。
- **验证集**：训练过程中的"模拟考"，用来决定什么时候停止训练、哪一组超参数最好。
- **测试集**：全部选择都做完之后才用，只用一次，用来报告最终成绩。

### 2.1 位置

`lstm_data.py:138-209` 的 `build_splits(folders)`。

- **输入**：温度文件夹列表，以及 manifest。
- **输出**：`{"train": [...], "val": [...], "test": [...]}`，每个元素是一个 `SeqInfo`（`lstm_data.py:104-118`）。

这个函数只读 manifest。第 188 行的 `info.path.exists()` 只检查文件是否存在，不会打开文件。

### 2.2 规则（`lstm_data.py:155-190`）

- **A. 所有 drive 文件 → 测试集**（`:174-177`）
- **B. 每个温度下编号最大的 Mixed 文件 → 验证集**（`:156-170`）
  - 如果编号最大的那个文件被拆成了多段，直接报错，交给人来决定（`:163-164`）。
  - 选出来的文件必须和写死的 `EXPECTED_VAL_IDS`（`:48-55`）一致，否则报错（`:165-169`）。
- **C. 其余 Mixed → 训练集**（`:180-181`）

### 2.3 完整清单

括号里是清洗后 CSV 的原始行数（0.1 s 一行）。重采样到 1 s 后，行数大约变成原来的 1/10。

| 温度 | 训练集（37 条） | 验证集（6 条） | 测试集（23 条） |
|---|---|---|---|
| −20 °C `n20degC` | `610_Mixed1`(45,342)、`610_Mixed2`(43,043)、`611_Mixed3`(45,163)、`611_Mixed4`(46,052)、`611_Mixed5`(45,457)、`611_Mixed6`(44,195)、`611_Mixed7`(40,682) | `611_Mixed8`(46,762) | `610_UDDS`、`610_HWFET`、`610_LA92`、`610_US06` |
| −10 °C `n10degC` | `601_Mixed1`(62,621)、`601_Mixed2`(65,253)、`604_Mixed3`(55,214)、`602_Mixed4_part1`(3,868)、`602_Mixed4_part2`(57,649)、`602_Mixed5`(60,086)、`604_Mixed6`(55,967)、`604_Mixed7`(55,939) | `604_Mixed8`(67,325) | `596_UDDS`、`596_HWFET`、`596_LA92`、`601_US06` |
| 0 °C `0degC` | `589_Mixed1`(67,311)、`589_Mixed2`(72,303)、`590_Mixed4`(68,026)、`590_Mixed5`(64,994)、`590_Mixed6`(62,483)、`590_Mixed7`(61,539) | `590_Mixed8`(75,838) | `589_UDDS`、`589_HWFET`、`589_LA92`、`589_US06` |
| 10 °C `10degC` | `567_Mixed1`(72,328)、`567_Mixed2`(77,751)、`571_Mixed4`(73,928)、`571_Mixed5`(67,173)、`571_Mixed6_part1`(391)、`571_Mixed6_part2`(73,529)、`571_Mixed7`(69,324) | `571_Mixed8`(81,422) | `576_UDDS`、`576_HWFET`、`582_LA92`、`567_US06` |
| 25 °C `25degC` | `551_Mixed1`(77,226)、`551_Mixed2`(79,171)、`552_Mixed3`(73,826)、`552_Mixed4`(80,850)、`552_Mixed5`(72,184)、`552_Mixed6`(78,105)、`552_Mixed7`(73,234) | `552_Mixed8`(85,915) | `551_UDDS`、`551_LA92`、`551_US06`（没有 HWFET） |
| 40 °C `40degC` | `556_Mixed1`(75,341)、`556_Mixed2`(78,955) | `557_Mixed3`(67,625) | `556_UDDS`、`556_HWFET`、`556_LA92`、`556_US06` |

几点说明：

- 0 °C 和 10 °C 没有 Mixed3（D42）。
- 40 °C 的 `562_Mixed4-8` 被排除了（D29），所以 40 °C 的验证文件是 `557_Mixed3`，训练集只有 2 条。这是 6 个温度里最少的，写报告时应当指出。
- 25 °C 没有 HWFET（`MISSING_DRIVE`，`lstm_data.py:57`）。`lstm_data.py:192-197` 会逐个温度检查测试集是不是"4 个工况各一个"。

### 2.4 为什么按整个文件划分

1. **相邻窗口几乎一模一样。** 窗口的步长是 1 秒。W=240 时，相邻两个窗口有 239/240 ≈ 99.6 % 的数据是重叠的。如果把所有窗口随机打乱再划分，几乎每个验证窗口在训练集里都有一个"只差一秒"的双胞胎，验证成绩就会虚高。这就是数据泄露。
2. **符合真实使用场景。** BMS 实际面对的，是一整段从来没见过的行驶过程。
3. **测试集用的是另一类工况。** 训练和验证用 Mixed，测试用 4 种单一工况，这样还能检验模型的泛化能力。

另外，验证集用的是一条事先定好的机械规则（"编号最大"），而且每个温度各有一个验证文件。这样可以避免人为挑选验证文件。

### 2.5 不重叠检查（`lstm_data.py:199-204`）

这里检查两层：一是按序列 id 检查；二是把 `_part` 后缀去掉、还原成原始文件名之后再检查。所以同一个原始文件拆出来的几段，不可能分到不同的集合里。

### 2.6 测试集保护

- **黑名单和开关**：`lstm_data.py:74-90` 的 `block_test_files()`。黑名单直接从 manifest 里挑出**所有** `test_type == "drive"` 的文件，包括 `_excluded/` 里的，一共 27 个路径。它不依赖划分的结果。
- **拦截**：`lstm_data.py:93-98` 的 `_check_not_test()`，以及 `read_raw()` 里的两道检查（`lstm_data.py:229-231`：先按路径查，再按 `split == "test"` 查）。所有数据读取都要经过 `read_raw()`。
- **打开保护并自检**：`sweep_lstm.py:353-359`。先打开保护，然后故意去读一个测试文件；如果没有报错，就抛出 `AssertionError("测试集保护没有生效")`。`combine_lstm_sweep.py:76` 和 `:139-144` 用的是同样的做法。
- 只有 `evaluate_lstm.py` 不打开保护。

局限：保护只能拦住"通过 `read_raw()` 去读"这一条路，拦不住别处直接调用 `pd.read_csv`。另外，`python src/lstm_data.py`（`:375-414`）会读取测试文件来统计行数（`:17` 有说明）。

---

## 阶段 3：重采样到 1 秒

### 3.1 调用链

```
load_sequences(infos)              lstm_data.py:249-257
  ├─ read_raw(info)                lstm_data.py:226-237
  ├─ resample_1s(df)               lstm_data.py:240-246
  ├─ 检查 NaN                      lstm_data.py:253-254
  └─ Sequence(info, sec, x, y)     lstm_data.py:215-223
```

`read_raw()` 做了三件事：

1. 测试集保护（`:229-231`）。
2. 只读 `time_s, voltage, current, temperature, soc` 这 5 列，`ah` 列根本不读进来（`:232-234`）。原因是 `ah` 是 SOC 标签的来源（D8）。
3. 检查 `time_s` 是否严格递增（`:235-236`）。

`resample_1s()`：

```python
sec = np.floor(df["time_s"].to_numpy()).astype(np.int64)   # :242  每一行属于第几秒
g = df.groupby(sec, sort=True)                              # :243  按秒分组
x = g[FEATURES].mean().to_numpy(dtype=np.float64)          # :244  V、I、T 取平均
y = g[TARGET].last().to_numpy(dtype=np.float64)            # :245  soc 取这一秒的最后一行
return g.size().index.to_numpy(dtype=np.int64), x, y       # :246
```

- **输入**：0.1 s 一行的 DataFrame。
- **输出**：`sec (n,)`、`x (n, 3)`（未标准化）、`y (n,)`。

### 3.2 例子：`25degC/551_Mixed1.csv` 的前 3 秒

**重采样前（30 行）**

| 行 | time_s | voltage | current | temperature | soc |
|---|---|---|---|---|---|
| 0 | 0.000 | 4.18464 | −0.05108 | 23.97615 | 1.000000 |
| 1 | 0.100 | 4.18396 | −0.07918 | 23.97615 | 1.000000 |
| 2 | 0.201 | 4.18363 | −0.08939 | 23.97615 | 0.999996 |
| 3 | 0.301 | 4.18363 | −0.09195 | 23.97615 | 0.999996 |
| 4 | 0.402 | 4.18346 | −0.09195 | 23.97615 | 0.999996 |
| 5 | 0.501 | 4.18346 | −0.09450 | 23.97615 | 0.999996 |
| 6 | 0.601 | 4.18346 | −0.09450 | 23.97615 | 0.999993 |
| 7 | 0.703 | 4.18346 | −0.09450 | 23.97615 | 0.999993 |
| 8 | 0.803 | 4.18346 | −0.09706 | 23.97615 | 0.999993 |
| 9 | 0.902 | 4.18329 | −0.09706 | 23.97615 | 0.999993 |
| 10 | 1.005 | 4.18329 | −0.09450 | 23.97615 | 0.999989 |
| 11 | 1.100 | 4.18329 | −0.09706 | 23.97615 | 0.999989 |
| 12 | 1.203 | 4.18329 | −0.09706 | 23.97615 | 0.999989 |
| 13 | 1.299 | 4.18329 | −0.09450 | 23.97615 | 0.999989 |
| 14 | 1.401 | 4.18329 | −0.09450 | 23.97615 | 0.999985 |
| 15 | 1.501 | 4.18329 | −0.09706 | 23.97615 | 0.999985 |
| 16 | 1.601 | 4.18329 | −0.09706 | 23.97615 | 0.999985 |
| 17 | 1.702 | 4.18329 | −0.09706 | 23.97615 | 0.999985 |
| 18 | 1.802 | 4.18329 | −0.09450 | 23.97615 | 0.999982 |
| 19 | 1.902 | 4.18312 | −0.09450 | 23.97615 | 0.999982 |
| 20 | 2.000 | 4.18312 | −0.09706 | 23.97615 | 0.999982 |
| 21 | 2.100 | 4.18312 | −0.09706 | 23.97615 | 0.999978 |
| 22 | 2.200 | 4.18312 | −0.09450 | 23.97615 | 0.999978 |
| 23 | 2.302 | 4.18312 | −0.09706 | 23.97615 | 0.999978 |
| 24 | 2.400 | 4.18312 | −0.09706 | 23.97615 | 0.999978 |
| 25 | 2.500 | 4.18312 | −0.09450 | 23.97615 | 0.999974 |
| 26 | 2.601 | 4.18312 | −0.09706 | 23.97615 | 0.999974 |
| 27 | 2.701 | 4.18312 | −0.09706 | 23.97615 | 0.999974 |
| 28 | 2.801 | 4.18312 | −0.09706 | 23.97615 | 0.999974 |
| 29 | 2.903 | 4.18312 | −0.09706 | 23.97615 | 0.999971 |

**重采样后（3 行）**

| sec | voltage（平均） | current（平均） | temperature（平均） | soc（最后一行） | 这一秒的行数 |
|---|---|---|---|---|---|
| 0 | 4.183645 | −0.088117 | 23.97615 | 0.999993（第 9 行） | 10 |
| 1 | 4.183273 | −0.095780 | 23.97615 | 0.999982（第 19 行） | 10 |
| 2 | 4.183120 | −0.096548 | 23.97615 | 0.999971（第 29 行） | 10 |

整个文件从 77,226 行变成 7,723 行（第 0 到第 7722 秒），中间没有缺失的秒。

大多数秒有 10 行，但有 663 秒是 11 行、662 秒是 9 行，这是采样时间有微小抖动造成的。

### 3.3 为什么这样做

- **为什么降到 1 s**：数据量减少到约 1/10；同样的 W，窗口覆盖的时间是原来的 10 倍（W=240 就是 4 分钟）。
- **为什么 V、I、T 取平均**：可以降低测量噪声，同时保留这一秒的平均电流。
- **为什么 soc 取最后一行**：标签的含义是"这一秒结束时的 SOC"，它和输入的因果关系是对的——先有这一秒的电流，才有这一秒末的 SOC。这一点是我的理解，代码里没有写注释。
- **为什么每条序列单独重采样**：窗口不跨序列，`_part1` 和 `_part2` 之间的断档也不能拼接。

需要注意：重采样**不做插值**。如果原始数据里有空隙（例如 D38 提到的 LA92 每跑完一遍有 1.4-2.1 s 的空隙），重采样后就会少一行，但 `sec` 数组里保留了真实的秒数。

---

## 阶段 4：标准化

**标准化**：`x_标准化 = (x − mean) / std`，目的是把三个量级差别很大的输入统一到均值约为 0、标准差约为 1。

### 4.1 计算：`lstm_data.py:263-274` 的 `compute_norm_stats()`

- `:264-265`：只要有一条序列不属于训练集，就报错。
- `:266`：把 37 条训练序列拼接起来。
- `:269-270`：按列计算 mean 和 std。`np.std` 默认 `ddof=0`，也就是总体标准差。
- 还会记录 `n_rows` 和 `computed_from`（用了哪些序列）。

这个函数在 `sweep_lstm.py:368` 被调用。

### 4.2 真实数值

这些数值是只读训练集的 37 个文件、按相同逻辑重新算出来的：

| 特征 | mean | std |
|---|---|---|
| voltage (V) | 3.6465658 | 0.2843751 |
| current (A) | −1.2345516 | 2.4040019 |
| temperature (°C) | 6.2353234 | 16.8460813 |

`n_rows = 224,664`。可以用它交叉验证：224,664 − 37 × 29 = 223,591，正好等于 README 里记录的 W=30 训练窗口数。

举个例子：`551_Mixed1` 第 0 秒是 `(4.183645, −0.088117, 23.97615)`，标准化后变成 `(1.889, 0.477, 1.053)`。

### 4.3 存放位置

1. `results/<批次>/norm_stats.json`（`sweep_lstm.py:369-378`）。如果文件已经存在，会检查新算出的值和文件里的值是否一致，不一致就停止。
2. 每个 `.pt` 文件里的 `"norm_stats"` 字段（`sweep_lstm.py:206-209`）。这样模型和它的标准化参数永远在一起。

### 4.4 验证集和测试集怎么套用

标准化只发生在一个地方：`normalize()`（`lstm_data.py:288-291`）。它只在 `WindowDataset.__init__` 里被调用（`lstm_data.py:323`）。

| 场景 | 代码 | 用的 stats |
|---|---|---|
| 训练集 | `sweep_lstm.py:146` | `sweep_lstm.py:368` 用训练集算出的 |
| 验证集 | `sweep_lstm.py:147` | 同一个 |
| 合并阶段 | `combine_lstm_sweep.py:158` | 从 `.pt` 里读出的 `ref_stats`（`:110`） |
| 测试集 | `evaluate_lstm.py:121` | 从 `.pt` 里读出的 `ckpt["norm_stats"]`（`:89`） |

一致性检查：`evaluate_lstm.py:92-96` 检查 `.pt` 和 json 是否一致；`combine_lstm_sweep.py:110-127` 检查 22 个 `.pt` 是否两两一致，并且和 json 一致。

**为什么只用训练集**：如果用全部数据来算 mean 和 std，测试集的统计信息就会通过这两个数漏进训练过程。

---

## 阶段 5：切窗口

**窗口**：连续 W 秒的历史数据。**滑动窗口，步长 1**：窗口每次往后移 1 秒。

### 5.1 一个样本

- **输入**：`(W, 3)`，标准化后的 `[V, I, T]`。
- **标签**：窗口最后一秒的 soc（`lstm_data.py:298-306`）。

### 5.2 W=240 的例子（`551_Mixed1`，7,723 行）

| 样本 | ends | 输入覆盖的行 | 标签 |
|---|---|---|---|
| 第 1 个 | 239 | 0-239 | `y[239]` = 0.988171 |
| 第 2 个 | 240 | 1-240 | `y[240]` = 0.988219 |
| 最后一个 | 7722 | 7483-7722 | `y[7722]` = 0.049982 |

这条序列一共能切出 7,484 个训练窗口。

### 5.3 "即时"切片

如果把 W=240 的训练窗口全部预先展开，大约需要 216,021 × 240 × 3 × 4 B ≈ 622 MB。代码的做法是只存一份原始数组和一个终点索引数组：

- `__init__`（`:308-341`）
  - 每条序列单独处理，标准化（`:323`）。
  - 合法的终点是 `loc = arange(first_end, n)`（`:325`），加上 `offset` 之后换算成全局编号（`:326`）。
  - 全部序列拼接成 `self.x`（`:336`）。
  - `self._rel = [-W+1, …, 0]`（`:341`）。
- `__getitem__`（`:346-348`）：`x[e-W+1 : e+1], y[e]`。
- `get_batch`（`:350-355`）：`rows = e[:, None] + _rel[None, :]`，用广播得到 `(batch, W)` 个行号，一次取出 `(batch, W, 3)`。
- `iter_batches`（`:364-369`）：训练时每一轮用 `randperm` 打乱，评估时按顺序。

`offset` 保证窗口不会跨序列。以 `551_Mixed1`（7,723 行）和 `551_Mixed2`（7,917 行）为例，W=240 时，Mixed2 的第一个终点是 7723 + 239 = 7962。

### 5.4 比窗口还短的序列

直接跳过，并记录下来（`:318-320`，日志在 `:330-332`，结果 JSON 里的字段是 `skipped_short_train_sequences`）。

| 序列 | 1 s 行数 | W=30 | W=60/120/180 | W=240 |
|---|---|---|---|---|
| `10degC/571_Mixed6_part1` | 39 | 保留（10 个窗口） | 跳过 | 跳过 |
| `n10degC/602_Mixed4_part1` | 387 | 保留 | 保留 | 保留（148 个窗口） |

用这个规则可以算出 W=240 的训练窗口数：(224,664 − 39) − 36 × 239 = 216,021，和 README 一致。

| W | 训练窗口 | 验证点（起点 239） |
|---|---|---|
| 30 | 223,591 | 41,057 |
| 60 | 222,501 | 41,057 |
| 120 | 220,341 | 41,057 |
| 180 | 218,181 | 41,057 |
| 240 | 216,021 | 41,057 |

### 5.5 评估起点 239

**问题**：如果不统一，W=30 从第 29 行就开始打分，W=240 要从第 239 行才开始，两者打分的点不一样，没法公平比较。

**做法**：验证集和测试集都只在"序列内行号 ≥ 239"的点上打分。

- 默认值：`lstm_data.py:63` 的 `EVAL_START_INDEX = 239`（第 59-64 行的注释说明了它的由来）。
- 生效位置：`lstm_data.py:311`，`first_end = max(W−1, min_end_index)`。训练集传 `None`，所以用 `W−1`；验证集和测试集传 239。
- 防呆检查：`sweep_lstm.py:324-327`、`combine_lstm_sweep.py:132-134`。
- 测试集：`evaluate_lstm.py:103`，`eval_start = max(args.eval_start, window − 1)`。
- 实际数字：起点 239 时有 41,057 个验证点；起点 119 时有 41,777 个。两者相差 6 × 120 = 720。

这意味着每个验证文件和测试文件的前 4 分钟都不打分。

---

## 阶段 6：模型结构

### 6.1 `lstm_model.py` 逐行说明

| 行 | 代码 | 说明 |
|---|---|---|
| 9 | `class LSTMSOC(nn.Module)` | PyTorch 模型都要继承 `nn.Module` |
| 10 | `__init__(hidden_size, num_layers, input_size=3)` | H、L 和输入特征数。W 不是模型的参数 |
| 11 | `super().__init__()` | 初始化父类，必须写 |
| 12-13 | `nn.LSTM(..., batch_first=True, dropout=0.0)` | 输入形状是 `(batch, 时间, 特征)`。dropout 只作用在两层 LSTM 之间，这里设为 0 |
| 14 | `nn.Linear(hidden_size, 1)` | 把 H 维向量变成 1 个 SOC 值 |
| 16-17 | `forward(x)`，`(batch, W, 3) -> (batch,)` | 前向传播 |
| 18 | `out, _ = self.lstm(x)` | `out` 是每个时间步最上一层的 h，形状 `(batch, W, H)`。初始状态为 0，窗口之间不传递记忆 |
| 19 | `self.fc(out[:, -1, :]).squeeze(-1)` | 取最后一个时间步 → Linear → 去掉长度为 1 的维度。没有 sigmoid，也不截断 |
| 22-23 | `count_parameters` | 统计可训练参数的总数 |

**LSTM** 每一秒读入 `(V, I, T)`，用遗忘门、输入门、候选值、输出门这 4 个"门"来更新两种记忆：隐藏状态 h 和细胞状态 c。

### 6.2 W240_H32_L1 的形状变化（batch = 256）

```
x              (256, 240, 3)
  → LSTM       out (256, 240, 32)
  → out[:,-1]  (256, 32)
  → fc         (256, 1)
  → squeeze    (256,)       ← 和 yb (256,) 一起算 MSE
```

### 6.3 参数量

| 参数 | 形状 | 个数 |
|---|---|---|
| `lstm.weight_ih_l0` | (128, 3) | 384 |
| `lstm.weight_hh_l0` | (128, 32) | 4,096 |
| `lstm.bias_ih_l0` | (128,) | 128 |
| `lstm.bias_hh_l0` | (128,) | 128 |
| `fc.weight` | (1, 32) | 32 |
| `fc.bias` | (1,) | 1 |
| **合计** | | **4,769** |

通用公式：每层 LSTM 有 `4H × (I + H + 2)` 个参数，fc 有 `H + 1` 个。PyTorch 的 LSTM 有两个偏置；如果按教科书只算一个偏置，结果是 4,641。

| 配置 | 参数量（手算） |
|---|---|
| H32_L1 | 4,769 |
| H32_L2 | 13,217 |
| H64_L1 | 17,729 |
| H64_L2 | 51,009 |
| H128_L1 | 68,225 |
| H128_L2 | 200,321 |

---

## 阶段 7：训练（`sweep_lstm.py`）

### 7.1 22 组超参数

- `sweep_lstm.py:58-60`：基础网格 W ∈ {30, 60, 120} × H ∈ {32, 64, 128} × L ∈ {1, 2}，共 18 组，输出名 `lstm_sweep`。
- `sweep_lstm.py:62-64`：扩展网格（`--ext`）W ∈ {180, 240} × H ∈ {32, 128} × L = 1，共 4 组，输出名 `lstm_sweep_ext`。扩展的原因是最初 18 组中最好的配置落在 W=120 这个边界上（第 61 行）。
- `:306-321`：根据参数选网格，还可以用 `--windows`、`--hidden`、`--layers` 只跑其中一部分，取值必须在当前网格里（`:315-319`）。
- `:308-311`：`--smoke`，只用 25 °C、2 组、每组 2 轮。
- `:322-323`：改了 `--max-epochs` 就必须换 `--name`。
- `config_name()`（`:103-104`）生成 `W240_H32_L1` 这样的名字。

### 7.2 共同设置（`:66-71`、`:335-339`）

| 设置 | 值 |
|---|---|
| seed | 42 |
| lr | 1e-3 |
| batch | 256 |
| max epochs | 200 |
| patience | 15 |
| 评估 batch | 2048 |
| 优化器 / loss | Adam / MSE |
| dropout | 0 |

可复现性：`:342-343` 设置 cudnn 为确定性模式；`:145` 每组开始时都调用一次 `set_seed(42)`（函数在 `:107-111`）。

### 7.3 `main()` 在训练前的准备

1. 打开测试集保护并自检（`:353-359`）。
2. 读取训练集和验证集（`:363-366`）。
3. 计算、保存、核对 stats（`:368-378`）。
4. 检查已有结果是否用了同样的设置（`:383-395`）。如果只有 `eval_start_index` 不同，会提示加 `--eval-start 119`。
5. 逐组训练；已完成的组跳过（`:398-408`）。

### 7.4 `train_one()`（`:143-237`）

**准备**（`:145-154`）：
- 训练集 `WindowDataset(..., None)`，验证集 `WindowDataset(..., 239)`。
- `y_val = val_ds.y[val_ds.ends]`。
- 创建模型、`Adam(lr=1e-3)`、`MSELoss`。
- `gen = torch.Generator().manual_seed(42)`，专门用来打乱顺序。

**每一轮**（`:160-171`）：

```python
model.train()
for xb, yb in train_ds.iter_batches(256, shuffle=True, generator=gen):
    opt.zero_grad(set_to_none=True)     # 清空梯度
    loss = loss_fn(model(xb), yb)       # 前向传播 + MSE
    loss.backward()                     # 反向传播
    opt.step()                          # 更新参数
    loss_sum += loss.detach().double() * len(yb)
train_loss = loss_sum / len(train_ds)
val_loss = mean((predict(model, val_ds) - y_val) ** 2)
```

以 W=240 为例：每轮有 844 个 batch，最后一批只有 213 个窗口，所以要按样本数加权。`gen` 只创建一次，因此每轮打乱的顺序都不同，但整个过程是确定、可复现的。

`predict()`（`:130-133`）：`@torch.no_grad()`，`model.eval()`，不打乱，每批 2048 个。41,057 个验证点分成 21 批。

换算关系：val RMSE（百分点）= √val_loss × 100。例如 val_loss = 1e-4 对应 1 %。

**early stopping**（`:174-187`）：
- 只有 `val_loss < best_loss`（严格小于）才算创新低。创新低时计数器清零，并把当前权重深拷贝到 CPU（`:177`）。
- 否则 `bad += 1`。`bad >= 15` 时停止训练。
- 日志里创新低的一轮会标 `*`，否则显示 `(n/15)`。

**最佳模型**（`:192-210`）：
- 训练结束后装回 `best_state`，调用 `metrics_pct()`（`:136-139`，误差 × 100 之后算 RMSE 和 MAE），并按温度分别计算（`:196-201`）。
- `torch.save` 保存 `state_dict`、`config`、`features`、`norm_stats`、`settings`、`best_epoch`、`val_rmse_pct`。先写 `.pt.tmp`，再改名为 `.pt`。

注意：验证集既用来选"哪一轮"，又用来选"哪一组配置"，所以验证成绩会偏乐观。这正是需要一个只用一次的测试集的原因。

### 7.5 断点续跑

- `:398-405`：`.json` 和 `.pt` 都存在，才算这一组已完成，跳过。
- `.json` 最后写（`:233-234`）。`write_json()`（`:122-127`）和保存 `.pt` 都是先写临时文件再改名，这是原子操作。
- `:383-395`：续跑前核对设置。
- 局限：中断时正在跑的那一组会从第 1 轮重新开始。

### 7.6 输出文件（以 `--ext` 的 W240_H32_L1 为例）

| 文件 | 写入位置 | 内容 |
|---|---|---|
| `models/lstm_sweep_ext/W240_H32_L1.pt` | `:206-210` | 最佳一轮的权重，以及配置、stats、设置 |
| `results/lstm_sweep_ext/runs/W240_H32_L1_history.csv` | `:213` | 每轮的 loss、验证 RMSE、用时 |
| `results/lstm_sweep_ext/curves/W240_H32_L1.png` | `:214`（`plot_curve` 在 `:240-262`） | 训练曲线 |
| `results/lstm_sweep_ext/runs/W240_H32_L1.json` | `:234` | 完整结果，同时是"已完成"的标记 |
| `results/lstm_sweep_ext/sweep_results.csv` | `:408`、`:410`（`write_summary` 在 `:265-283`） | 这一批的汇总表 |
| `results/lstm_sweep_ext/norm_stats.json` | `:378` | 标准化参数 |
| `results/lstm_sweep_ext/sweep_log.txt` | `Logger`（`:87-100`） | 日志，追加写入 |

---

## 阶段 8：合并结果（`combine_lstm_sweep.py`）

### 8.1 为什么要重新计算

| 批次 | 训练时的验证起点 | 验证点数 |
|---|---|---|
| 最初 18 组 | 119 | 41,777 |
| 扩展 4 组 | 239 | 41,057 |

两批用的验证点不同，没法直接比较；而且 W=240 在第 119 到 238 行没有预测值。所以把 22 组全部放在第 239 行及以后的点上重新计算。只做推理，不重新训练。

### 8.2 流程（`main()`，`:55-202`）

1. `:66-70`：列出 22 组（复用 `sweep_lstm` 的网格常量）。`:71-74`：输出路径，并用 assert 确保不会覆盖旧表。`:76`：最先打开测试集保护。
2. `:82-98`：找出已完成的组（`.json` 和 `.pt` 都在）。有缺失时报错，除非加了 `--allow-missing`。
3. `:100-105`：`torch.load(..., map_location="cpu", weights_only=True)`。
4. `:107-135`：一致性检查。训练设置（除评估起点外）相同、输入特征相同、22 个 stats 两两相同并且和两个 json 一致、扩展组的起点等于这次指定的起点、起点 ≥ W_max − 1。
5. `:137-146`：划分数据，自检测试集保护，只读取验证集。
6. `:151-162`：按 `.pt` 里的配置重建模型，装入权重，用 `WindowDataset(val, stats, w, 239)` 调用 `predict` 和 `metrics_pct`，并按温度计算（`per_temperature`，`:45-52`）。

### 8.3 怎么确认读回来的就是原来那个模型（`:164-173`）

在这一组**训练时用的起点**（119 或 239）上重新计算一次 RMSE，和 JSON 里记录的 `val_rmse_pct` 比较。如果差值超过 `CHECK_TOL_PCT = 0.01` 个百分点（`:42`），就在日志里标上 `<-- 与记录值相差较大，请检查`。

推理过程是确定的：权重、配置、stats、验证数据都相同，结果就应该一致。只要权重文件拿错、被覆盖、stats 不对或者数据变了，这个数就对不上。

注意：差值过大时只是提醒，不会让程序停止，所以要自己去看 `combine_log.txt`。

### 8.4 输出

- `results/lstm_sweep_ext/sweep_results_22.csv`：按 `val_rmse_pct`（起点 239）排序。另外包含 `sweep`、`early_stop_eval_start`、`val_rmse_pct_recorded`、`recheck_diff_pct`，以及从 JSON 复制过来的各列。
- `results/lstm_sweep_ext/combine_log.txt`。

写报告时需要说明：最初 18 组的"最佳一轮"是按起点 119 选出来的（README 第 191 行）。

---

## 阶段 9：测试集评估（`evaluate_lstm.py`）

这是唯一会读取测试集的脚本，不打开测试集保护。按照文件开头的说明（`:3-5`），应当在只根据验证集选定配置之后，运行一次。

### 9.1 读取哪些东西

- 参数（`:67-76`）：`--window`、`--hidden`、`--layers` 必填；`--name`、`--eval-start`（默认 239）、`--device` 可选。
- `.pt`（`:78-89`）：依次在 `models/lstm_sweep_ext/` 和 `models/lstm_sweep/` 里找。找不到或者两处都有，都会报错。`cfg`、`stats`、`settings` 全部从 `.pt` 里读取。
- 核对（`:90-96`）：输入特征的名字和顺序、`norm_stats.json` 与 `.pt` 是否一致。
- 重建模型（`:98-103`）：`eval_start = max(args.eval_start, W−1)`。
- 测试集（`:110-116`）：按 `settings["folders"]` 调用 `build_splits`，然后 `load_sequences(splits["test"])`，共 23 个文件。

### 9.2 逐个文件预测（`:118-138`）

对每个测试文件：建一个只包含它的 `WindowDataset`，调用 `predict`；用 `local_index` 找回每个点对应的真实 SOC 和秒数；计算 RMSE、MAE 和最大绝对误差；写出 predictions CSV 和图。

### 9.3 温度 × 工况表（`:143-166`）

- RMSE = √mean(e²)，MAE = mean(|e|)，其中 e = (预测 − 真值) × 100。
- 表的大小是 7 行（6 个温度 + all）× 5 列（UDDS, HWFET, LA92, US06 + all）。
- 每个格子：用掩码挑出对应的文件（`:156-160`），把这些文件的**逐点误差拼接在一起**之后再计算（`agg`，`:148-152`）。
- 没有文件的格子（25 °C 的 HWFET）是 NaN，打印成 `-`。
- `all` 行和 `all` 列是合并所有点重新计算的，**不是**对格子取平均，所以长文件（UDDS）的权重更大。

### 9.4 输出（`results/<批次>/test_eval/<配置>_start<起点>/`，`:106`）

| 文件 | 写入位置 | 内容 |
|---|---|---|
| `rmse_table.csv` / `mae_table.csv` | `:163` | 温度 × 工况 |
| `per_file_metrics.csv` | `:141` | 23 行，每个测试文件一行 |
| `summary.json` | `:169-178` | 全部测试点的 RMSE / MAE、点数、模型文件、验证 RMSE、评估起点 |
| `predictions/<温度>_<文件>.csv` | `:132-133` | `time_s, soc_true, soc_pred` |
| `plots/<温度>_<文件>.png` | `:134-137`（`plot_file` 在 `:45-63`） | 上图是真实 vs 预测 SOC，下图是误差 |

### 9.5 Windows 命令（W240_H32_L1）

在 Anaconda Prompt 里输入：

```bat
conda activate 你的环境名
cd /d D:\battery-fyp
python src\evaluate_lstm.py --window 240 --hidden 32 --layers 1
```

不需要加 `--name`，程序会自动在 `models\lstm_sweep_ext\` 里找到这个文件。结果输出在 `results\lstm_sweep_ext\test_eval\W240_H32_L1_start239\`。

请只对根据验证集选定的那一组运行这条命令。

---

## 阶段 10：总结

### 10.1 步骤总表

| # | 步骤 | 文件 | 函数（行号） | 输入 | 输出 |
|---|---|---|---|---|---|
| 0 | 数据清洗（更早的阶段） | `preprocess_kollmeyer.py` | `run`（`:495`） | 208 个原始 CSV | 102 条清洗后的 CSV、`manifest.json` |
| 1 | 筛选测试类型 | `lstm_data.py` | `build_splits`（`:148-153`） | `manifest.json`，231 条 | 66 条（Mixed 43、drive 23） |
| 2 | 划分数据集 | `lstm_data.py` | `build_splits`（`:155-209`） | 66 条 | train 37 / val 6 / test 23（`SeqInfo`） |
| 2b | 打开测试集保护 | `lstm_data.py` / `sweep_lstm.py` | `block_test_files`（`:78`）；自检 `sweep_lstm.py:353-359` | manifest | 27 个路径的黑名单，读取时报错 |
| 3 | 读取和重采样 | `lstm_data.py` | `load_sequences`（`:249`）→ `read_raw`（`:226`）→ `resample_1s`（`:240`） | CSV（0.1 s） | `Sequence(sec, x(n,3), y(n))`，1 s 一行 |
| 4 | 标准化参数 | `lstm_data.py` / `sweep_lstm.py` | `compute_norm_stats`（`:263`），调用处 `sweep_lstm.py:368` | 37 条训练序列（224,664 行） | mean、std → `norm_stats.json` 和 `.pt` |
| 5 | 标准化和切窗口 | `lstm_data.py` | `WindowDataset`（`:297`）、`normalize`（`:288`） | 序列、stats、W、起点 | 按需取出 `(batch, W, 3)` 和 `(batch,)` |
| 6 | 模型 | `lstm_model.py` | `LSTMSOC`（`:9`） | `(batch, W, 3)` | `(batch,)` SOC |
| 7 | 训练一组 | `sweep_lstm.py` | `train_one`（`:143`） | 训练集/验证集、stats、(W, H, L) | `.pt`、`runs/*.json`、history、curve |
| 7b | 训练全部组 | `sweep_lstm.py` | `main`（`:287`）、`write_summary`（`:265`） | 命令行参数 | 每一批的 `sweep_results.csv` |
| 8 | 合并 22 组 | `combine_lstm_sweep.py` | `main`（`:55`） | 22 个 `.json` 和 `.pt`、验证集 | `sweep_results_22.csv`、`combine_log.txt` |
| 9 | 选定配置 | （人工） | — | `sweep_results_22.csv`、训练曲线 | 一组 (W, H, L) |
| 10 | 测试集评估 | `evaluate_lstm.py` | `main`（`:66`） | 1 个 `.pt`、23 个测试文件 | `rmse_table.csv`、`mae_table.csv`、`per_file_metrics.csv`、`summary.json`、predictions、plots |

### 10.2 防止数据泄露的设计

| # | 设计 | 防止的是什么 | 代码位置 |
|---|---|---|---|
| 1 | `ah` 列不读入，并且在代码里明确禁止 | 标签的来源（库仑计数）被当作输入 | `lstm_data.py:5-6`、`:40-41`、`:232-234` |
| 2 | 按整个文件划分，而不是按窗口 | 相邻窗口 99 % 重叠，训练和验证之间"双胞胎"泄露 | `lstm_data.py:138-209` |
| 3 | `_part1` 和 `_part2` 必须在同一个集合 | 同一个原始文件跨越两个集合 | `lstm_data.py:201-204` |
| 4 | 三个集合两两不重叠 | 同一条序列出现在两个集合里 | `lstm_data.py:199-204` |
| 5 | 验证文件用机械规则选出，并和写死的名单核对 | 为了结果好看而挑选验证文件；数据变了但验证集被悄悄更换 | `lstm_data.py:48-55`、`:156-170` |
| 6 | 测试集是另一类工况（单一工况），并排除 PausCycl（它是 UDDS 的开头片段） | 测试工况的片段出现在训练集里 | `lstm_data.py:152`、`:174-177`（PausCycl 通过 `test_type` 被排除；原因见 D26） |
| 7 | 测试集黑名单直接从 manifest 生成，包括 `_excluded` 里的 drive 文件 | 划分逻辑出错时仍能拦住测试文件 | `lstm_data.py:78-90` |
| 8 | 所有读取都经过 `read_raw()`，有两道检查 | 训练和对比阶段读到测试文件 | `lstm_data.py:93-98`、`:229-231` |
| 9 | 训练和合并时先打开保护，并故意读一次测试文件做自检 | 保护被改坏了却没人发现 | `sweep_lstm.py:353-359`；`combine_lstm_sweep.py:76`、`:139-144` |
| 10 | mean 和 std 只用训练集计算，混入其他集合就报错 | 验证集或测试集的统计量泄露进训练过程 | `lstm_data.py:263-274`（`:264-265`） |
| 11 | 所有集合套用同一组 stats，并且 stats 跟着 `.pt` 走 | 验证集或测试集用了"自己的" stats，或者用错了 stats | `lstm_data.py:323`；`sweep_lstm.py:206-209`；`evaluate_lstm.py:89-96`；`combine_lstm_sweep.py:110-127` |
| 12 | stats 与已有文件核对 | 数据变了还在原目录续跑，结果混在一起 | `sweep_lstm.py:370-376` |
| 13 | 窗口不跨序列 | 把两次无关的实验拼在同一个窗口里 | `lstm_data.py:316-329`（`offset`、`first_end`） |
| 14 | 窗口是因果的：只用过去和当前，标签是最后一秒的 SOC；重采样时 soc 取这一秒最后一行 | 用到"未来"的信息 | `lstm_data.py:348`、`:245`；`lstm_model.py:19` |
| 15 | early stopping 和最佳一轮只看验证集 | 用测试集来决定训练到第几轮 | `sweep_lstm.py:171-187` |
| 16 | 超参数只按验证集排序和选择 | 用测试集选模型 | `combine_lstm_sweep.py:191-193`；README 第 6 步 |
| 17 | 测试集只由 `evaluate_lstm.py` 在选定配置之后读取 | 在对比阶段"偷看"测试成绩 | `evaluate_lstm.py:3-5`、`:116`；`lstm_data.py:96-97` 的报错信息 |
| 18 | 合并时不覆盖旧表、不重新训练，只重新推理 | 合并过程改变了模型 | `combine_lstm_sweep.py:1-10`、`:74` |

**保证公平比较、结果可信**（不属于泄露，但同样重要）：

- 统一评估起点：`lstm_data.py:63`、`:311`；`sweep_lstm.py:324-327`；`combine_lstm_sweep.py:132-134`；`evaluate_lstm.py:103`
- 核对读回的模型：`combine_lstm_sweep.py:164-173`
- 续跑时核对设置：`sweep_lstm.py:383-395`
- 原子写入：`sweep_lstm.py:122-127`、`:204-210`
- 固定随机种子：`sweep_lstm.py:107-111`、`:145`、`:153`、`:342-343`

### 10.3 残留风险（写报告时如实说明）

1. 测试集保护只拦截 `read_raw()` 这一条路径；`python src/lstm_data.py` 会读取测试文件来统计行数。
2. 验证集被用了两次（选轮次和选配置），所以验证成绩偏乐观。
3. 代码不限制 `evaluate_lstm.py` 的运行次数，"只评估一次"要靠自己遵守。
4. 最初 18 组的 early stopping 用的是起点 119，而合并表用的是 239。
5. 全部数据来自同一颗电池（D1、D3）。测试集检验的是"同一颗电池在新工况下的表现"，不是"换一颗新电池"。
6. 40 °C 的训练集只有 2 条序列，训练数据在各温度之间不平衡。
7. 每个验证文件和测试文件的前 239 s 不打分。
