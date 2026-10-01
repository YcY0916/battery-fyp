# 0–40 °C LSTM SOC 实验（GECOST 2027）运行说明

这次只用 0、10、25、40 °C 四个温度，重新做超参数对比、多 seed 实验、推理时间测试、测试集评估和论文图表。训练分到两台电脑上跑：Windows 笔记本（RTX 4050）和 Google Colab（T4，结果和模型写到 Google Drive）。

旧的 6 温度实验（−20 到 40 °C、22 组配置、最佳 W240_H32_L1）的代码保存在 git tag `archive-6temp-2026-09`，结果和模型用 `src/archive_old_runs.py` 归档到 `archive/2026-09_6temp/`。

---

## 一、实验设置

| 项目 | 设置 |
|---|---|
| 温度 | 0、10、25、40 °C（`--temps 0 10 25 40`，默认值；`--temps -20 -10 0 10 25 40` 可以按新流程重跑 6 温度，输出到 `results/paper_n20_n10_0_10_25_40/`） |
| 划分 | 按整个文件划分，`_part1/_part2` 跟着原文件走；绝不按窗口随机划分 |
| 输入 / 输出 | `voltage, current, temperature` → 窗口最后一个时间点的 `soc`；`ah` 列不读入 |
| 重采样 | 每条序列按 `floor(time_s)` 重采样到 1 s：电压、电流、温度取平均，soc 取该秒最后一行；SOC 标签直接用 `data/processed/kollmeyer_clean/*.csv` 里现成的 `soc` 列，不重新计算 |
| 标准化 | mean/std 只用训练集计算一次，存为 `results/paper_0to40/norm_stats.json`；之后所有脚本只读取这个文件，不重新计算；两台电脑用同一份文件 |
| 评估起点 | 所有验证和测试指标（以及早停用的验证 loss）只算每条序列第 299 行（300 s）及之后的点，所有 W 比较同一批点 |
| 训练 | MSE loss，Adam lr = 0.001，batch 256，dropout 0，最多 200 轮；验证 loss 连续 15 轮没有创新低就停止，保存验证 loss 最低那一轮的权重 |
| 第 2 步配置（18 组，seed 42） | W ∈ {60, 120, 180, 240, 300} × H ∈ {32, 64, 128}，L = 1（15 组）；W = 240、L = 2、H ∈ {32, 64, 128}（3 组） |
| 第 3 步候选 | 第 2 步验证 RMSE 前 4 名 ∪ 最佳 W 下所有 L = 1 的 H（去重）；每个候选用 seed 42–46（42 直接复用第 2 步） |
| 选择规则 | 验证 RMSE 均值最低的记为 best；门槛 = best 的均值 + best 的标准差；均值不超过门槛的候选里选参数量最少的 |
| 标准差 | 全部是样本标准差（ddof = 1） |

### 数据划分（`python src\lstm_data.py --paper` 的输出）

| 集合 | 温度 | 文件 | 1 s 行数 | 窗口 / 评估点数 |
|---|---|---|---|---|
| 训练 | 0 °C | 589_Mixed1–2、590_Mixed4–7（6 个） | 39,668 | 39,314（W=60）… 37,874（W=300） |
| 训练 | 10 °C | 567_Mixed1–2、571_Mixed4–7（571_Mixed6 分为 _part1/_part2，共 7 条序列） | 43,444 | 43,051 … 41,611 |
| 训练 | 25 °C | 551_Mixed1–2、552_Mixed3–7（7 个） | 53,462 | 53,049 … 51,369 |
| 训练 | 40 °C | 556_Mixed1–2（2 个） | 15,430 | 15,312 … 14,832 |
| **训练合计** | | 22 条序列 | 152,004 | W=60：150,726；W=120：149,466；W=180：148,206；W=240：146,946；W=300：145,686 |
| 验证 | 0 °C | 590_Mixed8 | 7,584 | 7,285 |
| 验证 | 10 °C | 571_Mixed8 | 8,142 | 7,843 |
| 验证 | 25 °C | 552_Mixed8 | 8,592 | 8,293 |
| 验证 | 40 °C | 557_Mixed3 | 6,763 | 6,464 |
| **验证合计** | | 4 个 | 31,081 | 29,885（所有 W 相同） |
| 测试 | 0 °C | 589_UDDS、589_HWFET、589_LA92、589_US06 | 30,555 | 29,359 |
| 测试 | 10 °C | 576_UDDS、576_HWFET、582_LA92、567_US06 | 33,026 | 31,830 |
| 测试 | 25 °C | 551_UDDS、551_LA92、551_US06（没有 HWFET） | 30,062 | 29,165 |
| 测试 | 40 °C | 556_UDDS、556_HWFET、556_LA92、556_US06 | 35,352 | 34,156 |
| **测试合计** | | 15 个 | 128,995 | 124,510（所有 W 相同） |

- 训练集的窗口数随 W 变化（每条序列从第 W−1 行起）；W ≥ 60 时 `571_Mixed6_part1`（只有 39 行）比窗口短，被跳过。
- 验证集和测试集的评估点从第 299 行起，所以 W ≤ 300 时各 W 完全相同。
- 三个集合两两没有重叠（按序列，也按原始文件），`lstm_data.py` 会检查并打印。
- 排除：25 °C HWFET（数据损坏）、40 °C 562_Mixed4–8（数据异常）、全部 −20 °C 和 −10 °C 文件、全部非工况文件（Cap_1C、C20DisCh、Dis_0p5C、Dis_2C、HPPC、PausCycl 等）。

---

## 二、代码文件

| 文件 | 作用 |
|---|---|
| `src/archive_old_runs.py` | 第 0 步：把旧的结果和模型移到 `archive/2026-09_6temp/`（默认 dry run，`--apply` 才移动） |
| `src/lstm_data.py` | 读取、筛选、划分、1 s 重采样、标准化、窗口 Dataset（复用旧代码，新增 `--temps`、`--data_root`、`--paper`） |
| `src/lstm_model.py` | 模型：`LSTM(3, H, L)` → 最后一个时间步 → `Linear(H, 1)` |
| `src/sweep_lstm.py` | 旧的 6 温度对比脚本；其中的训练循环 `fit()` 和 `val_metrics()` 被新流程复用 |
| `src/paper_common.py` | 新流程的公共设置：网格、路径（`--data_root` / `--out_root`）、`--shard` 的哈希分配、运行环境记录、标准化文件 |
| `src/sweep_paper.py` | 第 2 步（`--stage sweep`）和第 3 步（`--stage multiseed`）训练；`--shard`、`--configs`、`--timing`、`--init_norm`、`--dry_run` |
| `src/merge_results.py` | 合并所有 run JSON，写 `sweep_results.csv`、`candidates.json`、`multiseed_results.csv`，检查重复 / 缺失 / 提交号 / 标准化文件 hash |
| `src/select_config.py` | 按选择规则给出建议，写 `selected_config.json`（不碰测试集） |
| `src/benchmark_inference.py` | 第 3.5 步：在同一台电脑的 CPU 上测所有配置的推理时间 |
| `src/evaluate_paper.py` | 第 4 步：对你手动指定的配置做测试集评估（5 个 seed），`--unseen` 额外评估 −10 / −20 °C |
| `src/plot_paper_figs.py` | 第 5 步：论文图 1–4（PNG 300 dpi + PDF）和 `figure_notes.md` |
| `src/smoke_paper.py` | 一条命令跑完整个流程的 smoke test |
| `src/archive/` | 只服务于旧实验的脚本（`combine_lstm_sweep.py`、`evaluate_lstm.py`），已不再使用 |

所有脚本都有 `--smoke`（只用 25 °C、2 组配置、2 个 seed、每组 2 轮，输出到 `results/paper_0to40_smoke/` 和 `models/paper_0to40_smoke/`，不提交），也都有 `--out_root`、`--data_root`（默认都是仓库根目录）。

---

## 三、准备工作

### 笔记本（Windows，在 Anaconda Prompt 里输入）

`D:\battery-fyp` 换成仓库的实际位置，`你的环境名` 换成你的 conda 环境名。

```bat
conda activate 你的环境名
cd /d D:\battery-fyp
git fetch origin
git checkout paper-0to40
git pull origin paper-0to40
git rev-parse HEAD
```

Pull Request 合并以后改为 `git checkout master` 和 `git pull origin master`。

**最后一行打印的提交号要记下来**：两台电脑上的所有训练都必须用同一个提交号的代码（合并结果时会检查，不一致会报错）。所以开始训练以后，在全部训练结束之前不要 `git pull`，也不要改代码。建议先把 PR 合并，再从 master 的同一个提交开始跑全部训练。

确认 GPU：

```bat
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Colab（每次新开一个 Colab 会话都要做）

菜单 Runtime → Change runtime type → 选 **T4 GPU**。然后依次运行下面几个单元格：

```python
# 1. 挂载 Google Drive
from google.colab import drive
drive.mount('/content/drive')
```

```python
# 2. 获取代码（克隆到 /content，速度快；仓库里已经带有 data/）
#    仓库是私有的：在左侧"钥匙"图标（Secrets）里添加名为 GH_TOKEN 的 GitHub token
from google.colab import userdata
token = userdata.get('GH_TOKEN')
!git clone https://{token}@github.com/YcY0916/battery-fyp.git /content/battery-fyp
%cd /content/battery-fyp
!git checkout 笔记本上记下的提交号
!git rev-parse HEAD
```

仓库是公开的话，第 2 格直接用 `!git clone https://github.com/YcY0916/battery-fyp.git /content/battery-fyp`。**不要**用上传压缩包的方式获取代码：没有 `.git` 目录就记录不到提交号，合并时会报错。

```python
# 3. 输出位置（Drive）；数据用仓库自带的，或者用 Drive 上的那份
OUT = '/content/drive/MyDrive/battery-fyp'      # results/ 和 models/ 写到这里
DATA = '/content/drive/MyDrive/battery-fyp'     # 这个目录下要有 data/processed/kollmeyer_clean/
!nvidia-smi
```

下面 Colab 的命令都带 `--out_root {OUT}`。仓库自带的 `data/` 与笔记本上的相同，可以不加 `--data_root`；想用 Drive 上那份数据就再加 `--data_root {DATA}`（读 Drive 会慢一些）。

---

## 四、从头到尾的命令

### 第 0 步：归档旧实验（笔记本）

1. tag：云端环境推送 tag 被拒绝（HTTP 403），请在笔记本上补推（`be7b363` 是归档前 master 的最后一个提交）：

   ```bat
   git fetch origin
   git tag -a archive-6temp-2026-09 be7b363 -m "Archive: 6-temperature LSTM experiment before the 0-40 degC paper rerun"
   git push origin archive-6temp-2026-09
   ```

2. 归档旧的结果和模型。先 dry run 看清单，确认没问题再 `--apply`：

   ```bat
   python src\archive_old_runs.py
   python src\archive_old_runs.py --apply
   ```

   - 只移动，不删除；目标位置已有同名文件时不覆盖（报 CONFLICT，原文件留在原处）；`data\` 下的文件绝不碰。
   - 被 git 跟踪的文件留在原处（目前只有 `results\lstm_sweep\README.md`）。
   - 完成后 `archive\2026-09_6temp\ARCHIVE_README.md` 说明这批结果属于什么实验，`ARCHIVE_MANIFEST.csv` 列出每个移动过的文件。`archive\` 已加入 `.gitignore`。
   - 归档后 `--timing` 仍然能在 `archive\2026-09_6temp\results\lstm_sweep*\runs\` 里找到旧的训练轮数。

### 第 1 步：核对划分，生成标准化文件（笔记本）

```bat
python src\lstm_data.py --paper
python src\sweep_paper.py --init_norm
```

- 第一条打印训练、验证、测试的文件清单和每个温度的窗口数，并检查三个集合没有重叠（为了数行数会读取测试文件，不训练也不评估）。
- 第二条只读训练集，生成 `results\paper_0to40\norm_stats.json`，并打印它的 sha256。文件已存在时只读取，不重新计算。
- 把这个文件拷到 Drive（见"五、拷贝文件"），然后在 Colab 上确认：

  ```python
  !python src/sweep_paper.py --init_norm --out_root {OUT}
  ```

  输出里应该是 `Normalisation stats read from ... not recomputed`，sha256 与笔记本上打印的相同。

### smoke test（两台电脑各跑一次，约 5 分钟）

```bat
python src\smoke_paper.py --fresh
```

```python
!python src/smoke_paper.py --fresh --out_root /content/smoke
```

依次跑：划分 → 归档脚本自检 → `--timing` → 第 2 步（`--shard 1/2`、`--shard 2/2`，再跑一次 1/2 确认会跳过）→ 合并 → 第 3 步（两个 shard）→ 合并（另加 `--eval_start 359` 的重算）→ 故意放一个重复文件确认合并会报错 → 选择 → 推理时间 → 测试评估（含 `--unseen`）→ 画图。最后打印 `SMOKE OK`。Colab 上把 smoke 输出放在 `/content/smoke`，不占 Drive。

### 计时与分配（两台电脑各跑一次）

```bat
python src\sweep_paper.py --timing
```

```python
!python src/sweep_paper.py --timing --out_root {OUT} --epochs 笔记本上打印的典型轮数
```

- 完整训练 W300_H128_L1 一轮，再对 18 组配置各测几十个训练 step 和一次验证，换算成每组每轮的时间。
- 打印两种估计：最坏情况（每组 200 轮）和典型情况（旧 sweep 结果里实际训练轮数的平均值；在 `results\lstm_sweep*\runs\` 或 `archive\2026-09_6temp\` 下自动查找，找不到就要用 `--epochs N` 手动给出）。
- 每台电脑的结果存为 `results\paper_0to40\timing\timing_<主机名>.json`。把 Colab 的那个文件拷到笔记本的同一目录，再在笔记本上跑一次 `--timing`，就会打印两台电脑的最佳 `--shard` 分配（让两台大约同时跑完）。

（预计运行时间：计时完成后补充。）

### 第 2 步：超参数对比（两台电脑同时跑）

按上面的分配，例如两台各一半：

```bat
python src\sweep_paper.py --shard 1/2
```

```python
!python src/sweep_paper.py --shard 2/2 --out_root {OUT}
```

- 配置按配置名的 sha256 分到各个 shard，同一个配置在第 2 步和第 3 步总在同一台电脑上。`--shard 1,3/4` 这种写法表示一台电脑跑多个 shard。
- 先加 `--dry_run` 可以只看这台电脑要跑哪些配置。
- `--configs W240_H32_L1 W300_H64_L1` 手动指定配置，优先于 `--shard`。
- 可以随时中断（Colab 断线也一样），之后输入**同一条命令**继续：已经有 JSON 的运行直接跳过，没跑完的那一组从第 1 轮重新开始。
- 这一步读不到测试集：一旦读取任何测试文件，脚本直接报错停止（开始前会自检）。

### 合并第 2 步的结果（笔记本）

1. 把 Colab 的结果拷回笔记本（见"五、拷贝文件"）。
2. 合并：

   ```bat
   python src\merge_results.py
   ```

   - 写出 `sweep_results.csv`（按验证 RMSE 从小到大）和 `candidates.json`（第 3 步的候选）。
   - 检查重复文件（例如 `... (1).json`）、缺失的运行、所有运行的提交号和标准化文件 hash 是否一致，不一致就报错并列出是哪些运行。
   - 最佳配置的 W = 300（网格边缘）时会醒目地提醒 W 可能还没饱和。
3. 把 `results\paper_0to40\candidates.json` 拷到 Drive 的同一位置。

### 第 3 步：多 seed 实验（两台电脑同时跑，shard 写法与第 2 步相同）

```bat
python src\sweep_paper.py --stage multiseed --shard 1/2
```

```python
!python src/sweep_paper.py --stage multiseed --shard 2/2 --out_root {OUT}
```

- 每个候选跑 seed 43–46；seed 42 直接用第 2 步的结果，不会重新训练。
- 同样可以中断后继续。

### 合并第 3 步、给出建议（笔记本）

再拷一次 Colab 的结果，然后：

```bat
python src\merge_results.py
python src\select_config.py
```

- `multiseed_results.csv`：每个候选的验证 RMSE、MAE 和各温度 RMSE 的 mean ± std、参数量、训练时间（以及是哪台电脑跑的，不同电脑的训练时间不能互相比较）。
- `select_config.py` 打印选择过程（best、门槛、合格的候选、参数量最少的那个），写入 `selected_config.json`。这只是建议，不会碰测试集。

### 第 3.5 步：推理时间（笔记本，所有训练结束、权重都拷回来以后）

**测量时不要同时跑任何训练或其他占 CPU 的程序**（Colab 上的训练不影响，笔记本自己不要跑）。

```bat
python src\benchmark_inference.py
```

- 加载第 2 步 18 组配置的 seed 42 权重，固定 `torch.set_num_threads(1)`，在 CPU 上测 batch = 1 单个窗口：预热 100 次，再测 1000 次，取平均和标准差。
- 写 `inference_benchmark.csv`（含参数量和 CPU 型号），并自动把推理时间合并进 `sweep_results.csv` 和 `multiseed_results.csv`（以后再跑 `merge_results.py` 也会自动带上）。

### 第 4 步：测试集评估（笔记本，只对你决定的配置运行）

```bat
python src\evaluate_paper.py --config W240_H32_L1 --unseen
```

把 `W240_H32_L1` 换成你最终决定的配置。

- 5 个 seed 分别评估 15 个测试文件，都从第 299 行开始算。
- 输出"温度 × 工况"的 RMSE / MAE 表（5 个 seed 的 mean ± std）、15 个文件的总平均（每个 seed 先对 15 个文件的指标取平均，再算 5 个 seed 的 mean ± std；另外给出把所有测试点合在一起算的 RMSE 作参考），以及 `test_table.tex`。
- 表格脚注：25 °C 没有 HWFET；† 标出满容量由放电终止容量反推的文件（10 °C LA92、40 °C 四个工况）；‡ 标出 10 °C US06（满容量借用 576 批次的 Cap_1C 并做老化修正，该标签已确认存在偏差）。标记是根据 `manifest.json` 的 `capacity_rule` 自动生成的。
- `--unseen`：同一组模型额外评估 −10 °C 和 −20 °C 的四个标准工况，单独存成 `unseen_temp_results.csv`，不参与任何选择。

### 第 5 步：论文图表（笔记本）

```bat
python src\plot_paper_figs.py
```

- 最终配置默认取"唯一用 `evaluate_paper.py` 评估过的配置"，否则取 `selected_config.json` 的建议；也可以用 `--config W.._H.._L..` 指定。
- 输入还不全时（例如还没做第 4 步），缺输入的图会跳过并提示，其余照常生成。

---

## 五、拷贝文件

下面假设笔记本装了 Google Drive for desktop，Drive 挂在 `G:\My Drive`（中文系统可能是 `G:\我的云端硬盘`），Colab 的 `OUT` 对应 `G:\My Drive\battery-fyp`。没装的话，在 drive.google.com 上把文件夹下载成 zip，解压到例如 `D:\colab_download`，下面的 `%DRIVE%` 换成解压后的目录即可。

### 标准化文件和候选清单：笔记本 → Drive

```bat
set DRIVE=G:\My Drive\battery-fyp
mkdir "%DRIVE%\results\paper_0to40"
copy results\paper_0to40\norm_stats.json "%DRIVE%\results\paper_0to40\"
copy results\paper_0to40\candidates.json "%DRIVE%\results\paper_0to40\"
```

（第二个 `copy` 在合并第 2 步之后才需要。）也可以在网页上传到 `我的云端硬盘/battery-fyp/results/paper_0to40/`；如果提示同名文件，选"替换"，不要"保留两者"。拷完后在 Colab 上跑一次 `--init_norm`，确认 sha256 与笔记本相同。

### 训练结果：Drive → 笔记本

```bat
set DRIVE=G:\My Drive\battery-fyp
robocopy "%DRIVE%\results\paper_0to40" results\paper_0to40 /E /XC /XN /XO /XF norm_stats.json candidates.json
robocopy "%DRIVE%\models\paper_0to40" models\paper_0to40 /E /XC /XN /XO
```

- 拷的是 `runs\*.json`（以及 `*_history.csv`）、`val_preds\*.npz`、`sweep\curves\`、`multiseed\curves\`、`logs\`、`timing\` 和 `models\paper_0to40\*.pt`。
- `/XC /XN /XO` 表示笔记本上已经有的同名文件一律不覆盖。正常情况下两台电脑的运行文件不会同名；robocopy 汇总里 Skipped 不为 0，说明有同一个运行在两台电脑上都跑过，需要检查。
- 网页下载 zip 时，Google 有时会把同名文件改成 `xxx (1).json`，`merge_results.py` 会把这种文件当作重复报错，删掉多余的那个再合并。

---

## 六、输出文件（都在 `--out_root` 下）

| 路径 | 内容 |
|---|---|
| `results/paper_0to40/norm_stats.json` | 标准化参数（训练集的 mean、std、用了哪些序列）；所有电脑共用这一份 |
| `results/paper_0to40/runs/<配置>_seed<seed>.json` | 一次运行的结果：验证 RMSE / MAE（SOC 百分点）、各温度的验证 RMSE / MAE、最佳轮数、总轮数、训练时间、可训练参数量、运行环境（主机名、GPU 型号或 CPU、PyTorch 版本、git 提交号、标准化文件 sha256）；它存在就表示这次运行已完成 |
| `results/paper_0to40/runs/<配置>_seed<seed>_history.csv` | 每一轮的 train loss、val loss、val RMSE、用时 |
| `results/paper_0to40/val_preds/<配置>_seed<seed>.npz` | 验证集逐点预测（从第 W−1 行起的所有点）；以后加 W = 360 时可以用 `merge_results.py --eval_start 359` 在更晚的起点重算指标，不用重新训练 |
| `results/paper_0to40/sweep/curves/<运行>.png` | 第 2 步（seed 42）的训练曲线 |
| `results/paper_0to40/multiseed/curves/<运行>.png` | 第 3 步（seed 43–46）的训练曲线 |
| `results/paper_0to40/logs/` | 每台电脑的运行日志（`sweep_<主机名>.log`、`multiseed_<主机名>.log`、`merge.log` 等） |
| `results/paper_0to40/timing/timing_<主机名>.json` | `--timing` 的测量结果 |
| `results/paper_0to40/sweep_results.csv` | 第 2 步 18 组配置，按验证 RMSE 排序：验证 RMSE / MAE、各温度 RMSE、最佳轮、总轮数、参数量、训练时间和训练机器；第 3.5 步之后加上 `inference_ms_mean/std` 和 CPU 型号 |
| `results/paper_0to40/candidates.json` | 第 3 步候选及其来源（前 4 名、最佳 W 下的 L = 1 配置） |
| `results/paper_0to40/multiseed_results.csv` | 每个候选的 mean ± std（验证 RMSE、MAE、各温度 RMSE）、参数量、训练时间及机器、推理时间 |
| `results/paper_0to40/all_runs.csv` | 每次运行一行 |
| `results/paper_0to40/selected_config.json` | 选择规则给出的建议配置和选择过程 |
| `results/paper_0to40/inference_benchmark.csv` | 每组配置的 CPU 推理时间（ms，mean / std / median）、参数量、CPU 型号 |
| `results/paper_0to40/test/<配置>/test_results.csv` | 每个测试文件一行：RMSE / MAE 的 mean、std，容量标记；最后一行是 15 个文件的总平均 |
| `results/paper_0to40/test/<配置>/test_rmse_table.csv`、`test_mae_table.csv` | 温度 × 工况，"mean ± std" |
| `results/paper_0to40/test/<配置>/test_summary.csv` | 15 个文件的总平均（以及所有点合在一起算）的 RMSE / MAE，mean ± std |
| `results/paper_0to40/test/<配置>/test_table.tex` | 可以直接贴进 IEEE LaTeX 模板的表格（含脚注，不需要额外宏包） |
| `results/paper_0to40/test/<配置>/per_file_metrics.csv` | 每个（seed, 测试文件）一行 |
| `results/paper_0to40/test/<配置>/predictions_seed42/*.csv` | seed 42 在每个测试文件上的逐点预测（`time_s, local_index, soc_true, soc_pred`） |
| `results/paper_0to40/test/<配置>/unseen_temp_results.csv` | `--unseen`：−10 / −20 °C 的结果（不参与任何选择） |
| `results/paper_0to40/test/<配置>/summary.json`、`evaluate_log.txt` | 评估设置和日志 |
| `results/paper_0to40/figs/fig1_window_vs_rmse.png/.pdf` | 图 1（单栏）：W 对验证 RMSE（seed 42） |
| `results/paper_0to40/figs/fig2_params_vs_rmse.png/.pdf` | 图 2（单栏）：参数量（对数）对验证 RMSE（mean ± std），门槛虚线，标出选中的配置 |
| `results/paper_0to40/figs/fig3_temp_heatmap.png/.pdf` | 图 3（单栏）：L = 1 各配置在 0/10/25/40 °C 的验证 RMSE 热力图 |
| `results/paper_0to40/figs/fig4_us06_traces.png/.pdf` | 图 4（双栏）：最终配置在 US06 0/25/40 °C 的参考 / 预测 SOC 和误差，横轴分钟 |
| `results/paper_0to40/figure_notes.md` | 每张图的说明和关键数字（中文），方便写论文 |
| `models/paper_0to40/<配置>_seed<seed>.pt` | 验证 loss 最低那一轮的权重（附带配置、标准化参数、训练设置） |

图表格式：IEEE 单栏 3.5 in、双栏 7.16 in，字号 8 pt 左右，Times 字体（没有时用相近的衬线字体），PDF 内嵌 TrueType 字体；色盲友好配色（蓝 / 橙 / 青绿，并用线型区分 H）；每种视觉编码都有图例；没有用 marker 大小表示数值；训练时间不出现在任何图里。

---

## 七、注意事项

- 两台电脑必须用同一个 git 提交号；标准化文件只在笔记本上生成一次，再拷到 Drive。合并时任何不一致都会报错。
- 一次运行（一个配置 + 一个 seed）必须在同一台电脑上从头跑到尾；中断后重跑同一条命令即可。
- 推理时间只在笔记本上测，测的时候不要跑其他任务。
- smoke 的输出在 `results\paper_0to40_smoke\`、`models\paper_0to40_smoke\`，已被 `.gitignore` 忽略。
- 测试集只在第 4 步、由你指定配置后读取；第 2、3 步和合并、选择、推理时间测试都不读测试文件。
