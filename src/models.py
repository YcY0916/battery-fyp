"""四个 SOC 估计模型和它们的超参数网格。

接口与 LSTMSOC 完全相同：输入 (batch, W, 3)，输出 (batch,)，输出头都是 Linear(·, 1) 再 squeeze。

- lstm：LSTMSOC（src/lstm_model.py，原样导入，不改动）
- gru：GRU -> 最后一个时间步 -> Linear(H, 1)，除了 nn.LSTM 换成 nn.GRU 外与 LSTM 相同
- cnn：转置成 (batch, 3, W) -> N 个 [Conv1d(k=3, p=1) -> BatchNorm1d -> ReLU]
       -> AdaptiveAvgPool1d(1) -> Flatten -> Linear(F, 1)
- transformer：Linear(3, d_model) -> 正弦位置编码 -> TransformerEncoder
       （nhead=4，dim_feedforward=4*d_model，dropout=0.1，batch_first）-> 最后一个时间步 -> Linear(d_model, 1)

网格（窗口长度都用 LSTM 最终用到的 {30, 60, 120, 180, 240}）：
- lstm / gru：W x H{32, 64, 128} x L{1, 2}          30 组
- cnn：       W x F{32, 64, 128} x L{2, 3}          30 组
- transformer：W x D{32, 64} x L{1, 2}              20 组
配置名沿用 LSTM 的写法，例如 W120_H64_L2、W120_F64_L3、W120_D64_L2。
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import nn

from lstm_model import LSTMSOC, count_parameters  # noqa: F401  （count_parameters 供其他脚本从这里导入）

# LSTM 最终搜索用到的全部窗口长度（最初 18 组的 30/60/120 + 扩展 4 组的 180/240）
WINDOWS = [30, 60, 120, 180, 240]
INPUT_SIZE = 3
POS_MAX_LEN = 512  # 位置编码表的长度，必须 >= 最长窗口
assert POS_MAX_LEN >= max(WINDOWS)


class GRUSOC(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, input_size: int = INPUT_SIZE):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True, dropout=0.0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, W, 3) -> (batch,)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class CNNSOC(nn.Module):
    def __init__(self, filters: int, num_layers: int, input_size: int = INPUT_SIZE):
        super().__init__()
        blocks = []
        in_ch = input_size
        for _ in range(num_layers):
            blocks += [nn.Conv1d(in_ch, filters, kernel_size=3, padding=1),
                       nn.BatchNorm1d(filters), nn.ReLU()]
            in_ch = filters
        self.conv = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(filters, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, W, 3) -> (batch, 3, W) -> (batch, F, W) -> (batch, F) -> (batch,)
        z = self.conv(x.transpose(1, 2))
        return self.fc(self.flatten(self.pool(z))).squeeze(-1)


class SinusoidalPositionalEncoding(nn.Module):
    """PE[pos, 2i] = sin(pos / 10000^(2i/d))，PE[pos, 2i+1] = cos(pos / 10000^(2i/d))，直接加到输入上。"""

    def __init__(self, d_model: int, max_len: int = POS_MAX_LEN):
        super().__init__()
        if d_model % 2:
            raise ValueError(f"d_model 必须是偶数，收到 {d_model}")
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        # 固定的表，不训练，也不存进 state_dict（每次按公式重建）
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, W, d_model)
        if x.size(1) > self.pe.size(1):
            raise ValueError(f"窗口长度 {x.size(1)} 超过位置编码的 max_len {self.pe.size(1)}")
        return x + self.pe[:, :x.size(1)]


class TransformerSOC(nn.Module):
    def __init__(self, d_model: int, num_layers: int, input_size: int = INPUT_SIZE,
                 nhead: int = 4, dropout: float = 0.1, max_len: int = POS_MAX_LEN):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                           dim_feedforward=4 * d_model, dropout=dropout,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, W, 3) -> (batch, W, d_model) -> 最后一个时间步 (batch, d_model) -> (batch,)
        z = self.encoder(self.pos_enc(self.input_proj(x)))
        return self.fc(z[:, -1, :]).squeeze(-1)


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    name: str
    builder: Callable[..., nn.Module]
    # 结构超参数：(参数名, 配置名里的字母, 网格取值)，顺序就是配置名里的顺序
    hparams: tuple[tuple[str, str, tuple[int, ...]], ...]
    # 不参与搜索的固定结构设置，写进 settings 方便核对
    fixed: dict = field(default_factory=dict)

    @property
    def hp_names(self) -> list[str]:
        return [n for n, _, _ in self.hparams]

    def config_name(self, window: int, hp: dict) -> str:
        return f"W{window}_" + "_".join(f"{tag}{hp[n]}" for n, tag, _ in self.hparams)

    def grid(self, windows: list[int] | None = None) -> list[tuple[int, dict]]:
        """全部 (窗口, 超参数) 组合；顺序与 sweep_lstm.py 相同：先 W，再依次遍历各超参数。"""
        windows = WINDOWS if windows is None else windows
        values = [vals for _, _, vals in self.hparams]
        return [(w, dict(zip(self.hp_names, combo)))
                for w in windows for combo in itertools.product(*values)]

    def build(self, hp: dict) -> nn.Module:
        return self.builder(**hp)


SPECS: dict[str, ModelSpec] = {
    "lstm": ModelSpec("lstm", LSTMSOC,
                      (("hidden_size", "H", (32, 64, 128)), ("num_layers", "L", (1, 2))),
                      {"dropout": 0.0}),
    "gru": ModelSpec("gru", GRUSOC,
                     (("hidden_size", "H", (32, 64, 128)), ("num_layers", "L", (1, 2))),
                     {"dropout": 0.0}),
    "cnn": ModelSpec("cnn", CNNSOC,
                     (("filters", "F", (32, 64, 128)), ("num_layers", "L", (2, 3))),
                     {"kernel_size": 3, "padding": 1, "norm": "BatchNorm1d", "pool": "AdaptiveAvgPool1d(1)"}),
    "transformer": ModelSpec("transformer", TransformerSOC,
                             (("d_model", "D", (32, 64)), ("num_layers", "L", (1, 2))),
                             {"nhead": 4, "dim_feedforward": "4*d_model", "dropout": 0.1,
                              "pos_encoding": "sinusoidal", "pos_max_len": POS_MAX_LEN,
                              "readout": "last time step"}),
}
MODEL_NAMES = list(SPECS)


def build_model(name: str, hp: dict) -> nn.Module:
    return SPECS[name].build(hp)
