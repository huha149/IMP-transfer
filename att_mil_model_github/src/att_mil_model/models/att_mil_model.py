from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn


class AttentionModule(nn.Module):
    """Gated Attention module for MIL.

    参考：gated attention MIL（Tanh + Sigmoid gating），用于计算每个 instance 的注意力分数。
    - 输入：instance 特征 (N, input_dim)
    - 输出：bag 特征 (1, embed_dim_reduction) 与注意力权重 (1, N)

    注意：
    - 本实现默认对注意力 logits 做 softmax 归一化（更常见、也更稳定）。
    - 若你希望复现“未归一化直接加权求和”的行为，可将 normalize_attention=False。
    """

    def __init__(
        self,
        input_dim: int = 2048,
        embed_dim_reduction: int = 256,
        embed_dim: int = 256,
        dropout: bool = True,
        dropout_p: float = 0.05,
        normalize_attention: bool = True,
    ) -> None:
        super().__init__()
        self.normalize_attention = normalize_attention

        # 特征降维（H -> H_reduced）
        self.fc_dim_reduction = nn.Linear(input_dim, embed_dim_reduction)

        # V 分支（Tanh）
        v_layers = [nn.Linear(embed_dim_reduction, embed_dim), nn.Tanh()]
        if dropout:
            v_layers.append(nn.Dropout(dropout_p))
        self.attention_V = nn.Sequential(*v_layers)

        # U 分支（Sigmoid）
        u_layers = [nn.Linear(embed_dim_reduction, embed_dim), nn.Sigmoid()]
        if dropout:
            u_layers.append(nn.Dropout(dropout_p))
        self.attention_U = nn.Sequential(*u_layers)

        # 输出注意力 logits（每个 instance 一个标量）
        self.attention_weights = nn.Linear(embed_dim, 1)

    def forward(self, instance_feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward.

        Args:
            instance_feature: shape (N, input_dim)

        Returns:
            bag_feature: shape (1, embed_dim_reduction)
            attention: shape (1, N)  (softmax 后的注意力权重或原始权重)
        """
        if instance_feature.dim() != 2:
            raise ValueError(
                f"instance_feature must be 2D Tensor (N, D), got shape={tuple(instance_feature.shape)}"
            )

        h = self.fc_dim_reduction(instance_feature)  # (N, embed_dim_reduction)
        a_v = self.attention_V(h)                    # (N, embed_dim)
        a_u = self.attention_U(h)                    # (N, embed_dim)

        # gated attention logits: (N, 1)
        a_logits = self.attention_weights(a_v * a_u)

        if self.normalize_attention:
            a = torch.softmax(a_logits, dim=0)       # (N, 1)
        else:
            a = a_logits

        # bag feature: (1, embed_dim_reduction)
        bag_feature = torch.sum(a * h, dim=0, keepdim=True)

        # (1, N) 便于后续可视化/保存
        return bag_feature, a.transpose(0, 1)


@dataclass
class ModelOutput:
    """模型输出，便于阅读与类型提示。"""

    logits: torch.Tensor
    bag_feature: torch.Tensor
    attention: torch.Tensor


class att_mil_model(nn.Module):
    """Attention-based MIL model (Classification / Regression).

    参数说明：
    - Task=1: 回归任务，输出维度固定为 1
    - Task=2: 分类任务，输出维度为 output_dim（类别数）

    输入张量形状建议：
    - (1, N, input_dim)  (DataLoader batch_size=1 的常见输出)
    - 或 (N, input_dim)
    """

    def __init__(
        self,
        input_dim: int = 2048,
        embed_dim_reduction: int = 512,
        embed_dim: int = 256,
        Task: int = 2,
        output_dim: int = 3,
        attention_dropout: bool = True,
        normalize_attention: bool = True,
    ) -> None:
        super().__init__()
        if Task not in (1, 2):
            raise ValueError("Task must be 1 (regression) or 2 (classification).")

        self.input_dim = input_dim
        self.embed_dim_reduction = embed_dim_reduction
        self.embed_dim = embed_dim
        self.Task = Task
        self.output_dim = 1 if Task == 1 else output_dim

        self.attention_net = AttentionModule(
            input_dim=input_dim,
            embed_dim_reduction=embed_dim_reduction,
            embed_dim=embed_dim,
            dropout=attention_dropout,
            normalize_attention=normalize_attention,
        )

        self.fc = nn.Linear(embed_dim_reduction, self.output_dim)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """Forward.

        Args:
            x: (1, N, D) or (N, D)

        Returns:
            ModelOutput(logits, bag_feature, attention)
        """
        if x.dim() == 3:
            # (B, N, D) -> (B*N, D); 目前实现假设 B=1
            x = x.view(-1, x.size(-1))
        elif x.dim() != 2:
            raise ValueError(f"Expected 2D/3D Tensor, got shape={tuple(x.shape)}")

        bag_feature, attention = self.attention_net(x)
        logits = self.fc(bag_feature)

        return ModelOutput(logits=logits.float(), bag_feature=bag_feature, attention=attention)
