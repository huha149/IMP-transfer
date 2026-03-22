import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttentionNet(nn.Module):
    """
    Gated attention block used to compute instance attention scores.
    """

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 256, dropout: bool = False):
        super().__init__()
        attention_v = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        attention_u = [nn.Linear(input_dim, hidden_dim), nn.Sigmoid()]

        if dropout:
            attention_v.append(nn.Dropout(0.25))
            attention_u.append(nn.Dropout(0.25))

        self.attention_v = nn.Sequential(*attention_v)
        self.attention_u = nn.Sequential(*attention_u)
        self.attention_weights = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.attention_v(x)
        b = self.attention_u(x)
        return self.attention_weights(a * b)


class AttentionMILModel(nn.Module):
    """
    Attention-based MIL model for classification or regression.

    Args:
        input_dim: Dimension of input instance features.
        embed_dim: Dimension of projected bag representation.
        task: 1 for regression, 2 for classification.
        output_dim: Number of outputs for classification.
        dropout: Whether to use dropout in the attention block.
        shuffle: Whether to shuffle attention weights during training.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        embed_dim: int = 256,
        task: int = 2,
        output_dim: int = 2,
        dropout: bool = True,
        shuffle: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.task = task
        self.output_dim = output_dim
        self.shuffle = shuffle

        self.attention_net = GatedAttentionNet(
            input_dim=input_dim,
            hidden_dim=embed_dim,
            dropout=dropout,
        )
        self.projector = nn.Linear(input_dim, embed_dim)
        out_features = 1 if task == 1 else output_dim
        self.prediction = nn.Linear(embed_dim, out_features)

    def forward(self, instance_feature: torch.Tensor):
        """
        Args:
            instance_feature: Tensor shaped `(N, D)` or `(1, N, D)`.

        Returns:
            logits: Prediction logits.
            bag_feature: Aggregated bag representation.
            attention_scores: Raw attention scores before softmax.
        """
        if instance_feature.dim() == 3:
            instance_feature = instance_feature.squeeze(0)

        x = instance_feature.view(-1, instance_feature.size(-1))
        attention_scores = self.attention_net(x).transpose(1, 0)
        attention_weights = F.softmax(attention_scores, dim=1)

        if self.shuffle and self.training:
            permutation = torch.randperm(attention_weights.size(1), device=attention_weights.device)
            attention_weights = attention_weights[:, permutation]

        x_proj = self.projector(x)
        bag_feature = torch.mm(attention_weights, x_proj)
        logits = self.prediction(bag_feature)
        return logits.float(), bag_feature, attention_scores


# Backward-compatible alias.
att_mil_model = AttentionMILModel
