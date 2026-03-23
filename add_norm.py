import torch
import torch.nn as nn


class AddAndNormalize(nn.Module):

    def __init__(self, model_dim: int, dropout_rate: float = 0.1):
        super().__init__()

        self.layer_norm = nn.LayerNorm(model_dim)
        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, inputs, residual_output):
        combined = inputs + self.dropout_layer(residual_output)
        return self.layer_norm(combined)