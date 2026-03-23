import torch
import torch.nn as nn
from attention import MultiHeadAttention
from add_norm import AddAndNormalize
from ffn import FeedForwardNetwork


class EncoderLayer(nn.Module):

    def __init__(
        self,
        model_dim: int = 512,
        hidden_dim: int = 2048,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.self_attention = MultiHeadAttention(model_dim, num_heads)
        self.attention_residual = AddAndNormalize(model_dim, dropout_rate)
        self.feed_forward = FeedForwardNetwork(model_dim, hidden_dim, dropout_rate)
        self.feed_forward_residual = AddAndNormalize(model_dim, dropout_rate)

    def forward(self, inputs, source_mask=None):
        attention_output, _ = self.self_attention(
            inputs, inputs, inputs, attention_mask=source_mask
        )
        inputs = self.attention_residual(inputs, attention_output)

        feed_forward_output = self.feed_forward(inputs)
        inputs = self.feed_forward_residual(inputs, feed_forward_output)

        return inputs


class Encoder(nn.Module):

    def __init__(
        self,
        model_dim: int = 512,
        hidden_dim: int = 2048,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    model_dim=model_dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, inputs, source_mask=None):
        for layer in self.layers:
            inputs = layer(inputs, source_mask)
        return inputs