import torch
import math


def build_causal_mask(sequence_length: int, device: str = "cpu"):
    lower_triangular = torch.tril(torch.ones(sequence_length, sequence_length, device=device))
    return lower_triangular.unsqueeze(0).unsqueeze(0)


def build_padding_mask(sequence, pad_token_id: int = 0):
    valid_tokens = (sequence != pad_token_id).unsqueeze(1).unsqueeze(2)
    return valid_tokens.float()


class PositionalEncoding(torch.nn.Module):
    def __init__(self, model_dim: int, max_length: int = 512, dropout_rate: float = 0.1):
        super().__init__()

        self.dropout_layer = torch.nn.Dropout(dropout_rate)

        encoding = torch.zeros(max_length, model_dim)
        positions = torch.arange(0, max_length).unsqueeze(1).float()

        scale_factor = torch.exp(
            torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim)
        )

        encoding[:, 0::2] = torch.sin(positions * scale_factor)
        encoding[:, 1::2] = torch.cos(positions * scale_factor)

        encoding = encoding.unsqueeze(0)
        self.register_buffer("positional_encoding", encoding)

    def forward(self, inputs):
        seq_len = inputs.size(1)
        inputs = inputs + self.positional_encoding[:, :seq_len, :]
        return self.dropout_layer(inputs)