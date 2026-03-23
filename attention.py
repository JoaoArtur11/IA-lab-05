import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def compute_scaled_dot_product_attention(query, key, value, attention_mask=None):
    key_dim = query.size(-1)

    attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(key_dim)

    if attention_mask is not None:
        attention_scores = attention_scores.masked_fill(attention_mask == 0, float("-inf"))

    attention_weights = F.softmax(attention_scores, dim=-1)
    attention_weights = torch.nan_to_num(attention_weights, nan=0.0)

    context = torch.matmul(attention_weights, value)
    return context, attention_weights


class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()

        assert model_dim % num_heads == 0, "model_dim deve ser divisível por num_heads"

        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        self.query_projection = nn.Linear(model_dim, model_dim, bias=False)
        self.key_projection = nn.Linear(model_dim, model_dim, bias=False)
        self.value_projection = nn.Linear(model_dim, model_dim, bias=False)
        self.output_projection = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, query, key, value, attention_mask=None):
        batch_size = query.size(0)

        query = self.query_projection(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_projection(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_projection(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attention_output, attention_weights = compute_scaled_dot_product_attention(
            query, key, value, attention_mask
        )

        attention_output = (
            attention_output
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.head_dim)
        )

        final_output = self.output_projection(attention_output)
        return final_output, attention_weights