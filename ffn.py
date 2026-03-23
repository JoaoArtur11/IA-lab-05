import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardNetwork(nn.Module):

    def __init__(self, model_dim: int, hidden_dim: int, dropout_rate: float = 0.1):
        super().__init__()

        self.input_projection = nn.Linear(model_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, model_dim)
        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        hidden = self.input_projection(inputs)
        activated = F.relu(hidden)
        dropped = self.dropout_layer(activated)
        return self.output_projection(dropped)