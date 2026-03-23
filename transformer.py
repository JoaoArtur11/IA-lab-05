import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from utils import PositionalEncoding


class Transformer(nn.Module):

    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        model_dim: int = 128,
        hidden_dim: int = 512,
        num_heads: int = 4,
        num_layers: int = 2,
        max_length: int = 128,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.source_embedding = nn.Embedding(source_vocab_size, model_dim)
        self.target_embedding = nn.Embedding(target_vocab_size, model_dim)

        self.source_positional_encoding = PositionalEncoding(
            model_dim, max_length, dropout_rate
        )
        self.target_positional_encoding = PositionalEncoding(
            model_dim, max_length, dropout_rate
        )

        self.encoder = Encoder(
            model_dim=model_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
        )

        self.decoder = Decoder(
            model_dim=model_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
        )

        self.output_projection = nn.Linear(model_dim, target_vocab_size)

        self._initialize_weights()

    def _initialize_weights(self):
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(
        self,
        source_sequence,
        target_sequence,
        source_mask=None,
        target_mask=None,
    ):
        source_embeddings = self.source_positional_encoding(
            self.source_embedding(source_sequence)
        )
        target_embeddings = self.target_positional_encoding(
            self.target_embedding(target_sequence)
        )

        encoder_outputs = self.encoder(source_embeddings, source_mask)

        decoder_outputs = self.decoder(
            target_embeddings,
            encoder_outputs,
            target_mask,
            source_mask,
        )

        logits = self.output_projection(decoder_outputs)
        return logits

    def encode(self, source_sequence, source_mask=None):
        source_embeddings = self.source_positional_encoding(
            self.source_embedding(source_sequence)
        )
        return self.encoder(source_embeddings, source_mask)

    def decode(
        self,
        target_sequence,
        encoder_outputs,
        target_mask=None,
        source_mask=None,
    ):
        target_embeddings = self.target_positional_encoding(
            self.target_embedding(target_sequence)
        )
        return self.decoder(
            target_embeddings,
            encoder_outputs,
            target_mask,
            source_mask,
        )