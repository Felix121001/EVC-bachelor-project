import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (
    TransformerEncoder,
    TransformerEncoderLayer,
    Dropout,
    LayerNorm,
    Linear,
)
from torch.nn.functional import gelu


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_position_embeddings):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)

    def forward(self, input_tensor):
        seq_length = input_tensor.size(1)
        batch_size = input_tensor.size(0)
        hidden_size = (
            self.position_embeddings.embedding_dim
        )  

        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_tensor.device
        )

        position_embeddings = self.position_embeddings(
            position_ids
        )  # [seq_length, hidden_size]

        position_embeddings = position_embeddings.unsqueeze(0).expand(
            batch_size, -1, -1
        )

        return input_tensor + position_embeddings


class TransformerModel(nn.Module):
    def __init__(
        self,
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=4,
        intermediate_size=2048,
        hidden_dropout_prob=0.1,
        max_position_embeddings=512,
    ):
        super().__init__()

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention heads ({num_attention_heads})"
            )

        self.position_embeddings = PositionalEncoding(
            hidden_size=hidden_size, max_position_embeddings=max_position_embeddings
        )

        encoder_layers = TransformerEncoderLayer(
            hidden_size,
            num_attention_heads,
            dim_feedforward=intermediate_size,
            dropout=hidden_dropout_prob,
            activation=gelu,
            batch_first=True,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_hidden_layers)
        self.input_ln = LayerNorm(hidden_size)
        self.output_ln = LayerNorm(hidden_size)
        self.dropout = Dropout(hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask=None, class_enc=None):
        spectorgram = input_tensor[:, : self.num_cep]
        f0 = input_tensor[:, self.num_cep :]

        if self.position_embeddings is not None:
            input_tensor = self.position_embeddings(input_tensor)

        input_tensor = self.dropout(self.input_ln(input_tensor))

        if attention_mask is None:
            attention_mask = self.create_attention_mask(input_tensor)
        output = self.transformer_encoder(
            input_tensor, src_key_padding_mask=attention_mask
        )

        output = self.dropout(self.output_ln(output))
        return output

    def create_attention_mask(self, input_tensor):
        # Assuming padding is indicated by zeros
        attention_mask = input_tensor.sum(dim=-1) == 0  # [batch_size, seq_len]
        return attention_mask


import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm, Dropout
from torch.nn.functional import gelu


class PositionalEncoding2(nn.Module):
    def __init__(self, hidden_size, max_position_embeddings):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)

    def forward(self, input_tensor):
        seq_length = input_tensor.size(3)
        height = input_tensor.size(2)
        feature = input_tensor.size(1)
        batch_size = input_tensor.size(0)

        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_tensor.device
        )
        position_embeddings = self.position_embeddings(
            position_ids
        )  # [seq_length, hidden_size]

        position_embeddings = position_embeddings.transpose(0, 1).unsqueeze(
            0
        )  # [1, hidden_size, seq_length]
        position_embeddings = position_embeddings.expand(
            batch_size, feature, seq_length
        )  # [batch_size, feature, seq_length]

        position_embeddings = position_embeddings.unsqueeze(2).repeat(
            1, 1, height, 1
        )  # [batch_size, feature, height, seq_length]

        return input_tensor + position_embeddings


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=4,
        intermediate_size=2048,
        hidden_dropout_prob=0.1,
        max_position_embeddings=24,
    ):
        super().__init__()

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention heads ({num_attention_heads})"
            )

        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.position_embeddings = PositionalEncoding2(
            hidden_size=hidden_size, max_position_embeddings=max_position_embeddings
        )
        self.position_embeddings_class = PositionalEncoding2(
            hidden_size=4, max_position_embeddings=max_position_embeddings
        )

        encoder_layers = TransformerEncoderLayer(
            max_position_embeddings,
            num_attention_heads,
            dim_feedforward=intermediate_size,
            dropout=hidden_dropout_prob,
            activation=gelu,
            batch_first=True,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_hidden_layers)
        self.input_ln = LayerNorm(max_position_embeddings)
        self.output_ln = LayerNorm(max_position_embeddings)
        self.dropout = Dropout(hidden_dropout_prob)

    def forward(self, input_tensor, f0, class_enc, attention_mask=None):
        input_tensor_pos = self.position_embeddings(input_tensor)
        f0_pos = self.position_embeddings(f0)
        # class_enc_pos = self.position_embeddings_class(class_enc)
        class_enc_pos = class_enc

        batch_size = input_tensor.size(0)
        combined_input = torch.cat([input_tensor_pos, f0_pos, class_enc_pos], dim=1)
        combined_input = combined_input.view(
            batch_size, -1, self.max_position_embeddings
        )  # Reshape to 3D tensor

        combined_input = self.dropout(self.input_ln(combined_input))

        # Process through transformer encoder
        if attention_mask is None:
            attention_mask = self.create_attention_mask(combined_input)
        output = self.transformer_encoder(
            combined_input, src_key_padding_mask=attention_mask
        )
        output = self.dropout(self.output_ln(output))
        output = output.view(
            batch_size, 2 * self.hidden_size + 4, -1, self.max_position_embeddings
        )

        output_spectrogram = output[:, : self.hidden_size, :]
        output_f0 = output[:, self.hidden_size : -4, :]
        output_class_enc = output[:, -4:, :]

        return output_spectrogram, output_f0

    def create_attention_mask(self, input_tensor):
        attention_mask = input_tensor.sum(dim=-1) == 0  # [batch_size, seq_len]
        return attention_mask
