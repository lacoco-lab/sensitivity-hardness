import torch

from tqdm import tqdm, trange
from typing import Tuple


# simple Transformer model
class TorchModel(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_attn_heads: int,
        num_layers: int,
        dropout_prob: float = 0.,
        max_len=1024):
        super().__init__()

        self.embedding = torch.nn.Embedding(2, hidden_dim // 2)
        self.positional_encoding = torch.nn.Embedding(max_len, hidden_dim // 2)

        self.encoders = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_attn_heads, dim_feedforward=hidden_dim, 
                dropout=dropout_prob, activation='relu', batch_first=True
            ), num_layers=num_layers
        )

        self.linear_head = torch.nn.Linear(hidden_dim, 1, bias=True)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None, 
            attn_mask: torch.Tensor = None
        ) -> torch.Tensor:

        positional = self.positional_encoding(torch.arange(x.size(1), device=x.device)).repeat(x.size(0), 1, 1)
        x = torch.cat([self.embedding(x), positional], dim=-1)
        x = self.encoders(x, src_key_padding_mask=key_padding_mask, mask=attn_mask)
        return self.linear_head(x)[:, -1, :].view(-1)


# encoder-decoder Transformer model with a scratchpad
class TorchModelWithScratchpad(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_attn_heads: int,
        num_layers: int,
        dropout_prob: float = 0.,
        max_len=1024):
        super().__init__()

        self.embedding = torch.nn.Embedding(3, hidden_dim // 2)
        self.positional_encoding = torch.nn.Embedding(max_len + 1, hidden_dim // 2)

        self.model = torch.nn.Transformer(
            d_model=hidden_dim, nhead=num_attn_heads, num_encoder_layers=num_layers,
            num_decoder_layers=num_layers, dim_feedforward=hidden_dim, dropout=dropout_prob,
            activation='relu', batch_first=True
        )

        self.linear_head = torch.nn.Linear(hidden_dim, 1, bias=True)

    def prepare_tensors(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_positional = self.positional_encoding(torch.arange(x.size(1), device=x.device)).repeat(x.size(0), 1, 1)
        x = torch.cat([self.embedding(x), x_positional], dim=-1)

        y_positional = self.positional_encoding(torch.arange(y.size(1), device=y.device)).repeat(y.size(0), 1, 1)
        y = torch.cat([self.embedding(y), y_positional], dim=-1)

        return x, y

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x, y = self.prepare_tensors(x, y)
        tgt_mask = self.model.generate_square_subsequent_mask(y.size(1)).to(y.device)
        out = self.model(x, y, tgt_mask=tgt_mask)

        return self.linear_head(out).squeeze(-1)
    
    # autoregressive answer generation using scratchpad
    def autoregressive_answer(self, x: torch.Tensor, verbose=False) -> torch.Tensor:
        y = torch.ones((x.size(0), 1), device=x.device, dtype=int) * 2

        for i in trange(x.size(1), disable=not verbose):
            output = self(x, y)
            y = torch.cat([y, (output[:, -1:] > 0.5).long()], dim=1)

        return y[:, 1:]