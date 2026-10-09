import torch
import torch.nn as nn
from typing import Tuple

class ArielCore(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 512,
        embed_size: int = 300,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.encoder = nn.GRU(
            embed_size,
            hidden_size,
            batch_first=True,
            bidirectional=True,
        )

        self.rational_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self.decoder = nn.GRU(
            embed_size,
            hidden_size,
            batch_first=True,
        )

        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.embedding(x)
        enc_out, _ = self.encoder(emb)

        last = enc_out[:, -1, :]
        h_signal = self.rational_gate(last)

        decoder_hidden = enc_out[:, -1, :].mean(1).unsqueeze(0)  # [1, B, H]
        dec_input = torch.zeros(1, 1, self.hidden_size, device=x.device)
        dec_out, _ = self.decoder(dec_input, decoder_hidden)
        logits = self.out(dec_out.squeeze(1))        # [1, V]

        return logits, h_signal

    def decoder_forward(
        self,
        input_token: torch.LongTensor,
        hidden: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.LongTensor]:

        if input_token.dim() == 2:
            embedded = self.embedding(input_token)
        else:
            embedded = input_token

        dec_out, hidden = self.decoder(embedded, hidden)
        output = self.out(dec_out.squeeze(1))
        return output, hidden
