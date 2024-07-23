import torch
import torch.nn as nn

import dataloader as dataloader


class LanguageModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=256,
        hidden_size=512,
        n_layers=4,
        dropout_p=.3,
        max_length=255
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        super().__init__()

        self.emb = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=dataloader.PAD
        )
        self.rnn = nn.LSTM(
            embedding_dim,
            hidden_size,
            n_layers,
            batch_first=True,
            dropout=dropout_p
        )
        self.out = nn.Linear(hidden_size, vocab_size, bias=True)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, *args):
        # |x| = (batch_size, length)
        x = self.emb(x)
        # |x| = (batch_size, length, embedding_dim)
        x, _ = self.rnn(x)
        # |x| = (batch_size, length, hidden_size)
        x = self.out(x)
        # |x| = (batch_size, length, vocab_size)
        y_hat = self.log_softmax(x)

        return y_hat