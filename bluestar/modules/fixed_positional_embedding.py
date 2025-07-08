import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa

__all__ = ["FixedPositionalEmbedding"]

class FixedPositionalEmbedding(nn.Module):

    def __init__(self, vocab_size, embed_dim, max_seq_length, padding_idx, drop_rate):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=padding_idx)
        # self.pos_embedding = nn.Embedding(num_embeddings=max_seq_length, embedding_dim=embed_dim)
        self.pos_embedding = torch.zeros(max_seq_length, embed_dim)
        self.pos_embedding.requires_grad = False

        position = torch.arange(0, max_seq_length).unsqueeze(1)
        i2ndex = torch.arange(0,embed_dim, step=2)
        self.pos_embedding[:, 0::2] = torch.sin(position / 10000 ** (i2ndex / embed_dim))
        self.pos_embedding[:, 1::2] = torch.cos(position / 10000 ** (i2ndex / embed_dim))

        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        batch, seq_len = x.shape
        # out : (batch, seq_len, embedding_dim)
        pos = self.pos_embedding[:seq_len,:].unsqueeze(0).repeat(batch, 1, 1).to(x.device)
        x = self.dropout(self.embedding(x) + pos)
        return x
