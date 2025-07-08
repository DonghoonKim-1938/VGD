import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa

__all__ = ["WordEmbedding"]

from .base_embedding import *
@register_embedding("word_embedding")
class WordEmbedding(nn.Module):

    def __init__(
            self,
            vocab_size,
            embed_dim,
            max_seq_length,
            padding_idx:int,
            drop_rate:int,
            positional_embedding:str=None
    ):
        super().__init__()
        self.padding_idx = padding_idx

        self.word_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=padding_idx
        )
        if positional_embedding is not None:
            self.pos_embedding = build_embedding(
                positional_embedding,
                **{"max_seq_length":max_seq_length,
                   "embed_dim":embed_dim,
                   }
            )
        # self.layer_norm = nn.LayerNorm(embed_dim,)
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.word_embedding(tokens)
        if hasattr(self, "pos_embedding"):
            x = self.pos_embedding(x)
        # x = self.layer_norm(x)    # activate for virtex/deactivate for clip
        x = self.dropout(x)

        # Zero-out embeddings for positions which have padding tokens.
        # shape: (batch_size, max_caption_length, 1)
        token_mask = (tokens != self.padding_idx).unsqueeze(-1)

        # shape: (batch_size, max_caption_length, hidden_size)
        x = x * token_mask.type(x.dtype)
        return x
