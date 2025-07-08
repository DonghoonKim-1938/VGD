import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
from .base_embedding import register_embedding

__all__ = ["AbsolutePositionalEmbedding"]

@register_embedding("absolute_positional_embedding")
class AbsolutePositionalEmbedding(nn.Module):

    def __init__(self,
                 max_seq_length,
                 embed_dim
                 ):
        super().__init__()

        self.pos_embedding = nn.Embedding(
            num_embeddings=max_seq_length,
            embedding_dim=embed_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        pos = torch.arange(0, seq_len).to(x.device)
        x = x + self.pos_embedding(pos)
        return x