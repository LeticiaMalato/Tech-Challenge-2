"""Arquitetura Matrix Factorization com bias para recomendação em dados esparsos."""

import torch
import torch.nn as nn


class MatrixFactorizationNet(nn.Module):
    """Matrix Factorization com bias treinada via BCE.

    Implementa score(u, i) = <u_emb, i_emb> + b_u + b_i.
    Escolhida em vez de MLP porque datasets de e-commerce são esparsos
    (poucos exemplos por item), onde camadas ocultas tendem a overfittar.
    É equivalente ao SVD mas treinada discriminativamente com negativos.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 32,
    ) -> None:
        """Inicializa os embeddings e bias de usuários e itens.

        Args:
            n_users: Tamanho do vocabulário de usuários.
            n_items: Tamanho do vocabulário de itens.
            embed_dim: Dimensão dos vetores de embedding latente.
        """
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        """Calcula logit = <u, i> + b_u + b_i.

        Args:
            user_idx: Tensor de índices de usuário, shape (batch,).
            item_idx: Tensor de índices de item, shape (batch,).

        Returns:
            Logits de shape (batch,) sem sigmoid aplicado.
        """
        u = self.user_emb(user_idx)
        i = self.item_emb(item_idx)
        bu = self.user_bias(user_idx).squeeze(-1)
        bi = self.item_bias(item_idx).squeeze(-1)
        return (u * i).sum(dim=-1) + bu + bi
