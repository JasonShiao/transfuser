import torch
import torch.nn as nn

class TransformerWorldModel(nn.Module):
    def __init__(self, latent_dim, seq_len, nhead=4, num_layers=3, dim_feedforward=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.seq_len = seq_len

        self.pos_embedding = nn.Parameter(torch.randn(seq_len, latent_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(latent_dim, latent_dim)  # Predict next z

    def forward(self, z_seq):
        """
        z_seq: (B, T, D) where T == self.seq_len
        Returns:
          z_next_pred: (B, D)
        """
        z_seq = z_seq + self.pos_embedding.unsqueeze(0)  # Add positional encoding
        h = self.transformer(z_seq)
        return self.output_proj(h[:, -1])  # Only predict based on last hidden state
