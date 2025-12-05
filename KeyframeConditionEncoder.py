import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline, DDIMScheduler


class KeyframeConditionEncoder(nn.Module):
    def __init__(self, vae_latent_channels, conditioning_dim, num_tokens=16):
        super().__init__()
        self.context_dim = conditioning_dim
        self.num_tokens = num_tokens

        # Separate encoders for start and end frames
        self.pool_dim = int((num_tokens) ** 0.5)

        self.start_encoder = nn.Sequential(
            nn.Conv2d(vae_latent_channels, conditioning_dim, kernel_size=1),
            nn.AdaptiveAvgPool2d((self.pool_dim, self.pool_dim)),
        )
        self.end_encoder = nn.Sequential(
            nn.Conv2d(vae_latent_channels, conditioning_dim, kernel_size=1),
            nn.AdaptiveAvgPool2d((self.pool_dim, self.pool_dim)),
        )

        # Null tokens for classifier-free guidance (2*num_tokens because we have start + end)
        self.null_tokens = nn.Parameter(
            torch.zeros(1, 2 * num_tokens, conditioning_dim)
        )

    def forward(self, latent_start, latent_end):
        # Encode start and end frames separately
        start = self.start_encoder(latent_start)  # (B, C, H, W)
        end = self.end_encoder(latent_end)

        # Flatten to tokens
        b, c, h, w = start.shape
        start_tokens = start.view(b, c, h * w).permute(0, 2, 1)  # (B, num_tokens, dim)
        end_tokens = end.view(b, c, h * w).permute(0, 2, 1)  # (B, num_tokens, dim)

        # Concatenate along sequence dimension (not channel dimension)
        cond = torch.cat([start_tokens, end_tokens], dim=1)  # (B, 2*num_tokens, dim)
        return cond

    def null_embedding(self, batch_size):
        return self.null_tokens.expand(batch_size, -1, -1)
