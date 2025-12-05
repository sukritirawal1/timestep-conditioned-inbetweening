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

        # Use spatial convolutions to process spatial relationships
        # Gradually expand channels while maintaining spatial structure
        self.start_encoder = nn.Sequential(
            # First: process spatial features with 3x3 convs
            nn.Conv2d(
                vae_latent_channels, 64, kernel_size=3, padding=1
            ),  # (B, 4, 64, 64) -> (B, 64, 64, 64)
            nn.ReLU(),
            nn.Conv2d(
                64, 128, kernel_size=3, padding=1
            ),  # (B, 64, 64, 64) -> (B, 128, 64, 64)
            nn.ReLU(),
            # Reduce spatial dims gradually to preserve spatial relationships
            nn.Conv2d(
                128, 256, kernel_size=3, stride=2, padding=1
            ),  # (B, 128, 64, 64) -> (B, 256, 32, 32)
            nn.ReLU(),
            nn.Conv2d(
                256, 512, kernel_size=3, stride=2, padding=1
            ),  # (B, 256, 32, 32) -> (B, 512, 16, 16)
            nn.ReLU(),
            # Final adaptive pooling to desired token count
            nn.AdaptiveAvgPool2d((self.pool_dim, self.pool_dim)),  # (B, 512, 4, 4)
            # Project to conditioning dimension
            nn.Conv2d(
                512, conditioning_dim, kernel_size=1
            ),  # (B, 512, 4, 4) -> (B, 768, 4, 4)
        )

        self.end_encoder = nn.Sequential(
            nn.Conv2d(vae_latent_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((self.pool_dim, self.pool_dim)),
            nn.Conv2d(512, conditioning_dim, kernel_size=1),
        )

        # Null tokens for classifier-free guidance (2*num_tokens because we have start + end)
        self.null_tokens = nn.Parameter(
            torch.zeros(1, 2 * num_tokens, conditioning_dim)
        )

    def forward(self, latent_start, latent_end):
        # Encode with spatial convolutions
        start = self.start_encoder(latent_start)  # (B, 768, 4, 4)
        end = self.end_encoder(latent_end)  # (B, 768, 4, 4)

        # Flatten to tokens while preserving spatial order (row-major)
        # This maintains spatial relationships in the token sequence
        b, c, h, w = start.shape
        # Flatten spatial dims: each token corresponds to a spatial location
        start_tokens = start.flatten(2).transpose(1, 2)  # (B, h*w, c) = (B, 16, 768)
        end_tokens = end.flatten(2).transpose(1, 2)  # (B, 16, 768)

        # Concatenate along sequence dimension
        cond = torch.cat([start_tokens, end_tokens], dim=1)  # (B, 32, 768)
        return cond

    def null_embedding(self, batch_size):
        return self.null_tokens.expand(batch_size, -1, -1)
