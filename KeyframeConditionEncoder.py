import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline, DDIMScheduler


class KeyframeConditionEncoder(nn.Module):
    def  __init__(self, latent_channels, context_dim, num_tokens = 16):
        super().__init__()
        self.context_dim = context_dim
        self.num_tokens = num_tokens
        
        self.conv = nn.Conv2d(2*latent_channels, context_dim, kernel_size=1)
        
        self.pool_dim = int((num_tokens) ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d((self.pool_dim, self.pool_dim))
        
    def forward(self, latent_start, latent_end):
        x = torch.cat([latent_start, latent_end], dim=1)
        x = self.conv(x)
        x = self.pool(x)
        b, c, h, w = x.shape
        x = x.view(b, c, h*w).permute(0, 2, 1)
        return x
    
    def null_embedding(self, batch_size):
        return torch.zeros(batch_size, self.num_tokens, self.context_dim, device = self.conv.weight.device)
    
    
    