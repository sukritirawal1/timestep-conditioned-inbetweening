import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, DDIMScheduler
from transformers import CLIPVisionModelWithProjection, CLIPProcessor
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc

from dataloader import AnitaDataset
from torch.utils.data import DataLoader
import argparse
from optical_flow import OpticalFlow


class OpticalFlowConditionedDiffusionV3(nn.Module):
    def __init__(self, model_name="runwayml/stable-diffusion-v1-5", val_visualization_dir="val_viz", device=None):
        super().__init__()
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.val_vis_dir = val_visualization_dir
        
        # Load Stable Diffusion
        pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32).to(self.device)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        
        self.vae = pipe.vae
        self.unet = pipe.unet
        self.image_processor = pipe.image_processor
        self.scheduler = pipe.scheduler
        
        # Load CLIP for visual features
        self.clip_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
        # Visual adapter to project CLIP features
        context_dim = self.unet.config.cross_attention_dim
        self.visual_adapter = nn.Sequential(
            nn.Linear(768, context_dim),
            nn.LayerNorm(context_dim),
            nn.ReLU(),
            nn.Linear(context_dim, context_dim),
        )
        
        # Optical flow
        self.optical_flow = OpticalFlow()
        
        # Expand UNet input from 4 to 8 channels
        old_conv = self.unet.conv_in
        new_conv = nn.Conv2d(8, old_conv.out_channels, kernel_size=old_conv.kernel_size, 
                            stride=old_conv.stride, padding=old_conv.padding)
        with torch.no_grad():
            new_conv.weight[:, :4] = old_conv.weight.clone()
            new_conv.weight[:, 4:] = old_conv.weight.clone() * 0.1
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)
        self.unet.conv_in = new_conv
        
        # Freeze stuff we don't train
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        self.clip_model.eval()

    def encode_image(self, images):
        if images.ndim == 3:
            images = [images]
        elif images.ndim == 4:
            images = [img for img in images]
        img_list = [self.image_processor.preprocess(x) for x in images]
        img_batch = torch.cat(img_list, dim=0).to(device=self.device, dtype=self.vae.dtype)
        with torch.no_grad():
            latents = self.vae.encode(img_batch).latent_dist.sample() * self.vae.config.scaling_factor
        return latents

    def decode_latent(self, latent):
        with torch.no_grad():
            img = self.vae.decode(latent / self.vae.config.scaling_factor).sample
        img = (img / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
        img = (img * 255).round().astype("uint8")
        return [Image.fromarray(image) for image in img]

    def get_clip_features(self, images):
        inputs = self.clip_processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            patch_features = outputs.last_hidden_state[:, 1:, :]  # Skip CLS token
            patch_768 = self.clip_model.visual_projection(patch_features)
        return patch_768

    def get_visual_conditioning(self, start_frames, end_frames):
        start_feat = self.get_clip_features(start_frames)
        end_feat = self.get_clip_features(end_frames)
        combined = torch.cat([start_feat, end_feat], dim=1)
        return self.visual_adapter(combined)

    def get_optical_flow_latent(self, start_frames, end_frames, timestep):
        batch_size = start_frames.shape[0]
        flow_frames = []
        for i in range(batch_size):
            flow_frame = self.optical_flow.interpolate(start_frames[i], end_frames[i], t=timestep)
            flow_frames.append(flow_frame)
        flow_batch = torch.stack(flow_frames).to(self.device)
        with torch.no_grad():
            flow_latents = self.encode_image(flow_batch)
        return flow_latents

    def slerp(self, z0, z1, alpha):
        v0 = z0.reshape(z0.shape[0], -1)
        v1 = z1.reshape(z1.shape[0], -1)
        dot = torch.sum(v0 * v1, dim=1) / (v0.norm(dim=1) * v1.norm(dim=1))
        dot = torch.clamp(dot, -1.0, 1.0)
        
        if not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha, device=z0.device, dtype=z0.dtype)
        alpha = alpha.view(1, 1)
        
        theta = torch.arccos(dot)
        sin_theta = torch.sin(theta)
        s0 = torch.sin((1 - alpha) * theta) / sin_theta
        s1 = torch.sin(alpha * theta) / sin_theta
        
        v2 = s0.unsqueeze(1) * v0 + s1.unsqueeze(1) * v1
        return v2.reshape_as(z0)

    def training_step(self, batch):
        start = batch["anchor_start"].to(self.device)
        end = batch["anchor_end"].to(self.device)
        targets = batch["targets"].to(self.device)

        idx = np.random.randint(0, targets.shape[1])
        timestep = (idx + 1) / (targets.shape[1] + 1)
        target = targets[:, idx, :, :, :]

        with torch.no_grad():
            z_target = self.encode_image(target)
            flow_latents = self.get_optical_flow_latent(start, end, timestep)

        # Add noise for diffusion
        t = torch.randint(0, self.scheduler.config.num_train_timesteps, (z_target.shape[0],), device=self.device).long()
        noise = torch.randn_like(z_target)
        zt_noisy = self.scheduler.add_noise(z_target, noise, t)

        # Concatenate noisy latent + optical flow
        unet_input = torch.cat([zt_noisy, flow_latents], dim=1)

        # Get visual conditioning
        visual_cond = self.get_visual_conditioning(start, end)

        # Predict noise
        noise_pred = self.unet(unet_input, t, encoder_hidden_states=visual_cond).sample
        loss = F.mse_loss(noise_pred, noise)
        
        return loss

    def train_model(self, train_loader, val_loader, num_epochs=5, lr=1e-4, save_dir="model_ckpt"):
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float("inf")
        
        for epoch in range(num_epochs):
            self.train()
            train_loss = 0.0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                optimizer.zero_grad()
                loss = self.training_step(batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")
            
            # Validation
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for i, batch in enumerate(tqdm(val_loader, desc="Validation")):
                    loss = self.training_step(batch)
                    val_loss += loss.item()
                    if i == 0:
                        self.visualize_samples(batch, epoch)
            
            avg_val_loss = val_loss / len(val_loader)
            print(f"Validation Loss: {avg_val_loss:.4f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.state_dict(), os.path.join(save_dir, "best_model.pth"))
            
            torch.cuda.empty_cache()
            gc.collect()

    @torch.no_grad()
    def predict_frame(self, start_frames, end_frames, timestep, num_steps=25, noise_strength=0.2, guidance_scale=2.0):
        z0 = self.encode_image(start_frames)
        z1 = self.encode_image(end_frames)
        z_init = self.slerp(z0, z1, timestep)
        
        flow_latents = self.get_optical_flow_latent(start_frames, end_frames, timestep)
        visual_cond = self.get_visual_conditioning(start_frames, end_frames)
        
        self.scheduler.set_timesteps(num_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        start_idx = int(noise_strength * (len(timesteps) - 1))
        
        noise = torch.randn_like(z_init)
        latents = self.scheduler.add_noise(z_init, noise, timesteps[start_idx])
        
        for t in timesteps[start_idx:]:
            latent_with_flow = torch.cat([latents, flow_latents], dim=1)
            latent_input = self.scheduler.scale_model_input(latent_with_flow, t)
            
            # CFG
            uncond = torch.zeros_like(visual_cond)
            cond_input = torch.cat([uncond, visual_cond], dim=0)
            latent_input = torch.cat([latent_input, latent_input], dim=0)
            
            noise_pred = self.unet(latent_input, t, encoder_hidden_states=cond_input).sample
            noise_uncond, noise_cond = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        return self.decode_latent(latents)

    def visualize_samples(self, batch, epoch):
        os.makedirs(self.val_vis_dir, exist_ok=True)
        start = batch["anchor_start"].to(self.device)
        end = batch["anchor_end"].to(self.device)
        targets = batch["targets"].to(self.device)
        B, N, _, _, _ = targets.shape
        
        for i in range(min(B, 4)):
            gt_frames = [self._to_pil(start[i])]
            pred_frames = [self._to_pil(start[i])]
            
            for t_idx in range(N):
                timestep = (t_idx + 1) / (N + 1)
                gt_frames.append(self._to_pil(targets[i, t_idx]))
                pred = self.predict_frame(start[i:i+1], end[i:i+1], timestep)
                pred_frames.append(pred[0])
            
            gt_frames.append(self._to_pil(end[i]))
            pred_frames.append(self._to_pil(end[i]))
            
            fig, axes = plt.subplots(2, N+2, figsize=(3*(N+2), 6))
            for col in range(N+2):
                axes[0, col].imshow(gt_frames[col])
                axes[0, col].axis("off")
                axes[1, col].imshow(pred_frames[col])
                axes[1, col].axis("off")
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.val_vis_dir, f"epoch_{epoch+1:03d}_sample_{i}.png"))
            plt.close()

    @staticmethod
    def _to_pil(tensor):
        img = tensor.detach().cpu().clamp(0, 1)
        img = (img * 255).byte().permute(1, 2, 0).numpy()
        return Image.fromarray(img)


def main(args):
    print("=== V3: Optical Flow + CLIP Visual Conditioning ===")
    
    train_data = AnitaDataset(os.path.join(args.data_dir, "train"), image_shape=(224, 224))
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_data = AnitaDataset(os.path.join(args.data_dir, "val"), image_shape=(224, 224))
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = OpticalFlowConditionedDiffusionV3(val_visualization_dir=args.val_dir).to(device)
    
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint), strict=False)
    
    # Freeze UNet except conv_in and cross-attention
    for param in model.unet.parameters():
        param.requires_grad = False
    for param in model.unet.conv_in.parameters():
        param.requires_grad = True
    for name, param in model.unet.named_parameters():
        if "attn2" in name:
            param.requires_grad = True
    for param in model.visual_adapter.parameters():
        param.requires_grad = True
    
    print(f"Training {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
    
    model.train_model(train_loader, val_loader, num_epochs=args.epochs, lr=args.lr, save_dir=args.save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="model_ckpt_v3")
    parser.add_argument("--val_dir", type=str, default="val_viz_v3")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    main(args)
