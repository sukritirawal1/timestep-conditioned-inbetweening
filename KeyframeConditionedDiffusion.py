import os
import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from KeyframeConditionEncoder import KeyframeConditionEncoder
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

from dataloader import AnitaDataset
from torch.utils.data import DataLoader
import argparse

class KeyframeConditionedDiffusion(nn.Module):
    def __init__(self, model_name="runwayml/stable-diffusion-v1-5", val_visualization_dir = "val_viz", device=None):
        super().__init__()
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32 #if self.device == "cpu" else torch.float16
        self.val_vis_dir = val_visualization_dir
        pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=dtype
        ).to(self.device)
        
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        
        self.vae = pipe.vae
        self.unet = pipe.unet
        self.image_processor = pipe.image_processor
        self.scheduler = pipe.scheduler
        
        latent_channels = self.vae.config.latent_channels
        context_dim = self.unet.config.cross_attention_dim
        self.image_conditioner = KeyframeConditionEncoder(latent_channels=latent_channels,
                                                          context_dim = context_dim,
                                                          num_tokens = 64).to(self.device)
        
        pipe.safety_checker = None
        self.pipe = pipe
    
   
    
    def encode_image_to_latent(self, images):
        # if images.isinstance(list):
        #     pass
        if images.ndim == 3:
            images = [images]
        elif images.ndim == 4:
            images = [img for img in images]
            
        img_list = [self.image_processor.preprocess(x) for x in images]
        img_batch = torch.cat(img_list, dim = 0)
        img_batch = img_batch.to(device=self.device, dtype=self.vae.dtype)
        with torch.no_grad():
            dist = self.vae.encode(img_batch).latent_dist
            latents = dist.sample() * self.vae.config.scaling_factor
        return latents
    
    def decode_latent_to_image(self, latent):
        with torch.no_grad():
            latents = latent/self.vae.config.scaling_factor
            img = self.vae.decode(latents).sample
        img = (img / 2 + 0.5).clamp(0, 1)
        img = img.cpu().permute(0, 2, 3, 1).numpy()
        img = (img * 255).round().astype("uint8")
        
        return [Image.fromarray(image) for image in img]
    
    def get_keyframe_condition(self, start_frames, end_frames):
        with torch.no_grad():
            z_start = self.encode_image_to_latent(start_frames)
            z_end = self.encode_image_to_latent(end_frames)
        cond = self.image_conditioner(z_start, z_end)
        return cond
    
    def training_step(self, batch):
        start = batch["anchor_start"].to(self.device)
        end   = batch["anchor_end"].to(self.device)
        targets = batch["targets"].to(self.device)
        
        idx = np.random.randint(0, targets.shape[1])
        target = targets[:, idx, :, :, :]
        
        with torch.no_grad():
            z0 = self.encode_image_to_latent(target)
            
        #sample timestep
        num_timesteps = self.scheduler.config.num_train_timesteps
        t = torch.randint(0, num_timesteps, (z0.shape[0],), device=self.device).long()
        
        noise = torch.randn_like(z0)
        z0_noisy = self.scheduler.add_noise(z0, noise, t)
        
        keyframe_cond = self.get_keyframe_condition(start, end)
        
        noise_pred = self.unet(
            z0_noisy,
            t,
            encoder_hidden_states=keyframe_cond,
        ).sample
        
        loss = F.mse_loss(noise_pred, noise)
        return loss
    
    def val_metrics(self, val_loader, epoch):
        self.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, batch in tqdm(enumerate(val_loader), desc="Validation"):
                loss = self.training_step(batch)
                val_loss += loss.item()
                if i == 0:
                    pred_seq = self.predict_inbetween_sequence(
                        batch["anchor_start"].to(self.device),
                        batch["anchor_end"].to(self.device),
                        num_inbetweens=batch["targets"].shape[1],
                    )   
                    self.visualize_inbetweens(batch, pred_seq, epoch)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        return avg_val_loss
    
    def visualize_inbetweens(self, batch, pred_seq, epoch):
        os.makedirs(self.val_vis_dir, exist_ok=True)
        self.eval()
        start = batch["anchor_start"].to(self.device)
        end = batch["anchor_end"].to(self.device)
        targets = batch["targets"].to(self.device)
        B, N, _, _, _ = targets.shape
        
        for i in range(B):
            start_i = start[i]          # (3, H, W)
            end_i   = end[i]            # (3, H, W)
            targets_i = targets[i]      # (T, 3, H, W)

            # --- Build GT row: [start, gt_0, gt_1, ..., gt_{T-1}, end] ---
            gt_frames = [self._tensor_to_pil(start_i)]
            for t_idx in range(T):
                gt_frames.append(self._tensor_to_pil(targets_i[t_idx]))
            gt_frames.append(self._tensor_to_pil(end_i))

            # --- Build predicted row: [start, pred_0, pred_1, ..., pred_{T-1}, end] ---
            pred_frames = [self._tensor_to_pil(start_i)]
            for t_idx in range(N):
                pred_img_i = pred_seq[t_idx][i]  # frame t_idx, batch index i
                pred_frames.append(pred_img_i)
            pred_frames.append(self._tensor_to_pil(end_i))

            ncols = N + 2
            fig, axes = plt.subplots(2, ncols, figsize=(3 * ncols, 6))

            for col in range(ncols):
                # Top row: GT
                axes[0, col].imshow(gt_frames[col])
                axes[0, col].axis("off")
                if col == 0:
                    axes[0, col].set_title("Start (GT)")
                elif col == ncols - 1:
                    axes[0, col].set_title("End (GT)")
                else:
                    axes[0, col].set_title(f"GT {col}")

                # Bottom row: Pred
                axes[1, col].imshow(pred_frames[col])
                axes[1, col].axis("off")
                if col == 0:
                    axes[1, col].set_title("Start (Pred)")
                elif col == ncols - 1:
                    axes[1, col].set_title("End (Pred)")
                else:
                    axes[1, col].set_title(f"Pred {col}")

            fig.suptitle(f"Epoch {epoch+1} – Sample {i}", fontsize=14)
            fig.tight_layout()

            out_path = os.path.join(self.val_vis_dir, f"epoch_{epoch+1:03d}_sample_{i}.png")
            plt.savefig(out_path)
            plt.close(fig)
    
    def train_model(self, train_loader, val_loader, num_epochs = 5, lr = 1e-4, save_dir = "model_ckpt"):
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float("inf")
        for epoch in range(num_epochs):
            self.train()
            train_loss = 0.0
            for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
                optimizer.zero_grad()
                loss = self.training_step(batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader)
            print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")
            val_loss = self.val_metrics(val_loader, epoch)
            if val_loss <= best_val_loss:
                torch.save(self.state_dict(), os.path.join(save_dir, f"best_model.pth"))
        
    
    def cfg_forward(self, latents, t, cond_embeds, guidance_scale):
        batch_size = latents.shape[0]
        uncond_embeds = self.image_conditioner.null_embedding(batch_size)
        
        cond_embeds = torch.cat([uncond_embeds, cond_embeds], dim=0)
        latents = torch.cat([latents, latents], dim=0)
        
        noise_pred = self.unet(
            latents,
            t,
            encoder_hidden_states=cond_embeds,
        ).sample
        
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        return noise_pred
    
    @staticmethod
    def _tensor_to_pil(img_tensor):
        img = img_tensor.detach().cpu().clamp(0, 1)
        img = (img * 255).byte().permute(1, 2, 0).numpy()  # (H, W, 3)
        return Image.fromarray(img)
    
    @torch.no_grad()
    def predict_inbetween(self, start_frames, end_frames, num_inference_steps = 25, guidance_scale = 3.0):
        encoder_hidden_states = self.get_keyframe_condition(start_frames, end_frames)
                
        self.scheduler.set_timesteps(num_inference_steps, device = self.device)
        timesteps = self.scheduler.timesteps
        
        batch_size = start_frames.shape[0]
        latent_shape = (batch_size, self.vae.config.latent_channels, 64, 64)
        latents = torch.randn(latent_shape, device=self.device, dtype=self.vae.dtype)
        
        for t in timesteps:
            latent_in = self.scheduler.scale_model_input(latents, t)
            noise_pred = self.cfg_forward(
                latent_in,
                t,
                encoder_hidden_states,
                guidance_scale=guidance_scale
            )
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
        images = self.decode_latent_to_image(latents)
        return images
    
    def predict_inbetween_sequence(self, start_frames, end_frames, num_inbetweens=3, num_inference_steps=25, guidance_scale=3.0):
        inbetween_frames = []
        for _ in range(num_inbetweens):
            imgs = self.predict_inbetween(
                start_frames,
                end_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
            inbetween_frames.append(imgs)
            
        #OUTPUT FORMAT: LIST OF LISTS WITH FIRST DIM FRAME_INDEX, SECOND DIM BATCH_INDEX
        return inbetween_frames
    



def main(args):
    
    print("loading dataset and dataloader...")
    train_data = AnitaDataset("train", image_shape = (512,512))
    train_loader = DataLoader(train_data, batch_size=5, shuffle=False, num_workers=4)
    val_data = AnitaDataset("val", image_shape = (512,512))
    val_loader = DataLoader(val_data, batch_size=5, shuffle=False, num_workers=4)
    
    print("initializing model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = KeyframeConditionedDiffusion().to(device)
    
    print("starting training...")
    model.train_model(train_loader, val_loader, num_epochs=10, lr=1e-4, save_dir=args.save_dir)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion-Baseline-Inbetweening")
    parser.add_argument("--save_dir", default="model_ckpt", type=str)
    parser.add_argument("--num_denoising_steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    args = parser.parse_args()
    main(args)