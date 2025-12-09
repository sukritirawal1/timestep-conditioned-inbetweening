import os
import numpy as np
from PIL import Image
import torch
from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
    StableDiffusionImageVariationPipeline,
)
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
import multiprocessing as mp
import gc


class ImageConditionedDiffusion(nn.Module):
    def __init__(
        self,
        model_name="lambdalabs/sd-image-variations-diffusers",
        save_dir="output",
        device=None,
    ):
        super().__init__()
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        dtype = torch.float32  # if self.device == "cpu" else torch.float16
        self.val_vis_dir = save_dir
        pipe = StableDiffusionImageVariationPipeline.from_pretrained(
            model_name, torch_dtype=dtype, revision="v2.0"
        ).to(self.device)

        self.pipe = pipe
        pipe.safety_checker = None
        self.unet = pipe.unet
        self.scheduler = pipe.scheduler
        self.vae = pipe.vae
        self.image_encoder = pipe.image_encoder
        self.feature_extractor = pipe.feature_extractor
        self.feature_extractor.do_rescale = False
        self.image_processor = pipe.image_processor

        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.unet.parameters():
            param.requires_grad = False
        for param in self.vae.parameters():
            param.requires_grad = False

        context_dim = self.unet.config.cross_attention_dim
        self.visual_adapter = nn.Sequential(
            nn.Linear(768 * 2, context_dim),
            nn.SiLU(),
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, context_dim),
        )

    def slerp(self, z0, z1, alpha, dot_threshold=0.9995):
        # flatten
        v0 = z0.reshape(z0.shape[0], -1)
        v1 = z1.reshape(z1.shape[0], -1)

        dot = torch.sum(v0 * v1, dim=1) / (v0.norm(dim=1) * v1.norm(dim=1))
        dot = torch.clamp(dot, -1.0, 1.0)

        if not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha, device=z0.device, dtype=z0.dtype)
        alpha = alpha.view(1, 1)  # broadcast

        linear_mask = torch.abs(dot) > dot_threshold
        non_linear_mask = ~linear_mask

        v2 = torch.empty_like(v0)

        if linear_mask.any():
            v0_lin = v0[linear_mask]
            v1_lin = v1[linear_mask]
            v2[linear_mask] = (1 - alpha) * v0_lin + alpha * v1_lin

        if non_linear_mask.any():
            v0_nl = v0[non_linear_mask]
            v1_nl = v1[non_linear_mask]
            dot_nl = dot[non_linear_mask]

            theta_0 = torch.arccos(dot_nl)
            sin_theta_0 = torch.sin(theta_0)
            theta_t = theta_0 * alpha.squeeze()
            sin_theta_t = torch.sin(theta_t)

            s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
            s1 = sin_theta_t / sin_theta_0
            v2[non_linear_mask] = s0.unsqueeze(1) * v0_nl + s1.unsqueeze(1) * v1_nl

        return v2.reshape_as(z0)

    def encode_image_to_latent(self, images):
        # if images.isinstance(list):
        #     pass
        if images.ndim == 3:
            images = [images]
        elif images.ndim == 4:
            images = [img for img in images]

        img_list = [self.image_processor.preprocess(x) for x in images]
        img_batch = torch.cat(img_list, dim=0)
        img_batch = img_batch.to(device=self.device, dtype=self.vae.dtype)
        with torch.no_grad():
            dist = self.vae.encode(img_batch).latent_dist
            latents = dist.sample() * self.vae.config.scaling_factor
        return latents

    def encode_condition_CLIP(self, images):
        # imgs = images.detach().cpu().permute(0, 2, 3, 1).numpy()
        feats = self.feature_extractor(images=images, return_tensors="pt").pixel_values
        feats = feats.to(self.device, dtype=self.image_encoder.dtype)
        # (B, seqlen+1, 768)
        return self.image_encoder(pixel_values=feats).last_hidden_state
        # return self.image_encoder(images).image_embeds

    def generate_condition_image(self, start_frames, end_frames, timestep=0.5):
        images = self.slerp(start_frames, end_frames, timestep)
        return images

    def generate_condition_embeds(self, start_frames, end_frames, timestep=0.5):
        start_emb = self.encode_condition_CLIP(start_frames)
        end_emb = self.encode_condition_CLIP(end_frames)
        # slerped_images = self.slerp(start_frames, end_frames, timestep)
        both = torch.cat([start_emb, end_emb], dim=1)  # (B, 1536)
        # cond = self.visual_adapter(both) # (B, context_dim(768))
        # cond = (1 - timestep) * start_emb + timestep * end_emb
        return both

    def training_step(self, batch, structure_loss=False):
        start = batch["anchor_start"].to(self.device)
        end = batch["anchor_end"].to(self.device)
        targets = batch["targets"].to(self.device)

        idx = np.random.randint(0, targets.shape[1])
        timestep = (idx + 1) / (targets.shape[1] + 1)
        target = targets[:, idx, :, :, :]

        with torch.no_grad():
            z_target = self.encode_image_to_latent(target)
            condition = self.generate_condition_embeds(start, end, timestep=timestep)

        # sample timestep
        num_timesteps = self.scheduler.config.num_train_timesteps
        t = torch.randint(
            0, num_timesteps, (z_target.shape[0],), device=self.device
        ).long()

        noise = torch.randn_like(z_target)
        zt_noisy = self.scheduler.add_noise(z_target, noise, t)

        noise_pred = self.unet(
            zt_noisy,
            t,
            encoder_hidden_states=condition,
        ).sample

        loss = F.mse_loss(noise_pred, noise)

        return loss

    def val_metrics(
        self,
        val_loader,
        epoch,
        guidance_scale=1.0,
        noise_strength=0.3,
        num_inference_steps=50,
        structure_loss=False,
    ):
        self.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, batch in tqdm(enumerate(val_loader), desc="Validation"):
                loss = self.training_step(batch, structure_loss=structure_loss)
                val_loss += loss.item()
                if i == 0:
                    pred_seq = self.predict_inbetween_sequence(
                        batch["anchor_start"].to(self.device),
                        batch["anchor_end"].to(self.device),
                        num_inbetweens=batch["targets"].shape[1],
                        num_denoising_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                    )
                    self.visualize_inbetweens(batch, pred_seq, epoch)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        return avg_val_loss

    def visualize_inbetweens(self, batch, pred_seq, epoch):
        os.makedirs(os.path.join(self.val_vis_dir, "val_visualizations"), exist_ok=True)
        self.eval()
        start = batch["anchor_start"].to(self.device)
        end = batch["anchor_end"].to(self.device)
        targets = batch["targets"].to(self.device)
        B, N, _, _, _ = targets.shape

        for i in range(B):
            start_i = start[i]  # (3, H, W)
            end_i = end[i]  # (3, H, W)
            targets_i = targets[i]  # (T, 3, H, W)

            # --- Build GT row: [start, gt_0, gt_1, ..., gt_{T-1}, end] ---
            gt_frames = [self._tensor_to_pil(start_i)]
            for t_idx in range(N):
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

            out_path = os.path.join(
                self.val_vis_dir,
                "val_visualizations",
                f"epoch_{epoch+1:03d}_sample_{i}.png",
            )
            plt.savefig(out_path)
            plt.close(fig)

    def train_model(
        self,
        train_loader,
        val_loader,
        num_epochs=5,
        lr=1e-4,
        save_dir="output",
        noise_strength=0.3,
        num_denoising_steps=25,
        structure_loss=False,
        guidance_scale=1.0,
    ):
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "model_ckpt"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "val_visualizations"), exist_ok=True)
        best_val_loss = float("inf")
        for epoch in range(num_epochs):
            self.train()
            train_loss = 0.0
            i = 0
            for batch in tqdm(
                train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"
            ):
                # if i > 3:
                #     break
                optimizer.zero_grad()
                loss = self.training_step(batch, structure_loss=structure_loss)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                i += 1
            torch.cuda.empty_cache()
            gc.collect()
            avg_train_loss = train_loss / len(train_loader)
            print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")
            val_loss = self.val_metrics(
                val_loader,
                epoch,
                guidance_scale=guidance_scale,
                noise_strength=noise_strength,
                num_inference_steps=num_denoising_steps,
                structure_loss=structure_loss,
            )
            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    self.state_dict(),
                    os.path.join(save_dir, "model_ckpt", f"best_model.pth"),
                )

    def cfg_forward(self, latents, t, cond_embeds, guidance_scale):
        # batch_size = latents.shape[0]
        uncond_embeds = self.null_embedding(cond_embeds)

        cond_embeds = torch.cat([uncond_embeds, cond_embeds], dim=0)
        latents = torch.cat([latents, latents], dim=0)

        noise_pred = self.unet(
            latents,
            t,
            encoder_hidden_states=cond_embeds,
        ).sample

        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_cond - noise_pred_uncond
        )
        return noise_pred

    @staticmethod
    def _tensor_to_pil(img_tensor):
        img = img_tensor.detach().cpu().clamp(0, 1)
        img = (img * 255).byte().permute(1, 2, 0).numpy()  # (H, W, 3)
        return Image.fromarray(img)

    def decode_latent_to_image(self, latent):
        with torch.no_grad():
            latents = latent / self.vae.config.scaling_factor
            img = self.vae.decode(latents).sample
        img = (img / 2 + 0.5).clamp(0, 1)
        img = img.cpu().permute(0, 2, 3, 1).numpy()
        img = (img * 255).round().astype("uint8")
        return [Image.fromarray(image) for image in img]

    @torch.no_grad()
    def predict_inbetween(
        self,
        start_frames,
        end_frames,
        timestep=0.5,
        num_inference_steps=25,
        guidance_scale=1.0,
    ):

        cond_embeds = self.generate_condition_embeds(
            start_frames, end_frames, timestep=timestep
        )

        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        latents = torch.randn(
            (start_frames.shape[0], self.unet.in_channels, 64, 64),  # 64x64 for SD v1
            device=self.device,
            dtype=self.vae.dtype,
        )

        for t in self.scheduler.timesteps:
            noise_pred = self.unet(
                latents,
                t,
                encoder_hidden_states=cond_embeds,
            ).sample

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        images = self.decode_latent_to_image(latents)

        return images

    def predict_inbetween_sequence(
        self,
        start_frames,
        end_frames,
        num_inbetweens=3,
        num_denoising_steps=25,
        guidance_scale=1.0,
    ):
        inbetween_frames = []
        timesteps = [(i + 1) / (num_inbetweens + 1) for i in range(num_inbetweens)]
        for x in timesteps:
            imgs = self.predict_inbetween(
                start_frames,
                end_frames,
                timestep=x,
                num_inference_steps=num_denoising_steps,
                guidance_scale=guidance_scale,
            )
            inbetween_frames.append(imgs)
        # OUTPUT FORMAT: LIST OF LISTS WITH FIRST DIM FRAME_INDEX, SECOND DIM BATCH_INDEX
        return inbetween_frames


def main(args):

    print("Image Conditioned Diffusion6")

    print("loading dataset and dataloader...")
    train_data = AnitaDataset(
        os.path.join(args.data_dir, "train"), image_shape=(224, 224)
    )
    print(f"Loaded {len(train_data)} train images from {args.data_dir}/train")

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=mp.cpu_count()
    )
    print(f"Loaded {len(train_data)} train images from {args.data_dir}/train")
    val_data = AnitaDataset(os.path.join(args.data_dir, "val"), image_shape=(224, 224))

    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=True, num_workers=mp.cpu_count()
    )
    print(f"Loaded {len(val_data)} val images from {args.data_dir}/val")

    print("initializing model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ImageConditionedDiffusion(save_dir=args.save_dir).to(device)

    print("starting training...")

    for name, param in model.unet.named_parameters():
        if "attn2" in name:  # CA layers
            param.requires_grad = True
        else:
            param.requires_grad = False

    model.train_model(
        train_loader,
        val_loader,
        num_epochs=args.num_epochs,
        lr=1e-4,
        save_dir=args.save_dir,
        noise_strength=args.noise_strength,
        num_denoising_steps=args.num_denoising_steps,
        structure_loss=args.use_structure_loss,
        guidance_scale=args.guidance_scale,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion-Baseline-Inbetweening")
    parser.add_argument("--save_dir", default="output", type=str)
    parser.add_argument("--num_denoising_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--noise_strength", type=float, default=0.3)
    parser.add_argument("--num_epochs", type=int, default=25)
    parser.add_argument("--use_structure_loss", action="store_true")
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    main(args)
