import os
import numpy as np
from PIL import Image
import torch
from diffusers import (
    StableDiffusionImageVariationPipeline,
)
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

from src.dataset import AnitaDataset
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
        self.image_processor = pipe.image_processor

        register_weighted_attn_hooks(self)

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
        return self.image_encoder(images).image_embeds

    def generate_condition_image(self, start_frames, end_frames, timestep=0.5):
        images = self.slerp(start_frames, end_frames, timestep)
        return images

    def generate_condition_embeds(self, start_frames, end_frames, timestep=0.5):
        # start_emb = self.encode_condition_CLIP(start_frames)
        # end_emb = self.encode_condition_CLIP(end_frames)
        slerped_images = self.slerp(start_frames, end_frames, timestep)
        embeds = self.encode_condition_CLIP(slerped_images).unsqueeze(1)
        return embeds

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
        start_embed = self.encode_condition_CLIP(start)
        end_embed = self.encode_condition_CLIP(end)
        t_scalar = torch.tensor(timestep, device=self.device)

        self._hook_cache = {
            "start_embed": start_embed.unsqueeze(1),
            "end_embed": end_embed.unsqueeze(1),
            "t_scalar": t_scalar,
        }

        noise_pred = self.unet(
            zt_noisy,
            t,
            encoder_hidden_states=condition,
        ).sample

        loss = F.mse_loss(noise_pred, noise)
        self._hook_cache = None
        return loss

    def val_metrics(
        self,
        val_loader,
        epoch,
        guidance_scale=1.0,
        noise_strength=0.3,
        num_inference_steps=25,
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
            for batch in tqdm(
                train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"
            ):
                # if i > 5:
                #     break
                optimizer.zero_grad()
                loss = self.training_step(batch, structure_loss=structure_loss)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
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
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
        img = img_tensor.detach().cpu()
        if img.shape[0] != 3 and img.shape[-1] == 3:
            img = img.permute(2, 0, 1)  # if anything is HWC
        img = (img * std + mean).clamp(0, 1)
        img = (img * 255).byte().permute(1, 2, 0).numpy()
        return Image.fromarray(img)

    @torch.no_grad()
    def predict_inbetween(
        self,
        start_frames,
        end_frames,
        timestep=0.5,
        guidance_scale=1.0,
    ):
        start_embed = self.encode_condition_CLIP(start_frames)
        end_embed = self.encode_condition_CLIP(end_frames)
        t_scalar = torch.tensor(timestep, device=self.device)

        self._hook_cache = {
            "start_embed": start_embed.unsqueeze(1),
            "end_embed": end_embed.unsqueeze(1),
            "t_scalar": t_scalar,
        }

        condition = self.generate_condition_image(
            start_frames, end_frames, timestep=timestep
        )
        images = self.pipe(condition, guidance_scale=guidance_scale)
        self._hook_cache = None
        return images["images"]

    def predict_inbetween_sequence(
        self,
        start_frames,
        end_frames,
        num_inbetweens=3,
        guidance_scale=1.0,
    ):
        inbetween_frames = []
        timesteps = [(i + 1) / (num_inbetweens + 1) for i in range(num_inbetweens)]
        for x in timesteps:
            imgs = self.predict_inbetween(
                start_frames,
                end_frames,
                timestep=x,
                guidance_scale=guidance_scale,
            )
            inbetween_frames.append(imgs)
        # OUTPUT FORMAT: LIST OF LISTS WITH FIRST DIM FRAME_INDEX, SECOND DIM BATCH_INDEX
        return inbetween_frames


def register_weighted_attn_hooks(parent_model):
    def make_hook(attn_module):
        def hook(module, input, output):
            h = input[0]
            t = parent_model._hook_cache["t_scalar"]
            s_emb = parent_model._hook_cache["start_embed"]
            e_emb = parent_model._hook_cache["end_embed"]

            # Call forward directly to avoid hook recursion
            out_start = attn_module.forward(h, encoder_hidden_states=s_emb)
            torch.cuda.empty_cache()
            out_end = attn_module.forward(h, encoder_hidden_states=e_emb)
            torch.cuda.empty_cache()
            return (1 - t) * out_start + t * out_end

        return hook

    for name, module in parent_model.unet.named_modules():

        # print(name)
        if name.endswith(".attn2"):
            module.register_forward_hook(make_hook(module))


def main(args):

    print("Image Conditioned Diffusion")

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

    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print("Checkpoint loaded successfully!")

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
    parser.add_argument("--num_denoising_steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--noise_strength", type=float, default=0.3)
    parser.add_argument("--num_epochs", type=int, default=25)
    parser.add_argument("--use_structure_loss", action="store_true")
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()
    main(args)
