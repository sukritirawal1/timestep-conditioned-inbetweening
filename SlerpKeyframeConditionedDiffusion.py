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
import multiprocessing as mp
import gc

from transformers import CLIPVisionModelWithProjection, CLIPProcessor


class SlerpKeyframeConditionedDiffusion(nn.Module):
    def __init__(
        self,
        model_name="runwayml/stable-diffusion-v1-5",
        save_dir="output",
        device=None,
    ):
        super().__init__()
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        dtype = torch.float32  # if self.device == "cpu" else torch.float16
        self.val_vis_dir = save_dir
        pipe = StableDiffusionPipeline.from_pretrained(
            model_name, torch_dtype=dtype
        ).to(self.device)

        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

        self.vae = pipe.vae
        self.unet = pipe.unet
        self.image_processor = pipe.image_processor
        self.scheduler = pipe.scheduler

        # latent_channels = self.vae.config.latent_channels

        self.text_encoder = pipe.text_encoder
        self.tokenizer = pipe.tokenizer
        self.prompt = "a clean cartoon animation frame"
        self.text_embeds = self._get_text_embeds(self.prompt)

        context_dim = self.unet.config.cross_attention_dim
        self.visual_adapter = nn.Sequential(
            nn.Linear(768, context_dim),
            nn.LayerNorm(context_dim),
            nn.ReLU(),
            nn.Linear(context_dim, context_dim),
        )

        self.clip_model = CLIPVisionModelWithProjection.from_pretrained(
            "openai/clip-vit-large-patch14"
        ).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self.clip_projection = self.clip_model.visual_projection

        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.clip_model.eval()

        self.safety_checker = None
        self.pipe = pipe

        for param in self.unet.parameters():
            param.requires_grad = False
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def _get_text_embeds(self, prompt):
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        return text_embeddings

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

    def null_embedding(self, cond):
        return torch.zeros_like(cond)

    def encode_condition_CLIP(self, images):
        inputs = self.clip_processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            return outputs.image_embeds

    def decode_latent_to_image(self, latent):
        with torch.no_grad():
            latents = latent / self.vae.config.scaling_factor
            img = self.vae.decode(latents).sample
        img = (img / 2 + 0.5).clamp(0, 1)
        img = img.cpu().permute(0, 2, 3, 1).numpy()
        img = (img * 255).round().astype("uint8")

        return [Image.fromarray(image) for image in img]

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

    def get_keyframe_condition(self, start_frames, end_frames, timestep=0.5):
        # with torch.no_grad():
        #     z_start = self.encode_image_to_latent(start_frames)
        #     z_end = self.encode_image_to_latent(end_frames)
        # cond = self.image_conditioner(z_start, z_end)
        start_feat = self.encode_condition_CLIP(start_frames)
        end_feat = self.encode_condition_CLIP(end_frames)
        combined_feat = (1 - timestep) * start_feat + timestep * end_feat
        keyframe_cond = self.visual_adapter(combined_feat).unsqueeze(1)
        return keyframe_cond

    def training_step(self, batch, structure_loss=False):
        start = batch["anchor_start"].to(self.device)
        end = batch["anchor_end"].to(self.device)
        targets = batch["targets"].to(self.device)

        idx = np.random.randint(0, targets.shape[1])
        timestep = (idx + 1) / (targets.shape[1] + 1)
        target = targets[:, idx, :, :, :]

        with torch.no_grad():
            z_target = self.encode_image_to_latent(target)

            z_start = self.encode_image_to_latent(start)
            z_end = self.encode_image_to_latent(end)
            z_middle = self.slerp(z_start, z_end, alpha=timestep)

        # sample timestep
        num_timesteps = self.scheduler.config.num_train_timesteps
        t = torch.randint(
            0, num_timesteps, (z_target.shape[0],), device=self.device
        ).long()

        noise = torch.randn_like(z_target)
        zt_noisy = self.scheduler.add_noise(z_target, noise, t)

        batch_text_embeds = self.text_embeds.repeat(z_target.shape[0], 1, 1)

        keyframe_cond = self.get_keyframe_condition(start, end, timestep=timestep)
        # print(keyframe_cond.shape)
        combined_cond = torch.cat([batch_text_embeds, keyframe_cond], dim=1)

        noise_pred = self.unet(
            zt_noisy,
            t,
            encoder_hidden_states=combined_cond,
        ).sample

        loss = F.mse_loss(noise_pred, noise)

        if structure_loss:
            with torch.no_grad():
                z_pred = self.scheduler.step(noise_pred, t, zt_noisy).prev_sample
            structure_loss = F.mse_loss(z_pred, z_middle)

            loss += +0.1 * structure_loss

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
                        num_inference_steps=num_inference_steps,
                        noise_strength=noise_strength,
                        guidance_scale=guidance_scale,
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
        img = img_tensor.detach().cpu().clamp(0, 1)
        img = (img * 255).byte().permute(1, 2, 0).numpy()  # (H, W, 3)
        return Image.fromarray(img)

    @torch.no_grad()
    def predict_inbetween(
        self,
        start_frames,
        end_frames,
        num_inference_steps=25,
        noise_strength=0.3,
        timestep=0.5,
        guidance_scale=1.0,
    ):

        z0 = self.encode_image_to_latent(start_frames)
        z1 = self.encode_image_to_latent(end_frames)
        z_init = self.slerp(z0, z1, timestep)

        text_embeds = self.text_embeds.repeat(z_init.shape[0], 1, 1)
        visual_embeds = self.get_keyframe_condition(start_frames, end_frames, timestep)
        combined_embeds = torch.cat([text_embeds, visual_embeds], dim=1)

        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        noise_strength = 0.3
        start_idx = int(noise_strength * (len(timesteps) - 1))
        t_start = timesteps[start_idx]

        noise = torch.randn_like(z_init)
        latents = self.scheduler.add_noise(z_init, noise, t_start)

        for t in timesteps[start_idx:]:
            latent_input = self.scheduler.scale_model_input(latents, t)

            noise_pred = self.cfg_forward(
                latent_input,
                t,
                cond_embeds=combined_embeds,
                guidance_scale=guidance_scale,
            )

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        images = self.decode_latent_to_image(latents)
        return images

    def predict_inbetween_sequence(
        self,
        start_frames,
        end_frames,
        num_inbetweens=3,
        num_inference_steps=25,
        noise_strength=0.3,
        guidance_scale=1.0,
    ):
        inbetween_frames = []
        timesteps = [(i + 1) / (num_inbetweens + 1) for i in range(num_inbetweens)]
        for x in timesteps:
            imgs = self.predict_inbetween(
                start_frames,
                end_frames,
                num_inference_steps=num_inference_steps,
                noise_strength=noise_strength,
                timestep=x,
                guidance_scale=guidance_scale,
            )
            inbetween_frames.append(imgs)

        # OUTPUT FORMAT: LIST OF LISTS WITH FIRST DIM FRAME_INDEX, SECOND DIM BATCH_INDEX
        return inbetween_frames


def main(args):

    print("fixed2")

    print("loading dataset and dataloader...")
    train_data = AnitaDataset(
        os.path.join(args.data_dir, "train"), image_shape=(224, 224)
    )
    train_loader = DataLoader(
        train_data, batch_size=8, shuffle=True, num_workers=mp.cpu_count()
    )
    val_data = AnitaDataset(os.path.join(args.data_dir, "val"), image_shape=(224, 224))
    val_loader = DataLoader(
        val_data, batch_size=8, shuffle=True, num_workers=mp.cpu_count()
    )

    print("initializing model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SlerpKeyframeConditionedDiffusion(save_dir=args.save_dir).to(device)

    print("starting training...")
    # model.train_model(
    #     train_loader,
    #     val_loader,
    #     num_epochs=args.num_epochs_adapter_only,
    #     lr=1e-4,
    #     save_dir=args.save_dir,
    #     noise_strength=args.noise_strength,
    #     num_denoising_steps=args.num_denoising_steps,
    #     structure_loss=args.use_structure_loss,
    #     guidance_scale=args.guidance_scale,
    # )

    for name, param in model.unet.named_parameters():
        if "attn2" in name:  # CA layers
            param.requires_grad = True
        else:
            param.requires_grad = False

    model.train_model(
        train_loader,
        val_loader,
        num_epochs=args.num_epochs_unfreeze_CA,
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
    parser.add_argument("--num_epochs_adapter_only", type=int, default=3)
    parser.add_argument("--num_epochs_unfreeze_CA", type=int, default=2)
    parser.add_argument("--use_structure_loss", action="store_true")
    parser.add_argument("--data_dir", type=str, default="data")
    args = parser.parse_args()
    main(args)
