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
import gc
import wandb


class KeyframeConditionedDiffusion(nn.Module):
    def __init__(
        self,
        model_name="runwayml/stable-diffusion-v1-5",
        val_visualization_dir="val_viz",
        device=None,
    ):
        super().__init__()
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        dtype = torch.float32  # if self.device == "cpu" else torch.float16
        self.val_vis_dir = val_visualization_dir
        pipe = StableDiffusionPipeline.from_pretrained(model_name, dtype=dtype).to(
            self.device
        )

        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

        self.vae = pipe.vae
        self.unet = pipe.unet
        self.image_processor = pipe.image_processor
        self.scheduler = pipe.scheduler

        # Freeze VAE - we don't need to train it
        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae.eval()

        # UNet and image_conditioner will be trainable
        self.image_conditioner = KeyframeConditionEncoder(
            vae_latent_channels=self.vae.config.latent_channels,
            conditioning_dim=self.unet.config.cross_attention_dim,
        ).to(self.device)

        self.safety_checker = None
        self.pipe = pipe

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
        # Use the encoder's null embedding method
        batch_size = cond.shape[0]
        return self.image_conditioner.null_embedding(batch_size)

    def decode_latent_to_image(self, latent):
        with torch.no_grad():
            latents = latent / self.vae.config.scaling_factor
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

    def training_step(self, batch, cfg_dropout_prob=0.1):
        start = batch["anchor_start"].to(self.device)
        end = batch["anchor_end"].to(self.device)
        targets = batch["targets"].to(self.device)

        idx = np.random.randint(0, targets.shape[1])
        target = targets[:, idx, :, :, :]

        with torch.no_grad():
            z0 = self.encode_image_to_latent(target)

        # sample timestep
        num_timesteps = self.scheduler.config.num_train_timesteps
        t = torch.randint(0, num_timesteps, (z0.shape[0],), device=self.device).long()

        noise = torch.randn_like(z0)
        z0_noisy = self.scheduler.add_noise(z0, noise, t)

        # Get keyframe condition
        keyframe_cond = self.get_keyframe_condition(start, end)

        # Classifier-free guidance: randomly drop condition during training
        # This trains the null embedding and enables CFG during inference
        batch_size = keyframe_cond.shape[0]
        drop_mask = torch.rand(batch_size, device=self.device) < cfg_dropout_prob

        # Replace dropped conditions with null embeddings
        null_embeds = self.null_embedding(
            keyframe_cond
        )  # Shape: (B, 2*num_tokens, dim)

        # Use null embeddings where drop_mask is True
        keyframe_cond = torch.where(
            drop_mask.view(-1, 1, 1).expand_as(keyframe_cond),
            null_embeds,
            keyframe_cond,
        )

        noise_pred = self.unet(
            z0_noisy,
            t,
            encoder_hidden_states=keyframe_cond,
        ).sample

        loss = F.mse_loss(noise_pred, noise)
        return loss

    def val_metrics(
        self,
        val_loader,
        epoch,
        guidance_scale=3.0,
        num_inference_steps=25,
    ):
        self.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, batch in tqdm(enumerate(val_loader), desc="Validation"):
                # Use cfg_dropout_prob=0.0 during validation to always use actual conditions
                loss = self.training_step(batch, cfg_dropout_prob=0.0)
                val_loss += loss.item()
                if i == 0:
                    pred_seq = self.predict_inbetween_sequence(
                        batch["anchor_start"].to(self.device),
                        batch["anchor_end"].to(self.device),
                        num_inbetweens=batch["targets"].shape[1],
                        num_inference_steps=num_inference_steps,
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
                self.val_vis_dir, f"epoch_{epoch+1:03d}_sample_{i}.png"
            )
            plt.savefig(out_path)

            plt.close(fig)

    def train_model(
        self,
        train_loader,
        val_loader,
        num_epochs=5,
        lr=1e-3,
        save_dir="model_ckpt",
        guidance_scale=3.0,
        num_denoising_steps=25,
        cfg_dropout_prob=0.1,
        save_interval=5,
        lr_patience=5,
        lr_factor=0.9,
        lr_min=1e-6,
        use_wandb=False,
    ):
        # Only train UNet and image_conditioner (VAE is frozen)
        trainable_params = list(self.unet.parameters()) + list(
            self.image_conditioner.parameters()
        )
        optimizer = torch.optim.AdamW(trainable_params, lr=lr)

        # Learning rate scheduler - reduces LR when validation loss plateaus
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=lr_factor,
            patience=lr_patience,
            min_lr=lr_min,
        )

        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float("inf")
        for epoch in range(num_epochs):
            self.train()
            # Ensure VAE stays in eval mode even when model is in train mode
            self.vae.eval()
            train_loss = 0.0
            for batch in tqdm(
                train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"
            ):
                # if i > 5:
                #     break
                optimizer.zero_grad()
                loss = self.training_step(batch, cfg_dropout_prob=cfg_dropout_prob)
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
                num_inference_steps=num_denoising_steps,
            )

            # Update learning rate based on validation loss
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Current Learning Rate: {current_lr:.2e}")

            # Log to wandb
            if use_wandb:
                log_dict = {
                    "train/loss": avg_train_loss,
                    "val/loss": val_loss,
                    "train/learning_rate": current_lr,
                    "train/epoch": epoch + 1,
                }
                wandb.log(log_dict)

            # Save best model
            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    "epoch": epoch + 1,
                    "model_state_dict": self.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "train_loss": avg_train_loss,
                    "learning_rate": current_lr,
                }
                torch.save(checkpoint, os.path.join(save_dir, f"best_model.pth"))
                print(
                    f"Saved best model (epoch {epoch+1}, val_loss: {best_val_loss:.4f})"
                )

            # Periodic checkpoint saving
            if (epoch + 1) % save_interval == 0:
                checkpoint = {
                    "epoch": epoch + 1,
                    "model_state_dict": self.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": val_loss,
                    "train_loss": avg_train_loss,
                    "learning_rate": current_lr,
                }
                torch.save(
                    checkpoint,
                    os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth"),
                )
                print(f"Saved checkpoint at epoch {epoch+1}")

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
        self, start_frames, end_frames, num_inference_steps=25, guidance_scale=3.0
    ):
        encoder_hidden_states = self.get_keyframe_condition(start_frames, end_frames)

        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        batch_size = start_frames.shape[0]
        latent_shape = (batch_size, self.vae.config.latent_channels, 64, 64)
        latents = torch.randn(latent_shape, device=self.device, dtype=self.vae.dtype)

        for t in timesteps:
            latent_in = self.scheduler.scale_model_input(latents, t)
            noise_pred = self.cfg_forward(
                latent_in, t, encoder_hidden_states, guidance_scale=guidance_scale
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
        guidance_scale=3.0,
    ):
        inbetween_frames = []
        for _ in range(num_inbetweens):
            imgs = self.predict_inbetween(
                start_frames,
                end_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
            inbetween_frames.append(imgs)

        # OUTPUT FORMAT: LIST OF LISTS WITH FIRST DIM FRAME_INDEX, SECOND DIM BATCH_INDEX
        return inbetween_frames


def main(args):

    # Initialize wandb if enabled
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "learning_rate": args.learning_rate,
                "num_epochs": args.num_epochs,
                "batch_size": 5,  # Hardcoded in DataLoader, could be made configurable
                "guidance_scale": args.guidance_scale,
                "num_denoising_steps": args.num_denoising_steps,
                "cfg_dropout_prob": args.cfg_dropout_prob,
                "lr_patience": args.lr_patience,
                "lr_factor": args.lr_factor,
                "lr_min": args.lr_min,
            },
        )

    print("loading dataset and dataloader...")
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")
    train_data = AnitaDataset(train_dir, image_shape=(224, 224))
    train_loader = DataLoader(train_data, batch_size=5, shuffle=False, num_workers=4)
    val_data = AnitaDataset(val_dir, image_shape=(224, 224))
    val_loader = DataLoader(val_data, batch_size=5, shuffle=False, num_workers=4)

    print("initializing model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = KeyframeConditionedDiffusion(val_visualization_dir=args.val_vis_dir).to(
        device
    )

    print("starting training...")
    model.train_model(
        train_loader,
        val_loader,
        num_epochs=args.num_epochs,
        lr=args.learning_rate,
        save_dir=args.save_dir,
        guidance_scale=args.guidance_scale,
        num_denoising_steps=args.num_denoising_steps,
        cfg_dropout_prob=args.cfg_dropout_prob,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        lr_min=args.lr_min,
        use_wandb=args.use_wandb,
    )

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion-Baseline-Inbetweening")
    parser.add_argument(
        "--data_dir",
        default="data/data_split",
        type=str,
        help="Root directory containing train/val/test subdirectories (default: data/data_split)",
    )
    parser.add_argument(
        "--save_dir",
        default="model_ckpt",
        type=str,
        help="Directory to save model checkpoints (default: model_ckpt)",
    )
    parser.add_argument(
        "--val_vis_dir",
        default="val_viz",
        type=str,
        help="Directory to save validation visualizations (default: val_viz)",
    )
    parser.add_argument("--num_denoising_steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--cfg_dropout_prob",
        type=float,
        default=0.1,
        help="Probability of dropping condition during training for CFG (default: 0.1)",
    )
    parser.add_argument(
        "--lr_patience",
        type=int,
        default=5,
        help="Number of epochs with no improvement before reducing LR (default: 5)",
    )
    parser.add_argument(
        "--lr_factor",
        type=float,
        default=0.5,
        help="Factor by which LR is reduced (default: 0.5)",
    )
    parser.add_argument(
        "--lr_min",
        type=float,
        default=1e-6,
        help="Minimum learning rate (default: 1e-6)",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb_project",
        default="keyframe-conditioned-diffusion",
        type=str,
        help="W&B project name (default: keyframe-conditioned-diffusion)",
    )
    parser.add_argument(
        "--wandb_run_name",
        default=None,
        type=str,
        help="W&B run name (default: auto-generated)",
    )
    args = parser.parse_args()
    main(args)
