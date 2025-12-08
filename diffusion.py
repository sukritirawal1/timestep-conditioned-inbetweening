import os
import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler

import torch
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path


class Diffusion:
    def __init__(self, model_name="runwayml/stable-diffusion-v1-5", device=None):
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        dtype = torch.float32 if self.device == "cpu" else torch.float16

        pipe = StableDiffusionPipeline.from_pretrained(
            model_name, torch_dtype=dtype
        ).to(self.device)

        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

        self.vae = pipe.vae
        self.text_encoder = pipe.text_encoder
        self.tokenizer = pipe.tokenizer
        self.unet = pipe.unet
        self.image_processor = pipe.image_processor
        self.scheduler = pipe.scheduler

        self.prompt = "a clean cartoon animation frame"
        self.text_embeds = self._get_text_embeds(self.prompt)

        pipe.safety_checker = None
        self.pipe = pipe

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

    def decode_latent_to_image(self, latent):
        with torch.no_grad():
            latents = latent / self.vae.config.scaling_factor
            img = self.vae.decode(latents).sample
        img = (img / 2 + 0.5).clamp(0, 1)
        img = img.cpu().permute(0, 2, 3, 1).numpy()
        img = (img * 255).round().astype("uint8")

        images = [Image.fromarray(image) for image in img]
        return images if len(images) > 1 else images[0]

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

    def denoise_latent(
        self, z_interp, num_inference_steps=25, noise_strength=0.3, guidance_scale=1.0
    ):
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        noise = torch.randn_like(z_interp)

        start_idx = int(noise_strength * (len(timesteps) - 1))
        t_start = timesteps[start_idx]

        noise = torch.randn_like(z_interp)
        latents = self.scheduler.add_noise(z_interp, noise, t_start)

        for t in timesteps[start_idx:]:
            latent_input = self.scheduler.scale_model_input(latents, t)

            noise_pred = self.unet(
                latent_input,
                t,
                encoder_hidden_states=self.text_embeds,
            ).sample

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return latents

    @torch.no_grad()
    def generate_inbetweens(
        self,
        start_frames,
        end_frames,
        num_inbetweens=3,
        num_inference_steps=25,
        noise_strength=0.3,
    ):
        if not isinstance(start_frames, list):
            start_frames = [start_frames]
        if not isinstance(end_frames, list):
            end_frames = [end_frames]
        z0 = self.encode_image_to_latent(start_frames)
        z1 = self.encode_image_to_latent(end_frames)

        outputs = []
        for i in range(1, num_inbetweens + 1):
            alpha = i / (num_inbetweens + 1)
            z_interp = self.slerp(z0, z1, alpha)
            z_denoised = self.denoise_latent(
                z_interp, num_inference_steps, noise_strength
            )
            decoded = self.decode_latent_to_image(z_denoised)
            outputs.append(decoded)

        return outputs


def process_images(image_paths, size=(512, 512)):
    if isinstance(image_paths, (str, Path)):
        image_paths = [image_paths]

    images = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        img = img.resize(size, Image.LANCZOS)
        images.append(img)

    return images


def visualize_results(start_frames, end_frames, inbetweens, save_path=None):
    """
    Visualize the interpolation results.

    Args:
        start_frames: List of start frame PIL Images
        end_frames: List of end frame PIL Images
        inbetweens: List of lists containing inbetween frames
        save_path: Optional path to save the visualization
    """
    num_sequences = len(start_frames)
    num_inbetweens = len(inbetweens[0])
    total_frames = num_inbetweens + 2  # start + inbetweens + end

    fig, axes = plt.subplots(
        num_sequences, total_frames, figsize=(3 * total_frames, 3 * num_sequences)
    )

    if num_sequences == 1:
        axes = axes.reshape(1, -1)

    for seq_idx in range(num_sequences):
        # Plot start frame
        axes[seq_idx, 0].imshow(start_frames[seq_idx])
        axes[seq_idx, 0].set_title(f"Start {seq_idx+1}")
        axes[seq_idx, 0].axis("off")

        # Plot inbetween frames
        for i, frame in enumerate(inbetweens[seq_idx]):
            axes[seq_idx, i + 1].imshow(frame)
            axes[seq_idx, i + 1].set_title(f"Frame {i+1}")
            axes[seq_idx, i + 1].axis("off")

        # Plot end frame
        axes[seq_idx, -1].imshow(end_frames[seq_idx])
        axes[seq_idx, -1].set_title(f"End {seq_idx+1}")
        axes[seq_idx, -1].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()


# def main():
#     # Initialize the diffusion model
#     print("Loading diffusion model...")
#     diffusion = Diffusion()

#     # Example 1: Single pair of images
#     # print("\n=== Testing single image pair ===")
#     # start_images = process_images("start_frame.png")
#     # end_images = process_images("end_frame.png")

#     # inbetweens = diffusion.generate_inbetweens(
#     #     start_images,
#     #     end_images,
#     #     num_inbetweens=3,
#     #     num_inference_steps=5,
#     #     noise_strength=0.3
#     # )

#     # visualize_results(start_images, end_images, [inbetweens],
#     #                 save_path="single_interpolation.png")

#     # Example 2: Batch of image pairs
#     print("\n=== Testing batch of image pairs ===")
#     start_paths = ["start_frame.png", "start_frame2.png"]
#     end_paths = ["end_frame.png", "end_frame2.png"]

#     start_images_batch = process_images(start_paths)
#     end_images_batch = process_images(end_paths)

#     # Process each pair (you can modify generate_inbetweens to handle true batching)
#     all_inbetweens = []
#     for start, end in zip(start_images_batch, end_images_batch):
#         inbetweens = diffusion.generate_inbetweens(
#             [start],
#             [end],
#             num_inbetweens=5,
#             num_inference_steps=25,
#             noise_strength=0.3
#         )
#         all_inbetweens.append(inbetweens)

#     visualize_results(start_images_batch, end_images_batch, all_inbetweens,
#                     save_path="batch_interpolation.png")

#     # Example 3: Save individual frames
#     print("\n=== Saving individual frames ===")
#     output_dir = Path("output_frames")
#     output_dir.mkdir(exist_ok=True)

#     for i, frame in enumerate(inbetweens):
#         frame.save(output_dir / f"frame_{i+1:03d}.png")

#     print(f"Saved {len(inbetweens)} frames to {output_dir}")


# if __name__ == "__main__":
#     main()
