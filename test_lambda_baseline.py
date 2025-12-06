"""
Baseline 3: Lambda Labs SD Image Variations
Test script for image-conditioned Stable Diffusion model
Uses HuggingFace Diffusers (cached models)
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import os
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from diffusers import StableDiffusionImageVariationPipeline
from dataloader import AnitaDataset
from eval import FrameInterpolationEvaluator
import matplotlib.pyplot as plt

class LambdaSDImageVariations:
    """
    Lambda Labs' image-conditioned Stable Diffusion model
    Uses CLIP image embeddings to condition generation
    """
    def __init__(self, device='cuda', dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        
        print("Loading Lambda Labs SD Image Variations model...")
        
        # Load the image variation pipeline
        self.pipe = StableDiffusionImageVariationPipeline.from_pretrained(
            "lambdalabs/sd-image-variations-diffusers",
            revision="v2.0",
            torch_dtype=dtype,
        )
        self.pipe = self.pipe.to(device)
        
        # Load CLIP image encoder for keyframe encoding
        print("Loading CLIP image encoder...")
        self.clip_processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self.clip_model = CLIPVisionModelWithProjection.from_pretrained(
            "openai/clip-vit-large-patch14",
            torch_dtype=dtype
        ).to(device)
        
        print("Model loaded successfully!")
    
    def encode_image_to_clip_embedding(self, image):
        """
        Encode a PIL Image to CLIP embedding
        Args:
            image: PIL Image or torch tensor [C, H, W] in range [0, 1]
        Returns:
            CLIP embedding tensor [1, 1, 768]  # Note the extra dimension!
        """
        # Convert torch tensor to PIL if needed
        if isinstance(image, torch.Tensor):
            # Handle both [C, H, W] and [B, C, H, W]
            if image.ndim == 4:
                image = image[0]  # Take first batch
            if image.ndim == 3:
                # Convert [C, H, W] to PIL
                img_np = image.cpu().permute(1, 2, 0).numpy()
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)
                else:
                    img_np = img_np.astype(np.uint8)
                image = Image.fromarray(img_np)
        
        # Ensure it's RGB
        if isinstance(image, Image.Image):
            image = image.convert('RGB')
        
        # Process with CLIP processor
        inputs = self.clip_processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(device=self.device, dtype=self.dtype)
        
        # Get CLIP image embedding
        with torch.no_grad():
            image_embeds = self.clip_model(pixel_values).image_embeds
        
        # Add sequence dimension: [1, 768] -> [1, 1, 768]
        image_embeds = image_embeds.unsqueeze(1)
        
        return image_embeds
    
    def interpolate_embeddings(self, embed_start, embed_end, t):
        """
        Linear interpolation between two embeddings
        Args:
            embed_start: Start frame embedding [1, 768]
            embed_end: End frame embedding [1, 768]
            t: Interpolation factor in [0, 1]
        Returns:
            Interpolated embedding [1, 768]
        """
        return (1 - t) * embed_start + t * embed_end
    
    @torch.no_grad()
    def generate_from_embedding(self, embedding, num_inference_steps=50, guidance_scale=3.0):
        """
        Generate image from CLIP embedding using the Lambda model
        
        FIXED: This method now properly handles embedding generation
        """
        # Normalize the embedding (CLIP embeddings should be normalized)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        
        # Duplicate for classifier-free guidance (unconditional + conditional)
        if guidance_scale > 1.0:
            # Unconditional embedding is zeros
            uncond_embedding = torch.zeros_like(embedding)
            embedding = torch.cat([uncond_embedding, embedding])
        
        # Set random seed for reproducibility (optional)
        generator = torch.Generator(device=self.device)
        
        # Initialize latents
        latents = torch.randn(
            (1, self.pipe.unet.config.in_channels, 64, 64),  # Standard SD latent size
            generator=generator,
            device=self.device,
            dtype=self.dtype
        )
        
        # Set timesteps
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.pipe.scheduler.timesteps
        
        # Denoise loop
        for t in tqdm(timesteps, desc="Denoising", leave=False):
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)
            
            # Predict noise residual
            noise_pred = self.pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=embedding,
            ).sample
            
            # Classifier-free guidance
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            # Compute previous latent
            latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample
        
        # Decode latents to image
        latents = latents / self.pipe.vae.config.scaling_factor
        image = self.pipe.vae.decode(latents).sample
        
        # Post-process
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).round().astype("uint8")
        
        return Image.fromarray(image)
    
    def generate_inbetween_frames(self, start_frame, end_frame, num_inference_steps=50, guidance_scale=3.0):
        """
        Generate 3 inbetween frames at t=0.25, 0.5, 0.75
        
        Args:
            start_frame: PIL Image or torch tensor [C, H, W]
            end_frame: PIL Image or torch tensor [C, H, W]
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for generation
        
        Returns:
            List of 3 PIL Images
        """
        # Encode keyframes to CLIP embeddings
        embed_start = self.encode_image_to_clip_embedding(start_frame)
        embed_end = self.encode_image_to_clip_embedding(end_frame)
        
        # Generate frames at different timesteps
        generated_frames = []
        for t in [0.25, 0.5, 0.75]:
            print(f"  Generating frame at t={t}...")
            # Interpolate embeddings
            embed_t = self.interpolate_embeddings(embed_start, embed_end, t)
            
            # Generate frame conditioned on interpolated embedding
            frame = self.generate_from_embedding(
                embed_t, 
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
            generated_frames.append(frame)
        
        return generated_frames


def visualize_sample_grid(output_dir, num_samples=5):
    """
    Create visualization grid for first N samples
    Shows: Start | Target1 | Gen1 | Target2 | Gen2 | Target3 | Gen3 | End
    """
    output_dir = Path(output_dir)
    sample_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir()])[:num_samples]
    
    if len(sample_dirs) == 0:
        print("No samples to visualize!")
        return
    
    fig, axes = plt.subplots(num_samples, 8, figsize=(20, 2.5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, sample_dir in enumerate(sample_dirs):
        # Load images
        start = np.array(Image.open(sample_dir / "start.png"))
        end = np.array(Image.open(sample_dir / "end.png"))
        
        targets = [np.array(Image.open(sample_dir / f"target_{i:03d}.png")) for i in range(3)]
        interps = [np.array(Image.open(sample_dir / f"interp_{i:03d}.png")) for i in range(3)]
        
        # Plot
        axes[idx, 0].imshow(start)
        axes[idx, 0].set_title("Start", fontsize=10)
        axes[idx, 0].axis('off')
        
        for i in range(3):
            # Target
            axes[idx, 1 + i*2].imshow(targets[i])
            axes[idx, 1 + i*2].set_title(f"GT t={0.25*(i+1):.2f}", fontsize=10)
            axes[idx, 1 + i*2].axis('off')
            
            # Generated
            axes[idx, 2 + i*2].imshow(interps[i])
            axes[idx, 2 + i*2].set_title(f"Gen t={0.25*(i+1):.2f}", fontsize=10)
            axes[idx, 2 + i*2].axis('off')
        
        axes[idx, 7].imshow(end)
        axes[idx, 7].set_title("End", fontsize=10)
        axes[idx, 7].axis('off')
    
    plt.tight_layout()
    save_path = output_dir / "visualization_grid.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
    plt.close()


def main():
    # Configuration
    TEST_DIR = "data/data_split/test"
    OUTPUT_DIR = "data/lambda_baseline_results"
    NUM_SAMPLES = 2  # Test on 2 sequences
    NUM_INFERENCE_STEPS = 50  # Adjust for speed/quality tradeoff
    GUIDANCE_SCALE = 3.0  # Lower than typical text-to-image (3-5 works well for image variations)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DTYPE = torch.float32  # Use float32 for CPU compatibility
    
    print("="*60)
    print("Lambda Labs SD Image Variations - Baseline 3 Test")
    print("="*60)
    print(f"Device: {DEVICE}")
    
    # Initialize model
    model = LambdaSDImageVariations(device=DEVICE, dtype=DTYPE)
    
    # Load test dataset
    print(f"\nLoading test dataset from: {TEST_DIR}")
    dataset = AnitaDataset(TEST_DIR, between_frames=3)
    print(f"Total test samples: {len(dataset)}")
    print(f"Testing on: {NUM_SAMPLES} samples\n")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate frames for each test sample
    print("Generating inbetween frames...")
    for sample_id in tqdm(range(min(NUM_SAMPLES, len(dataset))), desc="Processing samples"):
        print(f"\nSample {sample_id + 1}/{NUM_SAMPLES}")
        batch = dataset[sample_id]
        
        start = batch["anchor_start"]  # [C, H, W] in [0, 1]
        end = batch["anchor_end"]      # [C, H, W] in [0, 1]
        targets = batch["targets"]      # [3, C, H, W] in [0, 1]
        
        # Create sample output directory
        sample_dir = Path(OUTPUT_DIR) / f"sample_{sample_id:04d}"
        sample_dir.mkdir(exist_ok=True)
        
        # Save start and end frames
        start_img = (start.permute(1, 2, 0).numpy() * 255).astype("uint8")
        end_img = (end.permute(1, 2, 0).numpy() * 255).astype("uint8")
        Image.fromarray(start_img).save(sample_dir / "start.png")
        Image.fromarray(end_img).save(sample_dir / "end.png")
        
        # Save ground truth targets
        for i in range(3):
            target_img = (targets[i].permute(1, 2, 0).numpy() * 255).astype("uint8")
            Image.fromarray(target_img).save(sample_dir / f"target_{i:03d}.png")
        
        # Generate inbetween frames
        try:
            generated_frames = model.generate_inbetween_frames(
                start, end,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE
            )
            
            # Save generated frames
            for i, frame in enumerate(generated_frames):
                # Resize to match target size if needed
                if frame.size != (targets.shape[3], targets.shape[2]):
                    frame = frame.resize((targets.shape[3], targets.shape[2]), Image.LANCZOS)
                frame.save(sample_dir / f"interp_{i:03d}.png")
            
            print(f"  ✓ Saved {len(generated_frames)} generated frames")
                
        except Exception as e:
            print(f"\n  ✗ Error processing sample {sample_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n✓ Generated frames saved to: {OUTPUT_DIR}")
    
    # Create visualization grid
    print("\nCreating visualization grid...")
    visualize_sample_grid(OUTPUT_DIR, num_samples=min(5, NUM_SAMPLES))
    
    # Run evaluation
    print("\n" + "="*60)
    print("Running Evaluation Metrics")
    print("="*60)
    
    evaluator = FrameInterpolationEvaluator(OUTPUT_DIR, device=DEVICE)
    all_results = evaluator.evaluate_all(num_samples=NUM_SAMPLES)
    stats = evaluator.compute_statistics(all_results)
    evaluator.save_results(all_results, stats, 'lambda_baseline_evaluation.json')
    
    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)
    print(f"Results directory: {OUTPUT_DIR}")
    print(f"Evaluation JSON: lambda_baseline_evaluation.json")
    print(f"Visualization: {OUTPUT_DIR}/visualization_grid.png")


if __name__ == "__main__":
    main()