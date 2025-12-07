import torch
from PIL import Image
import numpy as np
from SlerpKeyframeConditionedDiffusion import SlerpKeyframeConditionedDiffusion
from dataloader import AnitaDataset
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm

def load_checkpoint(model, checkpoint_path):
    """Load checkpoint into model"""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=model.device)
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            elif 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        # Ensure visual_adapter is on the correct device
        if hasattr(model, 'visual_adapter'):
            model.visual_adapter = model.visual_adapter.to(model.device)
        
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")

def tensor_to_pil(img_tensor):
    """Convert tensor to PIL Image"""
    img = img_tensor.detach().cpu().clamp(0, 1)
    img = (img * 255).byte().permute(1, 2, 0).numpy()  # (H, W, 3)
    return Image.fromarray(img)

def test_guidance_scales_on_dataset(val_dir, checkpoint_path, 
                                   guidance_scales=[1.0, 2.0, 3.0, 5.0, 7.5], 
                                   num_inbetweens=3, num_inference_steps=25, 
                                   noise_strength=0.3, output_dir="guidance_test",
                                   batch_size=1, image_shape=(224, 224), max_samples=None, seed=None):
    """
    Test model with different guidance scales on validation dataset
    
    Args:
        val_dir: Path to validation directory
        checkpoint_path: Path to model checkpoint
        guidance_scales: List of guidance scales to test
        num_inbetweens: Number of inbetween frames to generate
        num_inference_steps: Number of denoising steps
        noise_strength: Noise strength for denoising
        output_dir: Directory to save results
        batch_size: Batch size for processing
        image_shape: Image shape (H, W)
        max_samples: Maximum number of samples to test (None for all)
        seed: Random seed for shuffling (None for random)
    """
    # Set random seed if provided
    if seed is not None:
        import random
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize model
    print("Initializing model...")
    model = SlerpKeyframeConditionedDiffusion(device=device)
    model.eval()
    
    # Load checkpoint
    load_checkpoint(model, checkpoint_path)
    
    # Load validation dataset
    print(f"Loading validation dataset from {val_dir}...")
    val_dataset = AnitaDataset(root_dir=val_dir, between_frames=num_inbetweens, image_shape=image_shape)
    
    # Limit dataset size if max_samples is specified
    if max_samples is not None and max_samples < len(val_dataset):
        # Shuffle indices to get diverse samples from different dramas
        import random
        all_indices = list(range(len(val_dataset)))
        random.shuffle(all_indices)
        indices = all_indices[:max_samples]
        val_dataset = torch.utils.data.Subset(val_dataset, indices)
        print(f"Limited to {max_samples} samples (shuffled for diversity)")
    else:
        # Still shuffle if processing all samples to get diverse order
        import random
        all_indices = list(range(len(val_dataset)))
        random.shuffle(all_indices)
        val_dataset = torch.utils.data.Subset(val_dataset, all_indices)
        print(f"Shuffled dataset for diversity")
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    total_samples = len(val_dataset) if isinstance(val_dataset, torch.utils.data.Subset) else len(val_dataset)
    print(f"Testing on {total_samples} samples")
    
    os.makedirs(output_dir, exist_ok=True)
    
    samples_processed = 0
    
    # Process each batch
    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Processing batches")):
        start_frames = batch["anchor_start"].to(device)  # [B, C, H, W]
        end_frames = batch["anchor_end"].to(device)      # [B, C, H, W]
        targets = batch["targets"].to(device)            # [B, num_inbetweens, C, H, W]
        
        batch_size_actual = start_frames.shape[0]
        
        # Process each sample in batch
        for sample_idx in range(batch_size_actual):
            # Check if we've reached max_samples
            if max_samples is not None and samples_processed >= max_samples:
                break
                
            start_frame = start_frames[sample_idx:sample_idx+1]  # Keep batch dim
            end_frame = end_frames[sample_idx:sample_idx+1]
            target_frames = targets[sample_idx]  # [num_inbetweens, C, H, W]
            
            # Create output directory for this sample
            sample_dir = os.path.join(output_dir, f"sample_{samples_processed:04d}")
            os.makedirs(sample_dir, exist_ok=True)
            
            # Save ground truth frames
            gt_dir = os.path.join(sample_dir, "ground_truth")
            os.makedirs(gt_dir, exist_ok=True)
            tensor_to_pil(start_frame[0]).save(os.path.join(gt_dir, "00_start.png"))
            for i in range(num_inbetweens):
                tensor_to_pil(target_frames[i]).save(os.path.join(gt_dir, f"{i+1:02d}_target.png"))
            tensor_to_pil(end_frame[0]).save(os.path.join(gt_dir, f"{num_inbetweens+1:02d}_end.png"))
            
            # Test each guidance scale
            for guidance_scale in guidance_scales:
                with torch.no_grad():
                    inbetween_frames = model.predict_inbetween_sequence(
                        start_frame,
                        end_frame,
                        num_inbetweens=num_inbetweens,
                        num_inference_steps=num_inference_steps,
                        noise_strength=noise_strength,
                        guidance_scale=guidance_scale
                    )
                
                # Save results for this guidance scale
                scale_dir = os.path.join(sample_dir, f"guidance_{guidance_scale}")
                os.makedirs(scale_dir, exist_ok=True)
                
                # Save start frame
                tensor_to_pil(start_frame[0]).save(os.path.join(scale_dir, "00_start.png"))
                
                # Save inbetween frames
                for i, frame_list in enumerate(inbetween_frames):
                    frame_list[0].save(os.path.join(scale_dir, f"{i+1:02d}_inbetween.png"))
                
                # Save end frame
                tensor_to_pil(end_frame[0]).save(os.path.join(scale_dir, f"{num_inbetweens+1:02d}_end.png"))
            
            samples_processed += 1
        
        # Break outer loop if we've reached max_samples
        if max_samples is not None and samples_processed >= max_samples:
            break
    
    print(f"\nAll tests complete! Results saved in {output_dir}")
    print(f"Processed {samples_processed} samples with {len(guidance_scales)} guidance scales each")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test model with different guidance scales on validation dataset")
    parser.add_argument("--val_dir", type=str, default="val", 
                       help="Path to validation directory")
    parser.add_argument("--checkpoint", type=str, default="model_ckpt/clip_slerp/checkpoint.ckpt", 
                       help="Path to model checkpoint")
    parser.add_argument("--guidance_scales", type=float, nargs="+", default=[1.0, 2.0, 3.0, 5.0, 7.5],
                       help="List of guidance scales to test")
    parser.add_argument("--num_inbetweens", type=int, default=3, help="Number of inbetween frames")
    parser.add_argument("--num_inference_steps", type=int, default=25, help="Number of denoising steps")
    parser.add_argument("--noise_strength", type=float, default=0.3, help="Noise strength")
    parser.add_argument("--output_dir", type=str, default="guidance_test", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing")
    parser.add_argument("--image_shape", type=int, nargs=2, default=[224, 224], 
                       help="Image shape (height width)")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to test (default: all)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for shuffling (default: random)")
    
    args = parser.parse_args()
    
    test_guidance_scales_on_dataset(
        args.val_dir,
        args.checkpoint,
        guidance_scales=args.guidance_scales,
        num_inbetweens=args.num_inbetweens,
        num_inference_steps=args.num_inference_steps,
        noise_strength=args.noise_strength,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        image_shape=tuple(args.image_shape),
        max_samples=args.max_samples,
        seed=args.seed
    )