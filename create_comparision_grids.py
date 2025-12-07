import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

def create_comparison_grid(sample_dir, output_path, num_inbetweens=3):
    """
    Create a grid comparing keyframes, generated frames, and targets across guidance scales
    
    Grid layout:
    - Columns: Start, Frame1, Frame2, Frame3, End
    - Rows: Ground Truth + one row for each guidance scale
    
    Args:
        sample_dir: Directory containing guidance_X.X folders
        output_path: Path to save the comparison grid
        num_inbetweens: Number of inbetween frames
    """
    # Find all guidance scale directories
    guidance_dirs = {}
    for item in os.listdir(sample_dir):
        item_path = os.path.join(sample_dir, item)
        if os.path.isdir(item_path) and item.startswith("guidance_"):
            try:
                scale = float(item.replace("guidance_", ""))
                guidance_dirs[scale] = item_path
            except ValueError:
                continue
    
    if not guidance_dirs:
        print(f"No guidance scale directories found in {sample_dir}")
        return
    
    # Sort guidance scales
    guidance_scales = sorted(guidance_dirs.keys())
    
    # Check if ground truth exists
    gt_dir = os.path.join(sample_dir, "ground_truth")
    has_gt = os.path.exists(gt_dir)
    
    # Determine grid dimensions
    # Columns: Start, Frame1, Frame2, Frame3, End (5 columns)
    # Rows: GT (if exists) + one for each guidance scale
    n_cols = 5  # Start, 3 inbetweens, End
    n_rows = len(guidance_scales) + (1 if has_gt else 0)
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    
    # Handle single row case
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    row_idx = 0
    
    # Add ground truth row first
    if has_gt:
        # Start frame
        start_path = os.path.join(gt_dir, "00_start.png")
        if os.path.exists(start_path):
            img = Image.open(start_path)
            axes[row_idx, 0].imshow(img)
        axes[row_idx, 0].set_title("Start", fontsize=10)
        axes[row_idx, 0].axis('off')
        
        # Inbetween frames
        for col_idx in range(1, 4):  # Columns 1, 2, 3
            frame_num = col_idx
            frame_path = os.path.join(gt_dir, f"{frame_num:02d}_target.png")
            if os.path.exists(frame_path):
                img = Image.open(frame_path)
                axes[row_idx, col_idx].imshow(img)
            axes[row_idx, col_idx].set_title(f"Frame {frame_num}", fontsize=10)
            axes[row_idx, col_idx].axis('off')
        
        # End frame
        end_path = os.path.join(gt_dir, "04_end.png")
        if os.path.exists(end_path):
            img = Image.open(end_path)
            axes[row_idx, 4].imshow(img)
        axes[row_idx, 4].set_title("End", fontsize=10)
        axes[row_idx, 4].axis('off')
        
        # Add row label
        axes[row_idx, 0].text(-0.15, 0.5, "Ground Truth", 
                             transform=axes[row_idx, 0].transAxes,
                             rotation=90, va='center', ha='center', 
                             fontsize=12, fontweight='bold')
        row_idx += 1
    
    # Add guidance scale rows
    for guidance_scale in guidance_scales:
        guidance_dir = guidance_dirs[guidance_scale]
        
        # Start frame
        start_path = os.path.join(guidance_dir, "00_start.png")
        if os.path.exists(start_path):
            img = Image.open(start_path)
            axes[row_idx, 0].imshow(img)
        axes[row_idx, 0].set_title("Start", fontsize=10)
        axes[row_idx, 0].axis('off')
        
        # Inbetween frames
        for col_idx in range(1, 4):  # Columns 1, 2, 3
            frame_num = col_idx
            frame_path = os.path.join(guidance_dir, f"{frame_num:02d}_inbetween.png")
            if os.path.exists(frame_path):
                img = Image.open(frame_path)
                axes[row_idx, col_idx].imshow(img)
            axes[row_idx, col_idx].set_title(f"Frame {frame_num}", fontsize=10)
            axes[row_idx, col_idx].axis('off')
        
        # End frame
        end_path = os.path.join(guidance_dir, "04_end.png")
        if os.path.exists(end_path):
            img = Image.open(end_path)
            axes[row_idx, 4].imshow(img)
        axes[row_idx, 4].set_title("End", fontsize=10)
        axes[row_idx, 4].axis('off')
        
        # Add row label
        axes[row_idx, 0].text(-0.15, 0.5, f"Guidance={guidance_scale}", 
                             transform=axes[row_idx, 0].transAxes,
                             rotation=90, va='center', ha='center', 
                             fontsize=12, fontweight='bold')
        row_idx += 1
    
    plt.suptitle(f"Sample: {os.path.basename(sample_dir)}", fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison grid to {output_path}")

def create_all_comparison_grids(results_dir, output_dir=None, num_inbetweens=3):
    """
    Create comparison grids for all samples in results directory
    
    Args:
        results_dir: Directory containing sample_XXXX folders
        output_dir: Directory to save comparison grids (default: results_dir/comparisons)
        num_inbetweens: Number of inbetween frames
    """
    if output_dir is None:
        output_dir = os.path.join(results_dir, "comparisons")
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all sample directories
    sample_dirs = []
    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if os.path.isdir(item_path) and item.startswith("sample_"):
            sample_dirs.append((item, item_path))
    
    sample_dirs.sort()  # Sort by sample number
    
    print(f"Found {len(sample_dirs)} samples")
    
    for sample_name, sample_dir in sample_dirs:
        output_path = os.path.join(output_dir, f"{sample_name}_comparison.png")
        create_comparison_grid(sample_dir, output_path, num_inbetweens=num_inbetweens)
    
    print(f"\nAll comparison grids saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create comparison grids for guidance scale results")
    parser.add_argument("--results_dir", type=str, default="guidance_test",
                       help="Directory containing sample_XXXX folders")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save comparison grids (default: results_dir/comparisons)")
    parser.add_argument("--num_inbetweens", type=int, default=3,
                       help="Number of inbetween frames")
    parser.add_argument("--sample", type=str, default=None,
                       help="Process only a specific sample (e.g., sample_0000)")
    
    args = parser.parse_args()
    
    if args.sample:
        # Process single sample
        sample_dir = os.path.join(args.results_dir, args.sample)
        if not os.path.exists(sample_dir):
            print(f"Sample directory not found: {sample_dir}")
        else:
            output_path = os.path.join(args.results_dir, f"{args.sample}_comparison.png")
            create_comparison_grid(sample_dir, output_path, num_inbetweens=args.num_inbetweens)
    else:
        # Process all samples
        create_all_comparison_grids(args.results_dir, args.output_dir, 
                                  num_inbetweens=args.num_inbetweens)