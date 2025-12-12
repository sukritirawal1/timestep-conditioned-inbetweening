import numpy as np
from PIL import Image
import torch
from skimage.metrics import structural_similarity as ssim
import lpips
import cv2
from pathlib import Path
import json
from tqdm import tqdm
import argparse


class SSIMMetric:
    """Structural Similarity Index Measure"""

    def __init__(self):
        self.name = "SSIM"

    def compute(self, img1, img2):
        """
        img1, img2: numpy arrays [H, W, 3], range [0, 255], uint8
        Returns: SSIM score (higher is better, max=1.0)
        """
        # Convert to grayscale for SSIM
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

        score = ssim(gray1, gray2, data_range=255)
        return score


class LPIPSMetric:
    """Learned Perceptual Image Patch Similarity"""

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.name = "LPIPS"
        self.device = device
        # Use AlexNet backbone (faster)
        self.model = lpips.LPIPS(net="alex").to(device)

    def compute(self, img1, img2):
        """
        img1, img2: numpy arrays [H, W, 3], range [0, 255], uint8
        Returns: LPIPS distance (lower is better, min=0.0)
        """
        # Convert to torch tensors [1, 3, H, W], range [-1, 1]
        t1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        t2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0

        with torch.no_grad():
            dist = self.model(t1.to(self.device), t2.to(self.device))

        return dist.item()


class TemporalLPIPSMetric:
    """Temporal consistency using LPIPS between consecutive frames"""

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.name = "T-LPIPS"
        self.lpips_metric = LPIPSMetric(device)

    def compute_sequence(self, frames):
        """
        frames: list of numpy arrays [H, W, 3]
        Returns: list of T-LPIPS scores between consecutive frames
        """
        scores = []
        for i in range(len(frames) - 1):
            score = self.lpips_metric.compute(frames[i], frames[i + 1])
            scores.append(score)
        return scores


class OpticalFlowWarpingMetric:
    """Temporal Optical Flow Warping Error (T-OF)"""

    def __init__(self):
        self.name = "T-OF"

    def compute_flow(self, gray1, gray2):
        """Compute optical flow using Farneback"""
        flow = cv2.calcOpticalFlowFarneback(
            gray1,
            gray2,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        return flow

    def warp_frame(self, frame, flow):
        """Warp frame using optical flow"""
        h, w = frame.shape[:2]
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (x + flow[..., 0]).astype(np.float32)
        map_y = (y + flow[..., 1]).astype(np.float32)
        warped = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
        return warped

    def compute_sequence(self, frames):
        """
        frames: list of numpy arrays [H, W, 3]
        Returns: list of warping errors (L1 distance)
        """
        errors = []
        for i in range(len(frames) - 1):
            curr = frames[i]
            next_frame = frames[i + 1]

            # Convert to grayscale for flow computation
            curr_gray = cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY)
            next_gray = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)

            # Compute flow from curr to next
            flow = self.compute_flow(curr_gray, next_gray)

            # Warp curr towards next
            warped = self.warp_frame(curr, flow)

            # L1 distance
            error = np.mean(np.abs(warped.astype(float) - next_frame.astype(float)))
            errors.append(error)

        return errors


class FrameInterpolationEvaluator:
    """Main evaluator for frame interpolation results"""

    def __init__(
        self, results_dir, device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.results_dir = Path(results_dir)
        self.device = device

        # Initialize metrics
        self.ssim = SSIMMetric()
        self.lpips = LPIPSMetric(device)
        self.t_lpips = TemporalLPIPSMetric(device)
        self.t_of = OpticalFlowWarpingMetric()

    def load_image(self, path):
        """Load image as numpy array [H, W, 3], uint8"""
        return np.array(Image.open(path).convert("RGB"))

    def evaluate_sample(self, sample_dir):
        """Evaluate a single sample"""
        sample_dir = Path(sample_dir)

        # Load images
        start = self.load_image(sample_dir / "start.png")
        end = self.load_image(sample_dir / "end.png")

        targets = [
            self.load_image(sample_dir / "target_000.png"),
            self.load_image(sample_dir / "target_001.png"),
            self.load_image(sample_dir / "target_002.png"),
        ]

        interps = [
            self.load_image(sample_dir / "interp_000.png"),
            self.load_image(sample_dir / "interp_001.png"),
            self.load_image(sample_dir / "interp_002.png"),
        ]

        results = {}

        # 1. SSIM for each frame
        results["ssim"] = [self.ssim.compute(interps[i], targets[i]) for i in range(3)]

        # 2. LPIPS for each frame
        results["lpips"] = [
            self.lpips.compute(interps[i], targets[i]) for i in range(3)
        ]

        # 3. T-LPIPS: temporal consistency of generated sequence
        full_sequence_pred = [start] + interps + [end]
        results["t_lpips"] = self.t_lpips.compute_sequence(full_sequence_pred)

        # 4. T-OF: temporal optical flow warping error
        results["t_of"] = self.t_of.compute_sequence(full_sequence_pred)

        return results

    def evaluate_all(self, num_samples=None):
        """Evaluate all samples in results directory"""
        sample_dirs = sorted([d for d in self.results_dir.iterdir() if d.is_dir()])

        if num_samples:
            sample_dirs = sample_dirs[:num_samples]

        all_results = []

        print(f"Evaluating {len(sample_dirs)} samples...")
        for sample_dir in tqdm(sample_dirs):
            try:
                results = self.evaluate_sample(sample_dir)
                results["sample_name"] = sample_dir.name
                all_results.append(results)
            except Exception as e:
                print(f"Error processing {sample_dir.name}: {e}")

        return all_results

    def compute_statistics(self, all_results):
        """Compute mean and std for each metric"""
        stats = {}

        # SSIM per frame
        for i in range(3):
            scores = [r["ssim"][i] for r in all_results]
            stats[f"ssim_frame_{i}_mean"] = np.mean(scores)
            stats[f"ssim_frame_{i}_std"] = np.std(scores)

        # LPIPS per frame
        for i in range(3):
            scores = [r["lpips"][i] for r in all_results]
            stats[f"lpips_frame_{i}_mean"] = np.mean(scores)
            stats[f"lpips_frame_{i}_std"] = np.std(scores)

        # T-LPIPS (average across sequence)
        all_t_lpips = [np.mean(r["t_lpips"]) for r in all_results]
        stats["t_lpips_mean"] = np.mean(all_t_lpips)
        stats["t_lpips_std"] = np.std(all_t_lpips)

        # T-OF (average across sequence)
        all_t_of = [np.mean(r["t_of"]) for r in all_results]
        stats["t_of_mean"] = np.mean(all_t_of)
        stats["t_of_std"] = np.std(all_t_of)

        # Overall averages
        stats["ssim_overall_mean"] = np.mean([np.mean(r["ssim"]) for r in all_results])
        stats["lpips_overall_mean"] = np.mean(
            [np.mean(r["lpips"]) for r in all_results]
        )

        return stats

    def save_results(self, all_results, stats, output_file="evaluation_results.json"):
        """Save results to JSON file"""
        output = {"statistics": stats, "per_sample_results": all_results}

        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to {output_file}")

        # Print summary
        print("\n" + "=" * 50)
        print("EVALUATION SUMMARY")
        print("=" * 50)
        print(f"SSIM (higher=better, max=1.0):  {stats['ssim_overall_mean']:.4f}")
        print(f"LPIPS (lower=better, min=0.0):  {stats['lpips_overall_mean']:.4f}")
        print(f"T-LPIPS (temporal consistency): {stats['t_lpips_mean']:.4f}")
        print(f"T-OF (warping error):           {stats['t_of_mean']:.4f}")
        print("=" * 50)


def main(input_dir, output_file):
    # Configuration

    # Run evaluation
    evaluator = FrameInterpolationEvaluator(input_dir)
    all_results = evaluator.evaluate_all()
    stats = evaluator.compute_statistics(all_results)
    evaluator.save_results(all_results, stats, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", "-i", type=str, default="output")
    parser.add_argument(
        "--output_file", "-o", type=str, default="evaluation_results.json"
    )
    args = parser.parse_args()
    main(args.input_dir, args.output_file)
