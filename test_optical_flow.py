from dataloader import AnitaDataset
from optical_flow import OpticalFlow
from PIL import Image
import os

# Configuration
TEST_DIR = "data/data_split/test"
OUTPUT_DIR = "data/optical_flow_results"
NUM_SAMPLES = 5  # Change to None for all samples

# Initialize
dataset = AnitaDataset(TEST_DIR, between_frames=3)
optical_flow = OpticalFlow()
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Total samples: {len(dataset)}")

for sample_id in range(min(NUM_SAMPLES or len(dataset), len(dataset))):
    batch = dataset[sample_id]
    
    start = batch["anchor_start"]
    end = batch["anchor_end"]
    targets = batch["targets"]
    
    out_dir = f"{OUTPUT_DIR}/sample_{sample_id:04d}"
    os.makedirs(out_dir, exist_ok=True)
    
    # Save start, end
    Image.fromarray((start.permute(1,2,0).numpy() * 255).astype("uint8")).save(f"{out_dir}/start.png")
    Image.fromarray((end.permute(1,2,0).numpy() * 255).astype("uint8")).save(f"{out_dir}/end.png")
    
    # Save targets
    for t in range(3):
        tgt = (targets[t].permute(1,2,0).numpy() * 255).astype("uint8")
        Image.fromarray(tgt).save(f"{out_dir}/target_{t:03d}.png")
    
    # Generate optical flow
    for j, t in enumerate([0.25, 0.5, 0.75]):
        interp = optical_flow.interpolate(start, end, t)
        interp_img = (interp.permute(1,2,0).numpy() * 255).astype("uint8")
        Image.fromarray(interp_img).save(f"{out_dir}/interp_{j:03d}.png")
    
    print(f"Sample {sample_id + 1}")

print(f"\nDone! Results in {OUTPUT_DIR}/")