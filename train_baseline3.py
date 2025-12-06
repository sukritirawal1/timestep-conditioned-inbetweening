import torch 
from torch.utils.data import DataLoader
from dataloader import AnitaDataset
from diffusion import Diffusion
import os 
from tqdm import tqdm

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = Diffusion(device=device)
    train_dataset = AnitaDataset(root_dir="./data/data_split/train", between_frames=3)
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)

    optimizer = torch.optim.AdamW(model.keyframe_proj.parameters(), lr=1e-4)

    num_epochs = 5
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            start = batch["anchor_start"].to(device)
            end = batch["anchor_end"].to(device)
            targets = batch["targets"].to(device)

            num_intermediate_frames = targets.shape[1]
            batch_loss = 0.0
            for i in range(num_intermediate_frames):
                middle = targets[:, i, :, :, :]
                alpha = (i + 1 ) / (num_intermediate_frames + 1)

                z_start = model.encode_image_to_latent(start)
                z_end = model.encode_image_to_latent(end)
                z_target = model.encode_image_to_latent(middle)

                z_interp = model.slerp(z_start, z_end, alpha)
                keyframe_cond = model.encode_keyframes_to_conditioning(start, end)
                
                noise = torch.randn_like(z_target)
                timesteps = torch.randint(0, 1000, (z_target.shape[0],), device=device)
                noisy_latents = model.scheduler.add_noise(z_target, noise, timesteps)

                noise_pred = model.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=keyframe_cond
                ).sample

                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                batch_loss += loss

            
            # Average loss across intermediate frames
            batch_loss = batch_loss / num_intermediate_frames
            
            # Backprop once per batch
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{batch_loss.item():.4f}'})

        # After each epoch
        avg_loss = total_loss / num_batches
        print(f"\nEpoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        os.makedirs("checkpoints/baseline3", exist_ok=True)
        torch.save({
            "epoch": epoch,
            "keyframe_proj": model.keyframe_proj.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": avg_loss,
        }, f"checkpoints/baseline3/checkpoint_epoch_{epoch+1}.pth")
        print(f"Checkpoint saved: checkpoints/baseline3/checkpoint_epoch_{epoch+1}.pth")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "keyframe_proj": model.keyframe_proj.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": avg_loss,
            }, "checkpoints/baseline3/best_model.pth")
            print(f"Best model saved! Loss: {best_loss:.4f}")

    print(f"\nTraining completed. Best loss: {best_loss:.4f}")

if __name__ == "__main__":
    train()