import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
from diffusers import UNet2DModel, DDPMScheduler
from PIL import Image
import os
from tqdm.auto import tqdm
import numpy as np


def get_device():
    """
    Get the appropriate device (CUDA, MPS, or CPU)
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class PianoRollDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        # Convert grayscale to RGB while maintaining binary values
        gray_image = Image.open(image_path).convert('L')
        # Convert to binary image first (0 or 255)
        binary_image = gray_image.point(lambda x: 0 if x < 128 else 255, '1')
        # Convert to RGB
        rgb_image = binary_image.convert('RGB')
        image = self.transform(rgb_image)
        return image


def save_images(images, path, step):
    """Save a batch of images during training for monitoring."""
    images = (images / 2 + 0.5).clamp(0, 1)
    # Convert to binary
    images = (images > 0.5).float()
    grid = torchvision.utils.make_grid(images)
    # Convert to PIL image
    grid_image = torchvision.transforms.ToPILImage()(grid)
    os.makedirs(path, exist_ok=True)
    grid_image.save(f"{path}/sample_{step}.png")


def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, device):
    progress_bar = tqdm(total=config["num_epochs"] * len(train_dataloader))
    global_step = 0

    for epoch in range(config["num_epochs"]):
        model.train()
        for batch in train_dataloader:
            clean_images = batch.to(device)
            batch_size = clean_images.shape[0]

            # Sample noise and add to images
            noise = torch.randn(clean_images.shape).to(device)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (batch_size,),
                device=device
            ).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # Get model prediction
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]

            # Calculate loss
            loss = F.mse_loss(noise_pred, noise)
            print(f"loss:{loss}")

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.update(1)
            global_step += 1

            # Save sample images periodically
            if global_step % config["sample_interval"] == 0:
                model.eval()
                with torch.no_grad():
                    # Generate sample images
                    sample = torch.randn(8, 3, config["image_height"], config["image_width"]).to(device)
                    timesteps = torch.linspace(999, 0, 50).long().to(device)
                    for t in timesteps:
                        residual = model(sample, t.repeat(8), return_dict=False)[0]
                        sample = noise_scheduler.step(residual, t, sample).prev_sample
                save_images(sample, config["sample_dir"], global_step)
                model.train()

            if global_step % config["save_interval"] == 0:
                # Save checkpoint
                torch.save({
                    'step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, f"checkpoint_{global_step}.pt")

        # Save model after each epoch
        torch.save(model.state_dict(), f"model_epoch_{epoch}.pt")


def main():
    # Configuration
    config = {
        "image_height": 768,
        "image_width": 512,
        "batch_size": 2,
        "num_epochs": 1,
        "learning_rate": 1e-4,
        "save_interval": 100,
        "sample_interval": 1000,  # Interval for generating sample images
        "data_dir": "piano_roll_images",  # Your image directory
        "sample_dir": "samples"  # Directory to save generated samples
    }

    # Initialize device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")
    print(f"Using device: {device}")

    # Create dataset and dataloader
    dataset = PianoRollDataset(config["data_dir"])
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4
    )

    # Initialize model with 3 input/output channels for RGB
    model = UNet2DModel(
        sample_size=(config["image_height"], config["image_width"]),
        in_channels=3,  # RGB input
        out_channels=3,  # RGB output
        layers_per_block=1,
        block_out_channels=(32, 64, 128),  # Further reduced channels
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    ).to(device)

    # Initialize noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear"
    )

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

    # Train model
    train_loop(config, model, noise_scheduler, optimizer, dataloader, device)


if __name__ == "__main__":
    main()