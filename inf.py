import torch
import torchvision
from diffusers import UNet2DModel, DDPMScheduler
from torchvision import transforms
from PIL import Image
import os
from tqdm.auto import tqdm


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


def generate_images(
        checkpoint_path,
        image_height=768,
        image_width=512,
        output_dir="generated_images"
):
    # 配置
    config = {
        "image_height": image_height,
        "image_width": image_width,
        "sample_dir": output_dir
    }

    # 初始化设备
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")
    print(f"Using device: {device}")

    # 初始化模型
    model = UNet2DModel(
        sample_size=[image_height, image_width],
        in_channels=3,
        out_channels=3,
        layers_per_block=1,
        block_out_channels=(32, 64, 128),
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

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear"
    )

    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    with torch.no_grad():
        # Generate sample images
        sample = torch.randn(1, 3, config["image_height"], config["image_width"]).to(device)
        timesteps = torch.linspace(999, 0, 50).long().to(device)
        for t in timesteps:
            residual = model(sample, t.repeat(1), return_dict=False)[0]
            sample = noise_scheduler.step(residual, t, sample).prev_sample
    save_images(sample, config["sample_dir"], 500)


# 使用示例
if __name__ == "__main__":
    generate_images(
        checkpoint_path="checkpoint_500.pt",  # 你的checkpoint路径
        image_height=768,
        image_width=512,
        output_dir="generated_images"
    )