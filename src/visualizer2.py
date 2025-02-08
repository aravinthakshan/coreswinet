import os
import torch
import numpy as np
import torchmetrics.image
import matplotlib.pyplot as plt
from utils.dataloader import uiebd_dataset
from torch.utils.data import DataLoader
from utils.model.coreswinet import Model

def save_image(image_tensor, filename):
    """Saves a tensor image as a file."""
    image = image_tensor.cpu().numpy().transpose(1, 2, 0)
    image = np.clip(image, 0, 1) * 255  # Scale to [0, 255]
    image = image.astype(np.uint8)
    plt.imsave(filename, image)

def load_model(model_path, device):
    """Loads the main model from a checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    model = Model()
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model state dict from {model_path}.")
    return model

def un_tan_fi(data):
    """Reverses tanh normalization."""
    d = data.clone()
    d += 1
    d /= 2
    return d

def get_metrics(clean, output, psnr_metric, ssim_metric):
    """Computes PSNR and SSIM metrics."""
    psnr = psnr_metric(output, clean)
    ssim = ssim_metric(output, clean)
    return psnr.item(), ssim.item()

def main_vis(test_dir, noise_level=25, crop_size=128, num_crops=32):
    """Main visualization function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    main_model = load_model('/kaggle/input/underwater-image/best_model.pth', device)
    main_model.to(device).eval()
    main_model.bypass = True 
    print("Main Model Bypass Enabled!")

    dataset = uiebd_dataset(
        root_dir='/kaggle/input/cbsd68/CBSD68', 
        noise_level=noise_level, 
        crop_size=crop_size, 
        num_crops=num_crops,
        normalize=True,
        tanfi=True 
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    psnr_metric = torchmetrics.image.PeakSignalNoiseRatio().to(device)
    ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure().to(device)

    output_dir = "/kaggle/working"
    os.makedirs(output_dir, exist_ok=True)

    selected_indices = [10, 20]
    
    for i, (noise, clean) in enumerate(dataloader):
        if i not in selected_indices:
            continue

        noise, clean = noise.to(device), clean.to(device)

        with torch.no_grad():
            output_n2n = un_tan_fi(clean)
            output_main, _, _ = main_model(noise, output_n2n)

        psnr_main, ssim_main = get_metrics(clean, output_main, psnr_metric, ssim_metric)
        print(f"\nImage {i}:")
        print(f"Main Model - PSNR: {psnr_main:.4f}, SSIM: {ssim_main:.4f}")

        save_image(noise[0], os.path.join(output_dir, f"noisy_input_{i}.png"))
        save_image(clean[0], os.path.join(output_dir, f"ground_truth_{i}.png"))
        save_image(output_main[0], os.path.join(output_dir, f"model_output_{i}.png"))
        print(f"Saved images for index {i} in {output_dir}")

if __name__ == '__main__':
    main_vis('hello')