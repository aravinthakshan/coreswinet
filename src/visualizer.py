import os
import torch
import numpy as np
import wandb 
import torchmetrics.image
from utils.dataloader import CBSD68Dataset 
from torch.utils.data import DataLoader
from utils.model.plsworkmodel import Model  
from utils.model.archs.ZSN2N import N2NNetwork

def load_models(main_model_path, n2n_model_path, device):
    """Loads the main model and n2n model from separate checkpoints."""
    
    # Load main model checkpoint
    main_checkpoint = torch.load(main_model_path, map_location=device)
    main_model = Model()  # Initialize main model
    main_model.load_state_dict(main_checkpoint['model_state_dict'])
    print(f"Loaded main model state dict from {main_model_path}.")
    
    # Load n2n model checkpoint
    n2n_checkpoint = torch.load(n2n_model_path, map_location=device)
    n2n_model = N2NNetwork()  # Initialize n2n model
    n2n_model.load_state_dict(n2n_checkpoint['model_state_dict'])
    print(f"Loaded n2n model state dict from {n2n_model_path}.")
    
    max_ssim = main_checkpoint['max_ssim']  # Or use n2n_checkpoint, if you prefer
    max_psnr = main_checkpoint['max_psnr']
    epoch = main_checkpoint['epoch']
    
    print(f"Loaded max_ssim: {max_ssim}, max_psnr: {max_psnr}, epoch: {epoch}.")
    
    return main_model, n2n_model


def un_tan_fi(data):
    d = data.clone()
    d += 1
    d /= 2
    return d

def get_metrics(clean, output, psnr_metric, ssim_metric,n2n = False):
    
    if n2n:
        cln = clean.clone()
    else:
        cln = clean
        
    psnr = psnr_metric(output, cln)
    ssim = ssim_metric(output, cln)
    return psnr.item(), ssim.item()



def get_statistics(noise, clean, output, idx, suffix='',n2n = False, wb=True):
    stats = {}
    examples = []
    
    for data, data_suffix in [
        (noise, f'noisy_input{suffix}'),
        (clean, f'ground_truth{suffix}'),
        (output, f'model_output{suffix}')
    ]:
        # Reverse tan_fi for ground truth and output
        if 'noisy_input' not in data_suffix and n2n==False:
            data = un_tan_fi(data)
        
        if n2n==True and 'ground_truth' in data_suffix:
            data = un_tan_fi(data)
            
        np_data = data.cpu().numpy()
        stats[data_suffix] = {
            'min': np_data.min(),
            'max': np_data.max(),
            'mean': np_data.mean(),
            'std': np_data.std()
        }
        
        print(f"\n{data_suffix} Statistics:")
        for key, value in stats[data_suffix].items():
            print(f"{key.capitalize()}: {value:.4f}")
        
        if wb:
            np_data = np.clip(np_data.transpose(1, 2, 0), 0, 1) * 255
            rgb_data = np_data.astype(np.uint8)
            image = wandb.Image(rgb_data, caption=f"{data_suffix}_{idx}.png")
            examples.append(image)
    
    if wb and examples:
        wandb.log({f"examples{suffix}": examples})
        print(f"Images for index {idx} saved in wandb with suffix {suffix}")
    
    return stats

def main_vis(val_dir, use_wandb=True, noise_level=25, crop_size=256, num_crops=32):
    """Main visualization function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if use_wandb:
        wandb.init(project="image-denoising", config={
            "noise_level": noise_level,
            "crop_size": crop_size,
            "num_crops": num_crops
        })
    
    main_model, n2n_model = load_models(
    './main_model/best_model.pth', 
    './n2n_model/best_model_n2n.pth', 
    device
)

    main_model.to(device).eval()
    n2n_model.to(device).eval()
    
    dataset = CBSD68Dataset(
        root_dir=val_dir, 
        noise_level=noise_level, 
        crop_size=crop_size, 
        num_crops=num_crops,
        normalize=True,
        tanfi=True 
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    psnr_metric = torchmetrics.image.PeakSignalNoiseRatio().to(device)
    ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure().to(device)
    
    selected_indices = [10, 20]  
    
    all_stats = []
    for i, (noise, clean) in enumerate(dataloader):
        if i not in selected_indices:
            continue
        
        noise, clean = noise.to(device), clean.to(device)
        
        with torch.no_grad():
           output_n2n = n2n_model.denoise(noise)
           output_main, _, _ = main_model(noise, output_n2n)
        
        # Get metrics for both models
        psnr_main, ssim_main = get_metrics(clean, output_main, psnr_metric, ssim_metric)
        psnr_n2n, ssim_n2n = get_metrics(clean, output_n2n, psnr_metric, ssim_metric,n2n=True)
        
        print(f"\nImage {i}:")
        print(f"Main Model - PSNR: {psnr_main:.4f}, SSIM: {ssim_main:.4f}")
        print(f"N2N Model  - PSNR: {psnr_n2n:.4f}, SSIM: {ssim_n2n:.4f}")
        
        # Get statistics for both models with different suffixes
        stats_main = get_statistics(noise[0], clean[0], output_main[0], i, suffix='_main', wb=use_wandb)
        stats_n2n = get_statistics(noise[0], clean[0], output_n2n[0], i, suffix='_n2n', wb=use_wandb,n2n=True)
        
        if use_wandb:
            wandb.log({
                f"image_{i}/main_psnr": psnr_main,
                f"image_{i}/main_ssim": ssim_main,
                f"image_{i}/n2n_psnr": psnr_n2n,
                f"image_{i}/n2n_ssim": ssim_n2n
            })
        
        all_stats.append({
            "main_model": stats_main,
            "n2n_model": stats_n2n
        })
    
    if use_wandb:
        wandb.finish()

if __name__ == '__main__':
    main_vis(val_dir='/path/to/CBSD68/dataset')