import os
import torch
import numpy as np
import wandb 
import torchmetrics.image
from utils.dataloader import BSD400 
from torch.utils.data import DataLoader
from utils.model.plsworkmodel import Model  
from utils.model.archs.ZSN2N import N2NNetwork

def load_models(model_path, device):
    """Loads the main model and n2n model from a checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    
    main_model = Model()  
    main_model_dict = main_model.state_dict()
    
    main_pretrained_dict = {k: v for k, v in checkpoint['main_model'].items()
                            if k in main_model_dict and v.shape == main_model_dict[k].shape}
    main_model_dict.update(main_pretrained_dict)
    main_model.load_state_dict(main_model_dict, strict=False)
    print(f"Loaded {len(main_pretrained_dict)} / {len(main_model_dict)} layers into main model")
    
    n2n_model = N2NNetwork() 
    n2n_model_dict = n2n_model.state_dict()
    
    n2n_pretrained_dict = {k: v for k, v in checkpoint['n2n_model'].items()
                           if k in n2n_model_dict and v.shape == n2n_model_dict[k].shape}
    n2n_model_dict.update(n2n_pretrained_dict)
    n2n_model.load_state_dict(n2n_model_dict, strict=False)
    print(f"Loaded {len(n2n_pretrained_dict)} / {len(n2n_model_dict)} layers into n2n model")
    
    return main_model, n2n_model

def un_tan_fi(data):

    d = data.clone()
    d += 1
    d /= 2
    return d

def get_metrics(clean, output, psnr_metric, ssim_metric):

    psnr = psnr_metric(output, clean)
    ssim = ssim_metric(output, clean)
    return psnr.item(), ssim.item()

def get_statistics(noise, clean, output, idx, wb=True):

    stats = {}
    examples = []
    
    for data, suffix in [
        (noise, 'noisy_input'),
        (clean, 'ground_truth'),
        (output, 'model_output')
    ]:
        # Reverse tan_fi for ground truth and output
        if suffix != 'noisy_input':
            data = un_tan_fi(data)
        
        np_data = data.cpu().numpy()
        stats[suffix] = {
            'min': np_data.min(),
            'max': np_data.max(),
            'mean': np_data.mean(),
            'std': np_data.std()
        }
        
        print(f"\n{suffix} Statistics:")
        for key, value in stats[suffix].items():
            print(f"{key.capitalize()}: {value:.4f}")
        
        if wb:
            np_data = np.clip(np_data.transpose(1, 2, 0), 0, 1) * 255
            rgb_data = np_data.astype(np.uint8)
            image = wandb.Image(rgb_data, caption=f"{suffix}_{idx}.png")
            examples.append(image)
    
    if wb and examples:
        wandb.log({"examples": examples})
        print(f"Images for index {idx} saved in wandb")
    



def main_vis(val_dir, model_path="./best_models.pth", use_wandb=True, noise_level=25, crop_size=256, num_crops=32):
    """Main visualization function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if use_wandb:
        wandb.init(project="image-denoising", config={
            "noise_level": noise_level,
            "crop_size": crop_size,
            "num_crops": num_crops
        })
    
    main_model, n2n_model = load_models(model_path, device)
    main_model.to(device).eval()
    n2n_model.to(device).eval()
    
    dataset = BSD400(
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
            output_n2n = n2n_model(noise)
            output_main,_,_ = main_model(noise, output_n2n)
        
        psnr_main, ssim_main = get_metrics(clean, output_main, psnr_metric, ssim_metric)
        print(f"\nImage {i} - Main Model: PSNR: {psnr_main:.4f}, SSIM: {ssim_main:.4f}")
        
        stats_main = get_statistics(noise[0], clean[0], output_main[0], i, wb=use_wandb)
        # stats_n2n = get_statistics(noise[0], clean[0], output_n2n[0], i, wb=use_wandb)
        
        all_stats.append({
            "main_model": stats_main,
            # "n2n_model": stats_n2n
        })
    
    if use_wandb:
        wandb.finish()

if __name__ == '__main__':
    main_vis(val_dir='/kaggle/input/bsd-400/BSD_400/BSD400_noisy_25')