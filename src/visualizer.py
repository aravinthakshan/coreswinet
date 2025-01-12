import os
import torch
import numpy as np
import wandb 
import torchmetrics.image
from utils.dataloader import CBSD68Dataset 
from torch.utils.data import DataLoader
from utils.model.plsworkmodel import Model  

def load_model(model_path, device):

    model = Model()
    state_dict = torch.load(model_path, map_location=device)
    model_dict = model.state_dict()
    
    # Selective loading of weights
    pretrained_dict = {k: v for k, v in state_dict.items() 
                       if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)
    
    print(f"Loaded {len(pretrained_dict)} / {len(state_dict)} layers")
    return model

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
            # Convert to uint8 for visualization
            np_data = np.clip(np_data.transpose(1, 2, 0), 0, 1) * 255
            rgb_data = np_data.astype(np.uint8)
            image = wandb.Image(rgb_data, caption=f"{suffix}_{idx}.png")
            examples.append(image)
    
    if wb and examples:
        wandb.log({"examples": examples})
        print(f"Images for index {idx} saved in wandb")
    
    return stats

def main_vis(val_dir, model_path="./best_model.pth", use_wandb=True, 
             noise_level=25, crop_size=256, num_crops=32):

    # Setup device and wandb if needed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if use_wandb:
        wandb.init(project="image-denoising", config={
            "noise_level": noise_level,
            "crop_size": crop_size,
            "num_crops": num_crops
        })
    
    model = load_model(model_path, device)
    model.to(device)
    model.eval()
    
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
            output = model(noise)
        
        psnr, ssim = get_metrics(clean, output, psnr_metric, ssim_metric)
        print(f"\nImage {i}: PSNR: {psnr:.4f}, SSIM: {ssim:.4f}")
        
        stats = get_statistics(noise[0], clean[0], output[0], i, wb=use_wandb)
        all_stats.append(stats)
    
    if use_wandb:
        wandb.finish()

if __name__ == '__main__':
    main_vis(val_dir='/path/to/CBSD68/dataset')