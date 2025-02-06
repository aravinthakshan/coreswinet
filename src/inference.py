import torch
import torch.nn as nn
import argparse
from utils.model.coreswinet import Model, replace_decoder_convs
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
from utils.dataloader import CBSD68Dataset,McMasterDataset
from torch.utils.data import DataLoader, Subset
import torchmetrics
from tqdm import tqdm
from utils.misc import get_metrics
from visualizer import load_models, get_metrics, get_statistics
import wandb 

def un_tan_fi(data):
    d = data.clone()
    d += 1
    d /= 2
    return d

def test(
    batch_size,
    test_dir,
    use_wandb=True,
    device='cuda',
    noise_level=25,
    test_dataset='CBSD68',
    crop_size=256, 
    num_crops=34,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if use_wandb:
        wandb.init(project="DeFInet", config={
            "noise_level": noise_level,
            "crop_size": crop_size,
            "num_crops": num_crops
        })
    
    main_model = load_models(
        './main_model/best_model.pth',
        device
    )

    main_model.to(device).eval()
    
    main_model.bypass = True

    print("Main Model Bypass!")
    if test_dataset=='CBSD68':
        dataset = CBSD68Dataset(
            root_dir='/kaggle/input/cbsd68/CBSD68', 
            noise_level=25,
            crop_size=256,
            num_crops=34,
            normalize=True,
            tanfi=True 
        )

    elif test_dataset=='mcmaster':
        dataset = McMasterDataset(
            root_dir='/kaggle/input/mcmaster/McMaster', 
            noise_level=25,
            crop_size=256,
            num_crops=34,
            normalize=True,
            tanfi=True 
        )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    psnr_metric = torchmetrics.image.PeakSignalNoiseRatio().to(device)
    ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure().to(device)
    
    # Reset metrics at start
    psnr_metric.reset()
    ssim_metric.reset()
    
    # Initialize running averages
    total_psnr = 0
    total_ssim = 0
    num_batches = 0
    
    with tqdm(dataloader, desc="Testing Progress") as loader:
        for noise, clean in loader:
            noise, clean = noise.to(device), clean.to(device)
            
            with torch.no_grad():
                output_n2n = un_tan_fi(clean)
                output_main, _ = main_model(noise, output_n2n)
            
            # Calculate metrics for main model
            psnr_main, ssim_main = get_metrics(clean, output_main, psnr_metric, ssim_metric)
            
            # Update running averages (per batch, not per image)
            total_psnr += psnr_main
            total_ssim += ssim_main
            num_batches += 1
            
            # Update progress bar
            loader.set_postfix(psnr=psnr_main, ssim=ssim_main)
    
    # Calculate final averages
    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches
    
    print(f"\nFinal Results:")
    print(f"Average PSNR TEST: {avg_psnr:.4f}")
    print(f"Average SSIM TEST: {avg_ssim:.4f}")
    
    if use_wandb:
        wandb.log({
            "avg_psnr_test": avg_psnr,
            "avg_ssim_test": avg_ssim
        })
        wandb.finish()

def test_model(config):
    test(
        config['batch_size'],
        config['test_dir'],
        config['wandb'],
        config['device'], 
        config['noise_level'],
        config['test_dataset']
    )