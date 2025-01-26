import torch
import torch.nn as nn
import argparse
from utils.model.coreswinet import Model
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
from utils.dataloader import CBSD68Dataset
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

def test_model(
    batch_size =16,
    # test_dir= ,
    # noise_level=25,
    crop_size=256, 
    num_crops=34,
    device='cuda',
    use_wandb=True,
):
    # Load models
    main_model_snap, _ = load_models(
        '/kaggle/input/trained-div2k/final_model (3).pth', 
        './n2n_model/best_model_n2n.pth', 
        device
    )
    # main_model, _ = load_models(
    #     './main_model/best_model.pth', 
    #     './n2n_model/best_model_n2n.pth', 
    #     device
    # )

    # Set model states
    for model in [main_model_snap]:
        model.to(device)
        model.eval()

    # Set bypass states after loading
    main_model_snap.bypass_second = True 
    # main_model.bypass_first = True
    print("Models loaded and bypass states set!")
    
    # Initialize dataset and metrics
    dataset = CBSD68Dataset(
        root_dir='/kaggle/input/cbsd68/CBSD68', 
        noise_level=25,
        crop_size=crop_size,
        num_crops=num_crops,
        normalize=True,
        tanfi=True 
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    psnr_metric = torchmetrics.image.PeakSignalNoiseRatio().to(device)
    ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure().to(device)
    
    total_psnr = total_ssim = 0
    num_batches = 0
    
    with torch.inference_mode():  # More efficient than no_grad
        with tqdm(dataloader, desc="Testing Progress") as loader:
            for noise, clean in loader:
                noise, clean = noise.to(device), clean.to(device)
                
                gt = un_tan_fi(clean)
                main_model_snap_out, _, _ = main_model_snap(noise, gt)
                # x = un_tan_fi(main_model_snap_out)
                # output_main, _, _ = main_model(noise, x)
                output_main = main_model_snap_out
                
                psnr_main, ssim_main = get_metrics(clean, output_main, psnr_metric, ssim_metric)
                
                total_psnr += psnr_main
                total_ssim += ssim_main
                num_batches += 1
                
                # loader.set_postfix(psnr=psnr_main, ssim=ssim_main)
    
    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches
    
    print(f"\nFinal Results:")
    print(f"Average PSNR TEST: {avg_psnr:.4f}")
    print(f"Average SSIM TEST: {avg_ssim:.4f}")

if __name__ == '__main__':
    test_model()
