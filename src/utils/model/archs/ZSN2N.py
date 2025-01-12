import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torchvision.transforms import functional as TF
import numpy as np
from typing import Tuple, Optional

def calculate_metrics(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    """Calculate PSNR and SSIM for the given predictions and targets"""
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy().transpose(0, 2, 3, 1)
    
    # Convert to numpy arrays
    pred_np = to_numpy(pred.clamp(0, 1))
    target_np = to_numpy(target)
    
    # Calculate PSNR
    mse = np.mean((pred_np - target_np) ** 2)
    psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    
    # Calculate SSIM
    def ssim(img1, img2):
        C1 = (0.01 * 1) ** 2
        C2 = (0.03 * 1) ** 2
        
        mu1 = img1.mean()
        mu2 = img2.mean()
        
        sigma1_sq = ((img1 - mu1) ** 2).mean()
        sigma2_sq = ((img2 - mu2) ** 2).mean()
        sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
        
        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim.mean()
    
    ssim_val = np.mean([ssim(pred_np[i], target_np[i]) for i in range(pred_np.shape[0])])
    return psnr, ssim_val

def train_n2n(
    epochs: int,
    model: torch.nn.Module,
    clean_image: torch.Tensor,
    noise_generator: callable,
    device: str = 'cuda',
    image_size: Optional[Tuple[int, int]] = None,
    batch_size: int = 4,
    learning_rate: float = 0.001,
    scheduler_step_size: int = 1000,
    scheduler_gamma: float = 0.5
) -> torch.nn.Module:
    """
    Train N2N model on a single image
    
    Args:
        epochs: Number of training epochs
        model: N2N model to train
        clean_image: Clean reference image (1, C, H, W)
        noise_generator: Function that adds noise to images
        device: Device to train on
        image_size: Optional tuple of (height, width) to resize image
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        scheduler_step_size: Steps before learning rate decay
        scheduler_gamma: Learning rate decay factor
    """
    model = model.to(device)
    clean_image = clean_image.to(device)
    
    # Resize if specified
    if image_size is not None:
        clean_image = TF.resize(clean_image, size=image_size)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=scheduler_step_size, 
        gamma=scheduler_gamma
    )

    def pair_downsampler(img):
        c = img.shape[1]
        filter1 = torch.FloatTensor([[[[0, 0.5], [0.5, 0]]]]).to(img.device)
        filter2 = torch.FloatTensor([[[[0.5, 0], [0, 0.5]]]]).to(img.device)
        filter1 = filter1.repeat(c, 1, 1, 1)
        filter2 = filter2.repeat(c, 1, 1, 1)
        output1 = F.conv2d(img, filter1, stride=2, groups=c)
        output2 = F.conv2d(img, filter2, stride=2, groups=c)
        return output1, output2

    def loss_func(noisy_img):
        noisy1, noisy2 = pair_downsampler(noisy_img)
        pred1 = noisy1 - model(noisy1)
        pred2 = noisy2 - model(noisy2)
        loss_res = 0.5 * (F.mse_loss(noisy1, pred2) + F.mse_loss(noisy2, pred1))
        
        noisy_denoised = noisy_img - model(noisy_img)
        denoised1, denoised2 = pair_downsampler(noisy_denoised)
        loss_cons = 0.5 * (F.mse_loss(pred1, denoised1) + F.mse_loss(pred2, denoised2))
        
        return loss_res + loss_cons

    # Train N2N
    model.train()
    pbar = tqdm(range(epochs), desc="Training N2N")
    for epoch in pbar:
        # Generate noisy versions of the clean image
        noisy_batch = torch.cat([
            noise_generator(clean_image) for _ in range(batch_size)
        ], dim=0)
        
        # Training step
        loss = loss_func(noisy_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Calculate and display metrics
        if epoch % 10 == 0:  # Calculate metrics every 10 epochs
            model.eval()
            with torch.no_grad():
                noisy_test = noise_generator(clean_image)
                denoised = noisy_test - model(noisy_test)
                psnr, ssim = calculate_metrics(denoised, clean_image)
            model.train()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'PSNR': f'{psnr:.2f}',
                'SSIM': f'{ssim:.4f}'
            })
    
    return model

class N2NNetwork(nn.Module):
    def __init__(self, n_chan=3, chan_embed=48):
        super(N2NNetwork, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(n_chan, chan_embed, 3, padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv3 = nn.Conv2d(chan_embed, n_chan, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return x

    def denoise(self, noisy_img):
        with torch.no_grad():
            pred = torch.clamp(noisy_img - self(noisy_img), 0, 1)
        return pred

def train_n2n(epochs, train_loader, device='cuda'):
    """Pre-train the N2N model"""
    n2n_model = N2NNetwork().to(device)
    optimizer = optim.Adam(n2n_model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

    def pair_downsampler(img):
        c = img.shape[1]
        filter1 = torch.FloatTensor([[[[0, 0.5], [0.5, 0]]]]).to(img.device)
        filter2 = torch.FloatTensor([[[[0.5, 0], [0, 0.5]]]]).to(img.device)
        filter1 = filter1.repeat(c, 1, 1, 1)
        filter2 = filter2.repeat(c, 1, 1, 1)
        output1 = F.conv2d(img, filter1, stride=2, groups=c)
        output2 = F.conv2d(img, filter2, stride=2, groups=c)
        return output1, output2

    def loss_func(noisy_img):
        noisy1, noisy2 = pair_downsampler(noisy_img)
        pred1 = noisy1 - n2n_model(noisy1)
        pred2 = noisy2 - n2n_model(noisy2)
        loss_res = 0.5 * (F.mse_loss(noisy1, pred2) + F.mse_loss(noisy2, pred1))
        
        noisy_denoised = noisy_img - n2n_model(noisy_img)
        denoised1, denoised2 = pair_downsampler(noisy_denoised)
        loss_cons = 0.5 * (F.mse_loss(pred1, denoised1) + F.mse_loss(pred2, denoised2))
        
        return loss_res + loss_cons


    # Train N2N
    n2n_model.train()
    for epoch in tqdm(range(epochs), desc="Training N2N"):
        for batch in train_loader:
            noisy, _ = [x.to(device) for x in batch]
            loss = loss_func(noisy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    
    return n2n_model
