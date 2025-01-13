import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torchvision.transforms import functional as TF
import numpy as np
import os 
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def test(model, noisy_img, clean_img, save_dir="predictions/"):
    with torch.no_grad():
        # Prediction calculation
        pred = torch.clamp(noisy_img - model(noisy_img), 0, 1)
        
        # Calculate PSNR
        mse = F.mse_loss(clean_img, pred).item()
        psnr = 10 * np.log10(1 / mse)
        
    # Print range of prediction
    min_value = pred.min().item()
    max_value = pred.max().item()
    print(f"Range of values in pred_img: min={min_value}, max={max_value}")
    
    # Remove batch dimension if present and ensure the correct format (C x H x W)
    pred_img = pred.squeeze(0)  # Remove the batch dimension (if batch size is 1)
    
    # If it's a 3D tensor (C x H x W), convert to PIL Image
    if pred_img.ndimension() == 3:  # RGB or multi-channel image
        pred_img = pred_img.permute(1, 2, 0)  # Convert to H x W x C for PIL
    elif pred_img.ndimension() == 2:  # Grayscale image
        pred_img = pred_img.unsqueeze(0)  # Add channel dimension for grayscale (H x W -> 1 x H x W)
    
    pred_img = transforms.ToPILImage()(pred_img.cpu())  # Convert to PIL Image
    
    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Define the file path and save the image
    image_path = f"{save_dir}/predicted_image.png"
    pred_img.save(image_path)
    print(f"Saved predicted image to {image_path}")
    
    return psnr


def un_tan_fi(data):
    d = data.clone()
    d += 1
    d /= 2
    return d

def train_n2n(
    epochs: int,
    model: torch.nn.Module,
    dataloader,
    device: str = 'cuda',
) -> torch.nn.Module:
    """Train N2N model on a single image"""
    model = model.to(device)
    
    # Get single image pair from dataset
    noisy_img, clean_img = next(iter(dataloader))
    noisy_img, clean_img = noisy_img.to(device), clean_img.to(device)
    
    # beacuse it's comparing against a tanified image ( very low psnr )
    clean_img = un_tan_fi(clean_img)
    
    # Setup optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.001)
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
        pred1 = noisy1 - model(noisy1)
        pred2 = noisy2 - model(noisy2)
        loss_res = 0.5 * (F.mse_loss(noisy1, pred2) + F.mse_loss(noisy2, pred1))
        
        noisy_denoised = noisy_img - model(noisy_img)
        denoised1, denoised2 = pair_downsampler(noisy_denoised)
        loss_cons = 0.5 * (F.mse_loss(pred1, denoised1) + F.mse_loss(pred2, denoised2))
        
        return loss_res + loss_cons

    # Train N2N
    model.train()


    for epoch in tqdm(range(epochs), desc="Training ZSN2N"):
        loss = loss_func(noisy_img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % 100 == 0:  
            psnr = test(model, noisy_img, clean_img)
            tqdm.write(f"Epoch {epoch}, PSNR: {psnr:.2f}")
            tqdm.desc = f"Training Progress - Epoch {epoch}, PSNR: {psnr:.2f}"


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
