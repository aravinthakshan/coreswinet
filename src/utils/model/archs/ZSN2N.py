import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torchvision.transforms import functional as TF
import numpy as np
import numpy as np


def test(model, noisy_img, clean_img, save_dir="predictions/"):
    with torch.no_grad():
        pred = torch.clamp(noisy_img - model(noisy_img), 0, 1)
        mse = F.mse_loss(clean_img, pred).item()
        psnr = 10 * np.log10(1 / mse)

    # min_value = pred.min().item()
    # max_value = pred.max().item()
    # print(f"Range of values in pred_img: min={min_value}, max={max_value}")

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

    return model, psnr 

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


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

def train_one_shot(
    epochs: int,
    model: nn.Module,
    dataloader,
    device: str = 'cuda',
    base_lr: float = 0.0001,
    patience: int = 500,
) -> nn.Module:
    """
    Train model on a single image using a one-shot approach with multi-scale consistency
    and perceptual losses.
    """
    model = model.to(device)
    
    # Get single image pair from dataset
    noisy_img, clean_img = next(iter(dataloader))
    noisy_img, clean_img = noisy_img.to(device), clean_img.to(device)
    
    # Setup optimizer with cosine annealing
    optimizer = optim.AdamW(model.parameters(), lr=base_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=epochs//10, 
        T_mult=2
    )

    def create_image_pyramid(img, scales=[1.0, 0.5, 0.25]):
        """Create multi-scale image pyramid."""
        pyramid = []
        for scale in scales:
            if scale != 1.0:
                size = (int(img.shape[2] * scale), int(img.shape[3] * scale))
                scaled = F.interpolate(img, size=size, mode='bilinear', align_corners=False)
            else:
                scaled = img
            pyramid.append(scaled)
        return pyramid

    def consistency_loss(pred1, pred2):
        """Compute consistency loss between different scale predictions."""
        # Resize larger prediction to match smaller one
        if pred1.shape != pred2.shape:
            pred1 = F.interpolate(pred1, size=pred2.shape[2:], mode='bilinear', align_corners=False)
        return F.mse_loss(pred1, pred2)

    def compute_total_variation_loss(x):
        """Compute total variation loss to encourage spatial smoothness."""
        tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
        tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
        return (tv_h + tv_w) / 2.0

    def loss_func(noisy_img, model):
        # Create image pyramids
        noisy_pyramid = create_image_pyramid(noisy_img)
        
        # Get predictions at different scales
        predictions = []
        for scale_img in noisy_pyramid:
            pred = model(scale_img)
            predictions.append(pred)
        
        # Reconstruction loss at original scale
        recon_loss = F.mse_loss(predictions[0], clean_img)
        
        # Multi-scale consistency loss
        cons_loss = 0
        for i in range(len(predictions)-1):
            cons_loss += consistency_loss(predictions[i], predictions[i+1])
        cons_loss /= (len(predictions)-1)
        
        # Total variation loss for spatial smoothness
        tv_loss = compute_total_variation_loss(predictions[0])
        
        # Combine losses
        total_loss = recon_loss + 0.1 * cons_loss + 0.01 * tv_loss
        
        return total_loss, {
            'recon_loss': recon_loss.item(),
            'cons_loss': cons_loss.item(),
            'tv_loss': tv_loss.item()
        }

    def compute_psnr(pred, target):
        mse = F.mse_loss(pred, target).item()
        return 10 * torch.log10(torch.tensor(1.0) / mse)

    # Training loop
    model.train()
    best_psnr = 0
    best_state = None
    patience_counter = 0
    
    for epoch in tqdm(range(epochs), desc="Training One-Shot Model"):
        # Forward pass and loss computation
        loss, loss_components = loss_func(noisy_img, model)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Evaluation and logging
        if epoch % 100 == 0:
            with torch.no_grad():
                pred = model(noisy_img)
                current_psnr = compute_psnr(pred, clean_img)
                
                if current_psnr > best_psnr:
                    best_psnr = current_psnr
                    best_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                tqdm.write(
                    f"Epoch {epoch}, PSNR: {current_psnr:.2f}, "
                    f"Recon Loss: {loss_components['recon_loss']:.4f}, "
                    f"Cons Loss: {loss_components['cons_loss']:.4f}, "
                    f"TV Loss: {loss_components['tv_loss']:.4f}"
                )
        
        # Early stopping check
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs")
            break
    
    # Load best model state
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, best_psnr