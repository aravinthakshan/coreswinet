import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

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
