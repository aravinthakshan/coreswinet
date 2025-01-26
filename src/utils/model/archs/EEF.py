from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

class DHA(nn.Module):
    def __init__(self, channel_num, reduction=16):
        super().__init__()
        self.L0 = nn.Linear(channel_num, channel_num)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.Hash = nn.Sequential(
            nn.Conv2d(channel_num, max(channel_num // reduction, 1), 1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(max(channel_num // reduction, 1), channel_num, 1, padding=0, bias=True),
            nn.Tanh()
        )
        
    def forward(self, x):
        y = self.avg_pool(x)
        threshold = self.L0(y.squeeze(-1).transpose(2,1)).transpose(2,1).unsqueeze(-1)
        w = torch.abs(self.Hash(y))
        zero = torch.zeros_like(w)
        one = torch.ones_like(w)
        y = torch.where(w > threshold, one, zero)
        return x * y

class DF(nn.Module):
    def __init__(self, channel_num):
        super().__init__()
        # Ensure channel divisions result in at least 1 channel
        div_channels = max(channel_num//3, 1)
        
        self.C0 = nn.Sequential(
            nn.Conv2d(channel_num, div_channels, groups=1, 
                     kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True))
             
        self.C1 = nn.Sequential(
            nn.Conv2d(channel_num, div_channels, groups=1,
                     kernel_size=3, stride=1, padding=2, dilation=2),
            nn.LeakyReLU(inplace=True))    

        self.C2 = nn.Sequential(
            nn.Conv2d(channel_num, div_channels, groups=1,
                     kernel_size=3, stride=1, padding=3, dilation=3),
            nn.LeakyReLU(inplace=True))

        self.R = nn.GELU()

    def forward(self, x):
        l = self.R(self.C2(x))
        m = self.R(self.C1(x) - l)
        h = self.R(self.C0(x) - self.C1(x))
        return l, m, h

class EFF(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU())
        self.DF = DF(dim)
        self.channel_mixer = nn.ModuleDict()  # Create on first use
        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, groups=dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True))
        self.linear2 = nn.Sequential(nn.Linear(dim, out_dim))

    def forward(self, x, H, W):
        bs, hw, c = x.size()
        x = self.linear1(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        sht = x
        l, m, h = self.DF(x)
        # Ensure channel dimensions match before concatenation and convolution
        x = torch.cat((l, m, h), dim=1)
        x = nn.Conv2d(x.size(1), sht.size(1), 1)(x)  # Adjust channels to match shortcut
        x = self.dwconv(x)
        x = x + sht
        x = rearrange(x, 'b c h w -> b (h w) c', h=H, w=W)
        x = self.linear2(x)
        return x

class AFEBlock(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.LN1 = nn.LayerNorm(dim)
        self.attn = DHA(dim)
        self.LN2 = nn.LayerNorm(dim)
        self.EFF = EFF(dim, out_dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        shortcut = x
        x = self.LN1(x)
        xx = self.attn(x.contiguous().view(B, C, H, W))
        x = xx.contiguous().view(B, H * W, C) + shortcut
        x = self.EFF(self.LN2(x), H, W)
        return x
