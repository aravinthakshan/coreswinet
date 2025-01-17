import torch.nn as nn
import torch

def discriminator_block(in_filters, out_filters, normalization=False):
    layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1)]
    layers.append(nn.LeakyReLU(0.2))
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
    return layers

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),  # Downsample to 64x64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),         # Downsample to 32x32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),       # Downsample to 16x16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, stride=1, padding=0)          # Final output
        )
        
    def forward(self, x):
        x = (x + 1) / 2  # Scale from [-1, 1] to [0, 1]
        return torch.sigmoid(self.model(x))  # Sigmoid for binary classification
    
    
    
class ConditionalDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(ConditionalDiscriminator, self).__init__()
        # Double the input channels to account for the condition
        self.model = nn.Sequential(
            nn.Conv2d(in_channels * 2, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 1, 4, stride=1, padding=0)
        )
        
    def forward(self, x, condition):
        x = torch.cat([x, condition], dim=1)
        return torch.sigmoid(self.model(x))