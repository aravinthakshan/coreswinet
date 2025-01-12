import torch.nn as nn
import torch 

class SimpleChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SimpleChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=(2, 3), keepdim=True)
        max_out = torch.amax(x, dim=(2, 3), keepdim=True)
        attention = self.fc1(avg_out) + self.fc1(max_out)
        attention = self.relu(attention)
        attention = self.sigmoid(self.fc2(attention))
        return x * attention

class SqueezeExcitationBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        reduced_channels = max(channel // reduction, 1)
        
        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Squeeze and Excitation layers
        self.fc = nn.Sequential(
            # Squeeze: dimensionality reduction
            nn.Linear(channel, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            # Excitation: dimensionality restoration with sigmoid
            nn.Linear(reduced_channels, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Get batch size and channel dimensions
        b, c, _, _ = x.size()
        
        # Global average pooling
        y = self.avg_pool(x).view(b, c)
        
        # Learn channel-wise importance
        y = self.fc(y).view(b, c, 1, 1)
        
        # Scale input features
        return x * y.expand_as(x)
    