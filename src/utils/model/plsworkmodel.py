import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torchsummary import summary
# from utils.model.archs.SwinBlocks import SwinTransformerBlock
# from utils.model.archs.AttentionModules import SimpleChannelAttention, SqueezeExcitationBlock
# from utils.model.archs.ZSN2N import N2NNetwork
from archs.SwinBlocks import SwinTransformerBlock
from archs.AttentionModules import SimpleChannelAttention, SqueezeExcitationBlock

class PReLUBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
        )
    
    def forward(self, x):
        return self.block(x)

class Model(nn.Module):
    def __init__(self, in_channels=3, contrastive=True):
        super().__init__()

        # First encoder (for noisy input)
        self.unet1 = smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=16,
            decoder_channels=(512, 256, 128, 64, 64),
        )

        # Second encoder (for N2N denoised input)
        self.unet2 = smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=16,
            decoder_channels=(512, 256, 128, 64, 64),
        )

        self.encoder1 = self.unet1.encoder
        self.encoder2 = self.unet2.encoder
        self.decoder = self.unet1.decoder

        encoder_channels = self.encoder1.out_channels

        # Create Swin Transformer block for each encoder level
        self.swin_blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=ch,
                input_resolution=(256 // (2 ** i), 256 // (2 ** i)),
                num_heads=min(8, max(1, ch // 32)),
                window_size=min(7, max(3, ch // 32)),
                mlp_ratio=4.0
            ) for i, ch in enumerate(encoder_channels)
        ])

        # Squeeze attention for bottleneck
        self.bottleneck_attention = SqueezeExcitationBlock(encoder_channels[-1])

        # Contrastive heads
        self.contrastive = contrastive
        if contrastive:
            self.contrastive_head1 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_features=encoder_channels[-1], out_features=512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(in_features=512, out_features=64),
                nn.BatchNorm1d(64),
            )
            self.contrastive_head2 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_features=encoder_channels[-1], out_features=512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(in_features=512, out_features=64),
                nn.BatchNorm1d(64),
            )

        # Final processing
        self.final = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, in_channels, kernel_size=1),
            nn.Tanh()
        )

    def process_features(self, feat1, feat2, swin_block):
        # Element-wise maximum of the features from both encoders
        max_feat = torch.maximum(feat1, feat2)

        # Process through the Swin Transformer block
        B, C, H, W = max_feat.shape
        feat_reshaped = max_feat.flatten(2).transpose(1, 2)
        swin_out = swin_block(feat_reshaped)
        return swin_out.transpose(1, 2).reshape(B, C, H, W)

    def forward(self, x_noisy, x_n2n):
        """
        Forward pass of the model
        Args:
            x_noisy (torch.Tensor): Noisy input image
            x_n2n (torch.Tensor): N2N denoised version of the input image
        """
        # Get features from both encoders
        features1 = list(self.encoder1(x_noisy))
        features2 = list(self.encoder2(x_n2n))

        # Process each encoder level
        processed_features = []
        for i in range(len(features1)):
            # Process through the Swin Transformer block after element-wise maximum
            processed_feat = self.process_features(
                features1[i], 
                features2[i], 
                self.swin_blocks[i]
            )
            processed_features.append(processed_feat)

        # The last processed feature becomes the bottleneck
        bottleneck = self.bottleneck_attention(processed_features[-1])

        # Pass processed features into decoder as skip connections
        decoder_features = processed_features[:-1]
        decoder_output = self.decoder(*decoder_features, bottleneck)

        output = self.final(decoder_output)

        if self.contrastive:
            f1 = self.contrastive_head1(features1[-1])
            f2 = self.contrastive_head2(features2[-1])
            return output, f1, f2
        return output

if __name__ == "__main__":
    model = Model(in_channels=3)
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 256, 256)
    dummy_n2n = torch.randn(batch_size, 3, 256, 256)  # Simulated N2N output
    output = model(dummy_input, dummy_n2n)

    if isinstance(output, tuple):
        print(f"Output shape: {output[0].shape}")
        print(f"Contrastive feature shapes: {output[1].shape}, {output[2].shape}")
    else:
        print(f"Output shape: {output.shape}")
