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

class Model(nn.Module):
    def __init__(self, in_channels=3, contrastive=True, bypass=False):
        super().__init__()
        self.bypass = bypass

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

        # Contrastive heads for each encoder level
        self.contrastive = contrastive
        if contrastive:
            self.contrastive_heads1 = nn.ModuleList([
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(in_features=ch, out_features=512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Linear(in_features=512, out_features=64),
                    nn.BatchNorm1d(64),
                ) for ch in encoder_channels
            ])
            
            self.contrastive_heads2 = nn.ModuleList([
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(in_features=ch, out_features=512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Linear(in_features=512, out_features=64),
                    nn.BatchNorm1d(64),
                ) for ch in encoder_channels
            ])

        # Final processing
        self.final = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, in_channels, kernel_size=1),
            nn.Tanh()
        )

    def process_features(self, feat1, feat2, swin_block):
        if self.bypass:
            # Skip element-wise max and directly process feat1 through Swin
            B, C, H, W = feat1.shape
            feat_reshaped = feat1.flatten(2).transpose(1, 2)
            swin_out = swin_block(feat_reshaped)
            return swin_out.transpose(1, 2).reshape(B, C, H, W)
        else:
            # Original processing with element-wise maximum
            max_feat = torch.maximum(feat1, feat2)
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
        Returns:
            tuple: (output, contrastive_features) where contrastive_features is a list of 
                  (f1, f2) pairs for each encoder level if contrastive=True
        """
        # Get features from first encoder
        features1 = list(self.encoder1(x_noisy))
        
        # Get features from second encoder only if not bypassing
        if not self.bypass:
            features2 = list(self.encoder2(x_n2n))
        else:
            features2 = features1  # Use same features when bypassing

        # Process each encoder level and compute contrastive features
        processed_features = []
        contrastive_features = []
        
        for i in range(len(features1)):
            # Process features through Swin block
            processed_feat = self.process_features(
                features1[i], 
                features2[i], 
                self.swin_blocks[i]
            )
            processed_features.append(processed_feat)
            
            # Compute contrastive features for this level
            if self.contrastive:
                f1 = self.contrastive_heads1[i](features1[i])
                if self.bypass:
                    f2 = self.contrastive_heads2[i](features1[i])
                else:
                    f2 = self.contrastive_heads2[i](features2[i])
                contrastive_features.append((f1, f2))

        # The last processed feature becomes the bottleneck
        bottleneck = self.bottleneck_attention(processed_features[-1])

        # Pass processed features into decoder as skip connections
        decoder_features = processed_features[:-1]
        decoder_output = self.decoder(*decoder_features, bottleneck)

        output = self.final(decoder_output)

        if self.contrastive:
            return output, contrastive_features
        return output
    
if __name__ == "__main__":
    import torch
    import segmentation_models_pytorch as smp
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Initialize model
    model = Model(in_channels=3, contrastive=True, bypass=False)
    model.eval()  # Set to evaluation mode
    
    # Create dummy input data (batch_size=2, channels=3, height=256, width=256)
    batch_size = 2
    x_noisy = torch.randn(batch_size, 3, 256, 256)
    x_n2n = torch.randn(batch_size, 3, 256, 256)
    
    # Forward pass
    with torch.no_grad():
        output, contrastive_features = model(x_noisy, x_n2n)
    
    # Print shapes of all outputs
    print("\n=== Output Shapes ===")
    print(f"Main output shape: {output.shape}")  # Should be [batch_size, 3, 256, 256]
    
    print("\n=== Contrastive Feature Shapes ===")
    for i, (f1, f2) in enumerate(contrastive_features):
        print(f"\nEncoder Level {i}:")
        print(f"f1 shape: {f1.shape}")  # Should be [batch_size, 64]
        print(f"f2 shape: {f2.shape}")  # Should be [batch_size, 64]
    
    # Print encoder channel dimensions for reference
    print("\n=== Encoder Channel Dimensions ===")
    encoder_channels = model.encoder1.out_channels
    for i, channels in enumerate(encoder_channels):
        print(f"Level {i}: {channels} channels")
        
    # Verify bypass mode
    print("\n=== Testing Bypass Mode ===")
    model_bypass = Model(in_channels=3, contrastive=True, bypass=True)
    model_bypass.eval()
    
    with torch.no_grad():
        output_bypass, contrastive_features_bypass = model_bypass(x_noisy, x_n2n)
    
    print("\nBypass Mode - Output Shapes:")
    print(f"Main output shape: {output_bypass.shape}")
    
    print("\nBypass Mode - Contrastive Feature Shapes:")
    for i, (f1, f2) in enumerate(contrastive_features_bypass):
        print(f"\nEncoder Level {i}:")
        print(f"f1 shape: {f1.shape}")
        print(f"f2 shape: {f2.shape}")
        
    # Verify memory efficiency
    def get_model_size(model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return (param_size + buffer_size) / 1024**2  # Size in MB
    
    print(f"\n=== Model Memory Usage ===")
    print(f"Model size: {get_model_size(model):.2f} MB")
    
    # Test contrastive=False mode
    model_no_contrastive = Model(in_channels=3, contrastive=False, bypass=False)
    model_no_contrastive.eval()
    
    with torch.no_grad():
        output_no_contrastive = model_no_contrastive(x_noisy, x_n2n)
    
    print("\n=== No Contrastive Mode ===")
    print(f"Output shape: {output_no_contrastive.shape}")