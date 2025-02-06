import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from archs.AttentionModules import SimpleChannelAttention, SqueezeExcitationBlock

class LightweightStudent(nn.Module):
    def __init__(self, in_channels=3, contrastive=True, bypass=False):
        super().__init__()
        self.bypass = bypass
        self.contrastive = contrastive

        # First encoder (for noisy input)
        self.unet1 = smp.Unet(
            encoder_name="timm-mobilenetv3_small_075",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=16,
            decoder_channels=(128, 64, 32, 16, 8),
        )

        # Second encoder (for N2N denoised input)
        self.unet2 = smp.Unet(
            encoder_name="timm-mobilenetv3_small_075",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=16,
            decoder_channels=(128, 64, 32, 16, 8),
        )

        # Extract encoders and decoder
        self.encoder1 = self.unet1.encoder
        self.encoder2 = self.unet2.encoder
        self.decoder = self.unet1.decoder
        
        # Get encoder channels for feature matching
        encoder_channels = self.encoder1.out_channels

        # Lightweight channel attention for bottleneck
        self.bottleneck_attention = SqueezeExcitationBlock(encoder_channels[-1])


        # Contrastive heads for each encoder level
        if contrastive:
            self.contrastive_heads1 = nn.ModuleList([
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(in_features=ch, out_features=ch//2),
                    nn.BatchNorm1d(ch//2),
                    nn.ReLU(),
                    nn.Linear(in_features=ch//2, out_features=32),
                    nn.BatchNorm1d(32),
                ) for ch in encoder_channels
            ])
            
            self.contrastive_heads2 = nn.ModuleList([
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(in_features=ch, out_features=ch//2),
                    nn.BatchNorm1d(ch//2),
                    nn.ReLU(),
                    nn.Linear(in_features=ch//2, out_features=32),
                    nn.BatchNorm1d(32),
                ) for ch in encoder_channels
            ])

        # Final processing
        self.final = nn.Sequential(
            nn.Conv2d(8, 4, kernel_size=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, in_channels, kernel_size=1),
            nn.Tanh()
        )

    def process_features(self, feat1, feat2):
        """Process features from both encoders based on bypass mode"""
        if self.bypass:
            return feat1
        else:
            return torch.maximum(feat1, feat2)

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
            features2 = features1  # Use same features in bypass mode

        # Process each encoder level and compute contrastive features
        processed_features = []
        contrastive_features = []
        
        for i in range(len(features1)):
            # Process features
            processed_feat = self.process_features(features1[i], features2[i])
            processed_features.append(processed_feat)
            
            # Compute contrastive features for this level if enabled
            if self.contrastive:
                f1 = self.contrastive_heads1[i](features1[i])
                if self.bypass:
                    f2 = self.contrastive_heads2[i](features1[i])
                else:
                    f2 = self.contrastive_heads2[i](features2[i])
                contrastive_features.append((f1, f2))

        # Apply attention to bottleneck
        bottleneck = self.bottleneck_attention(processed_features[-1])

        # Decode
        decoder_features = processed_features[:-1]
        decoder_output = self.decoder(*decoder_features, bottleneck)

        # Final processing
        output = self.final(decoder_output)

        if self.contrastive:
            return output, contrastive_features
        return output

def count_parameters(model):
    """Helper function to count model parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Test both normal and bypass modes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create models
    model_normal = LightweightStudent(in_channels=3, contrastive=True, bypass=False).to(device)
    model_bypass = LightweightStudent(in_channels=3, contrastive=True, bypass=True).to(device)
    
    # Print parameter counts
    num_params = count_parameters(model_normal)
    print(f"\nTotal trainable parameters: {num_params:,}")
    
    # Test forward pass
    batch_size = 2
    x_noisy = torch.randn(batch_size, 3, 256, 256).to(device)
    x_n2n = torch.randn(batch_size, 3, 256, 256).to(device)
    
    # Test normal mode
    output_normal, contrastive_features_normal = model_normal(x_noisy, x_n2n)
    print(f"\nNormal mode:")
    print(f"Output shape: {output_normal.shape}")
    print("Contrastive feature shapes:")
    for i, (f1, f2) in enumerate(contrastive_features_normal):
        print(f"Level {i}: {f1.shape}, {f2.shape}")
    
    # Test bypass mode
    output_bypass, contrastive_features_bypass = model_bypass(x_noisy, x_n2n)
    print(f"\nBypass mode:")
    print(f"Output shape: {output_bypass.shape}")
    print("Contrastive feature shapes:")
    for i, (f1, f2) in enumerate(contrastive_features_bypass):
        print(f"Level {i}: {f1.shape}, {f2.shape}")