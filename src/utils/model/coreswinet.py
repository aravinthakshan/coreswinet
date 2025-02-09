import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torchsummary import summary
from utils.model.archs.SwinBlocks import SwinTransformerBlock
from utils.model.archs.AttentionModules import SimpleChannelAttention, SqueezeExcitationBlock
from utils.model.archs.ZSN2N import N2NNetwork
# from archs.SwinBlocks import SwinTransformerBlock
# from archs.AttentionModules import SimpleChannelAttention, SqueezeExcitationBlock
    


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

        # Residual prediction layers (without Tanh)
        self.residual_layers = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, in_channels, kernel_size=1)
        )
        
        # Final Tanh activation as separate layer
        self.final_activation = nn.Tanh()

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

    def forward(self, x_noisy, enc2_in):
        """
        Forward pass of the model
        Args:
            x_noisy (torch.Tensor): Noisy input image
            enc2_in (torch.Tensor): N2N denoised version of the input image
        Returns:
            torch.Tensor or tuple: If contrastive is True, returns (denoised_image, f1, f2),
                                 otherwise returns denoised_image
        """
        # Get features from first encoder
        features1 = list(self.encoder1(x_noisy))
        
        # Get features from second encoder only if not bypassing
        if not self.bypass:
            features2 = list(self.encoder2(enc2_in))
        else:
            features2 = features1  # Dummy assignment, won't be used

        # Process each encoder level
        processed_features = []
        for i in range(len(features1)):
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

        # Get the residual prediction (before Tanh)
        residual = self.residual_layers(decoder_output)
        
        # Add the residual to the noisy input
        pre_activation = residual + x_noisy
        
        # Apply final Tanh activation
        output = self.final_activation(pre_activation)

        if self.contrastive:
            f1 = self.contrastive_head1(features1[-1])
            if self.bypass:
                f2 = self.contrastive_head2(features1[-1])  # Use features1 when bypassing
            else:
                f2 = self.contrastive_head2(features2[-1])
            return output, f1, f2
        return output
    
# ----------> BELOW IS OUR IMPORTANT MODEL <-----------
    
# class Model(nn.Module):
#     def __init__(self, in_channels=3, contrastive=True, bypass=False):
#         super().__init__()
#         self.bypass = bypass

#         # First encoder (for noisy input)
#         self.unet1 = smp.Unet(
#             encoder_name="resnet18",
#             encoder_weights="imagenet",
#             in_channels=in_channels,
#             classes=16,
#             decoder_channels=(512, 256, 128, 64, 64),
#         )

#         # Second encoder (for N2N denoised input)
#         self.unet2 = smp.Unet(
#             encoder_name="resnet18",
#             encoder_weights="imagenet",
#             in_channels=in_channels,
#             classes=16,
#             decoder_channels=(512, 256, 128, 64, 64),
#         )

#         self.encoder1 = self.unet1.encoder
#         self.encoder2 = self.unet2.encoder
#         self.decoder = self.unet1.decoder

#         encoder_channels = self.encoder1.out_channels

#         # Create Swin Transformer block for each encoder level
#         self.swin_blocks = nn.ModuleList([
#             SwinTransformerBlock(
#                 dim=ch,
#                 input_resolution=(256 // (2 ** i), 256 // (2 ** i)),
#                 num_heads=min(8, max(1, ch // 32)),
#                 window_size=min(7, max(3, ch // 32)),
#                 mlp_ratio=4.0
#             ) for i, ch in enumerate(encoder_channels)
#         ])

#         # Squeeze attention for bottleneck
#         self.bottleneck_attention = SqueezeExcitationBlock(encoder_channels[-1])

#         # Contrastive heads
#         self.contrastive = contrastive
#         if contrastive:
#             self.contrastive_head1 = nn.Sequential(
#                 nn.AdaptiveAvgPool2d(1),
#                 nn.Flatten(),
#                 nn.Linear(in_features=encoder_channels[-1], out_features=512),
#                 nn.BatchNorm1d(512),
#                 nn.ReLU(),
#                 nn.Linear(in_features=512, out_features=64),
#                 nn.BatchNorm1d(64),
#             )
#             self.contrastive_head2 = nn.Sequential(
#                 nn.AdaptiveAvgPool2d(1),
#                 nn.Flatten(),
#                 nn.Linear(in_features=encoder_channels[-1], out_features=512),
#                 nn.BatchNorm1d(512),
#                 nn.ReLU(),
#                 nn.Linear(in_features=512, out_features=64),
#                 nn.BatchNorm1d(64),
#             )

#         # Final processing
#         self.final = nn.Sequential(
#             nn.Conv2d(64, 16, kernel_size=1),
#             nn.BatchNorm2d(16),
#             nn.Conv2d(16, in_channels, kernel_size=1),
#             nn.Tanh()
#         )

#     def process_features(self, feat1, feat2, swin_block):
#         if self.bypass:
#             # Skip element-wise max and directly process feat1 through Swin
#             B, C, H, W = feat1.shape
#             feat_reshaped = feat1.flatten(2).transpose(1, 2)
#             swin_out = swin_block(feat_reshaped)
#             return swin_out.transpose(1, 2).reshape(B, C, H, W)
#         else:
#             # Original processing with element-wise maximum
#             max_feat = torch.maximum(feat1, feat2)
#             B, C, H, W = max_feat.shape
#             feat_reshaped = max_feat.flatten(2).transpose(1, 2)
#             swin_out = swin_block(feat_reshaped)
#             return swin_out.transpose(1, 2).reshape(B, C, H, W)

#     def forward(self, x_noisy, enc2_in):
#         """
#         Forward pass of the model
#         Args:
#             x_noisy (torch.Tensor): Noisy input image
#             enc2_in (torch.Tensor): N2N denoised version of the input image
#         """
#         # Get features from first encoder
#         features1 = list(self.encoder1(x_noisy))
        
#         # Get features from second encoder only if not bypassing
#         if not self.bypass:
#             features2 = list(self.encoder2(enc2_in))
#         else:
#             features2 = features1  # Dummy assignment, won't be used

#         # Process each encoder level
#         processed_features = []
#         for i in range(len(features1)):
#             processed_feat = self.process_features(
#                 features1[i], 
#                 features2[i], 
#                 self.swin_blocks[i]
#             )
#             processed_features.append(processed_feat)
            
#         # The last processed feature becomes the bottleneck
#         bottleneck = self.bottleneck_attention(processed_features[-1])

#         # Pass processed features into decoder as skip connections
#         decoder_features = processed_features[:-1]
#         decoder_output = self.decoder(*decoder_features, bottleneck)

#         output = self.final(decoder_output)

#         if self.contrastive:
#             f1 = self.contrastive_head1(features1[-1])
#             if self.bypass:
#                 f2 = self.contrastive_head2(features1[-1])  # Use features1 when bypassing
#             else:
#                 f2 = self.contrastive_head2(features2[-1])
#             return output, f1, f2
#         return output



# class Model(nn.Module):
#     def __init__(self, in_channels=3, contrastive=True, bypass=False):
#         super().__init__()
#         self.bypass = bypass

#         # First encoder (for noisy input) - Now using UNet++
#         self.unetpp1 = smp.UnetPlusPlus(
#             encoder_name="resnet18",
#             encoder_weights="imagenet",
#             in_channels=in_channels,
#             classes=16,
#             decoder_channels=(512, 256, 128, 64, 64),
#         )

#         # Second encoder (for N2N denoised input) - Now using UNet++
#         self.unetpp2 = smp.UnetPlusPlus(
#             encoder_name="resnet18",
#             encoder_weights="imagenet",
#             in_channels=in_channels,
#             classes=16,
#             decoder_channels=(512, 256, 128, 64, 64),
#         )

#         self.encoder1 = self.unetpp1.encoder
#         self.encoder2 = self.unetpp2.encoder
#         self.decoder = self.unetpp1.decoder

#         encoder_channels = self.encoder1.out_channels

#         # Contrastive heads (preserved from original)
#         self.contrastive = contrastive
#         if contrastive:
#             self.contrastive_head1 = nn.Sequential(
#                 nn.AdaptiveAvgPool2d(1),
#                 nn.Flatten(),
#                 nn.Linear(in_features=encoder_channels[-1], out_features=512),
#                 nn.BatchNorm1d(512),
#                 nn.ReLU(),
#                 nn.Linear(in_features=512, out_features=64),
#                 nn.BatchNorm1d(64),
#             )
#             self.contrastive_head2 = nn.Sequential(
#                 nn.AdaptiveAvgPool2d(1),
#                 nn.Flatten(),
#                 nn.Linear(in_features=encoder_channels[-1], out_features=512),
#                 nn.BatchNorm1d(512),
#                 nn.ReLU(),
#                 nn.Linear(in_features=512, out_features=64),
#                 nn.BatchNorm1d(64),
#             )

#         # Final processing (preserved from original)
#         self.final = nn.Sequential(
#             nn.Conv2d(64, 16, kernel_size=1),
#             nn.BatchNorm2d(16),
#             nn.Conv2d(16, in_channels, kernel_size=1),
#             nn.Tanh()
#         )

#     def process_features(self, feat1, feat2):
#         """
#         Simplified process_features without Swin blocks
#         """
#         if self.bypass:
#             return feat1
#         else:
#             return torch.maximum(feat1, feat2)

#     def forward(self, x_noisy, enc2_in):
#         """
#         Forward pass of the model
#         Args:
#             x_noisy (torch.Tensor): Noisy input image
#             enc2_in (torch.Tensor): N2N denoised version of the input image
#         """
#         # Get features from first encoder
#         features1 = list(self.encoder1(x_noisy))
        
#         # Get features from second encoder only if not bypassing
#         if not self.bypass:
#             features2 = list(self.encoder2(enc2_in))
#         else:
#             features2 = features1  # Dummy assignment, won't be used

#         # Process each encoder level (simplified without Swin blocks)
#         processed_features = []
#         for i in range(len(features1)):
#             processed_feat = self.process_features(features1[i], features2[i])
#             processed_features.append(processed_feat)
            
#         # Use the last processed feature directly as bottleneck (no attention)
#         bottleneck = processed_features[-1]

#         # Pass processed features into decoder as skip connections
#         decoder_features = processed_features[:-1]
#         decoder_output = self.decoder(*decoder_features, bottleneck)

#         output = self.final(decoder_output)

#         if self.contrastive:
#             f1 = self.contrastive_head1(features1[-1])
#             if self.bypass:
#                 f2 = self.contrastive_head2(features1[-1])
#             else:
#                 f2 = self.contrastive_head2(features2[-1])
#             return output, f1, f2
#         return output


if __name__ == "__main__":
    # Set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test both modes
    model_normal = Model(in_channels=3).to(device)
    model_bypass = Model(in_channels=3).to(device)
    
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 256, 256).to(device)
    dummy_n2n = torch.randn(batch_size, 3, 256, 256).to(device)

    # Print model summary using torchsummary for two inputs
    print("\nModel Summary (Normal Mode):")
    summary(model_normal, input_size=[(3, 256, 256), (3, 256, 256)], device=device)

    print("\nModel Summary (Bypass Mode):")
    summary(model_bypass, input_size=[(3, 256, 256), (3, 256, 256)], device=device)

    output_normal = model_normal(dummy_input, dummy_n2n)
    output_bypass = model_bypass(dummy_input, dummy_n2n)

    if isinstance(output_normal, tuple):
        print(f"Normal mode output shape: {output_normal[0].shape}")
        print(f"Normal mode contrastive feature shapes: {output_normal[1].shape}, {output_normal[2].shape}")
    
    if isinstance(output_bypass, tuple):
        print(f"Bypass mode output shape: {output_bypass[0].shape}")
        print(f"Bypass mode contrastive feature shapes: {output_bypass[1].shape}, {output_bypass[2].shape}")