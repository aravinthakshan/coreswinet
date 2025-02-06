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

#         # Contrastive heads for each encoder level
#         self.contrastive = contrastive
#         if contrastive:
#             self.contrastive_heads1 = nn.ModuleList([
#                 nn.Sequential(
#                     nn.AdaptiveAvgPool2d(1),
#                     nn.Flatten(),
#                     nn.Linear(in_features=ch, out_features=512),
#                     nn.BatchNorm1d(512),
#                     nn.ReLU(),
#                     nn.Linear(in_features=512, out_features=64),
#                     nn.BatchNorm1d(64),
#                 ) for ch in encoder_channels
#             ])
            
#             self.contrastive_heads2 = nn.ModuleList([
#                 nn.Sequential(
#                     nn.AdaptiveAvgPool2d(1),
#                     nn.Flatten(),
#                     nn.Linear(in_features=ch, out_features=512),
#                     nn.BatchNorm1d(512),
#                     nn.ReLU(),
#                     nn.Linear(in_features=512, out_features=64),
#                     nn.BatchNorm1d(64),
#                 ) for ch in encoder_channels
#             ])

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

#     def forward(self, x_noisy, x_n2n):
#         """
#         Forward pass of the model
#         Args:
#             x_noisy (torch.Tensor): Noisy input image
#             x_n2n (torch.Tensor): N2N denoised version of the input image
#         Returns:
#             tuple: (output, contrastive_features) where contrastive_features is a list of 
#                   (f1, f2) pairs for each encoder level if contrastive=True
#         """
#         # Get features from first encoder
#         features1 = list(self.encoder1(x_noisy))
        
#         # Get features from second encoder only if not bypassing
#         if not self.bypass:
#             features2 = list(self.encoder2(x_n2n))
#         else:
#             features2 = features1  #

#         # Process each encoder level and compute contrastive features
#         processed_features = []
#         contrastive_features = []
        
#         for i in range(len(features1)):
#             # Process features through Swin block
#             processed_feat = self.process_features(
#                 features1[i], 
#                 features2[i], 
#                 self.swin_blocks[i]
#             )
#             processed_features.append(processed_feat)
            
#             # Compute contrastive features for this level
#             if self.contrastive:
#                 f1 = self.contrastive_heads1[i](features1[i])
#                 if self.bypass:
#                     f2 = self.contrastive_heads2[i](features1[i])
#                 else:
#                     f2 = self.contrastive_heads2[i](features2[i])
#                 contrastive_features.append((f1, f2))

#         # The last processed feature becomes the bottleneck
#         bottleneck = self.bottleneck_attention(processed_features[-1])

#         # Pass processed features into decoder as skip connections
#         decoder_features = processed_features[:-1]
#         decoder_output = self.decoder(*decoder_features, bottleneck)

#         output = self.final(decoder_output)

#         if self.contrastive:
#             return output, contrastive_features
#         return output

# class BlindSpotConv2d(nn.Conv2d):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#         self.register_buffer('mask', self.weight.data.clone())
#         _, _, kH, kW = self.weight.size()
#         self.mask.fill_(1)
#         self.mask[:, :, kH//2, kH//2] = 0

#     def forward(self, x):
#         self.weight.data *= self.mask
#         return super().forward(x)
    

# def replace_decoder_convs(model):
#     conv_count = 0
#     replaced_count = 0
#     decoder_convs = {}  # Track convs per decoder level
    
#     def _is_main_decoder(module_path):
#         # Only match the main decoder path, not unet1/unet2 decoders
#         parts = module_path.split('.')
#         return parts[0] == 'decoder'
    
#     def _get_decoder_level(path):
#         parts = path.split('.')
#         for i, part in enumerate(parts):
#             if part == 'blocks':
#                 # Get the level number that comes after 'blocks'
#                 if i + 1 < len(parts) and parts[i + 1].isdigit():
#                     return int(parts[i + 1])
#         return None
    
#     def _replace_conv_in_module(module, path=''):
#         nonlocal conv_count, replaced_count
        
#         for name, child in module.named_children():
#             current_path = f"{path}.{name}" if path else name
            
#             if isinstance(child, nn.Conv2d) and not isinstance(child, BlindSpotConv2d):
#                 conv_count += 1
#                 if _is_main_decoder(current_path):
#                     level = _get_decoder_level(current_path)
                    
#                     # Track convs per decoder level
#                     if level is not None:
#                         decoder_convs[level] = decoder_convs.get(level, 0) + 1
                    
#                     # Replace convs in odd-numbered decoder levels
#                     if level is not None and level % 2 == 1:
#                         new_conv = BlindSpotConv2d(
#                             in_channels=child.in_channels,
#                             out_channels=child.out_channels,
#                             kernel_size=child.kernel_size,
#                             stride=child.stride,
#                             padding=child.padding,
#                             dilation=child.dilation,
#                             groups=child.groups,
#                             bias=child.bias is not None,
#                             padding_mode=child.padding_mode
#                         )
                        
#                         # Initialize weights with original conv weights
#                         with torch.no_grad():
#                             new_conv.weight.data = child.weight.data.clone()
#                             if child.bias is not None:
#                                 new_conv.bias.data = child.bias.data.clone()
                        
#                         setattr(module, name, new_conv)
#                         replaced_count += 1
#                         print(f"Replaced Conv2d in decoder level {level}, path: {current_path}")
#                     else:
#                         print(f"Kept original Conv2d in decoder level {level}, path: {current_path}")
            
#             if len(list(child.children())) > 0:
#                 _replace_conv_in_module(child, current_path)
    
#     _replace_conv_in_module(model)
#     print(f"\nSummary:")
#     print(f"Total Conv2d layers: {conv_count}")
#     print(f"Decoder Conv2d layers replaced with BlindSpotConv2d: {replaced_count}")
#     print("\nConvolutions per decoder level:")
#     for level in sorted(decoder_convs.keys()):
#         print(f"Level {level}: {decoder_convs[level]} convolutions")
#         if level % 2 == 1:
#             print(f"  -> All replaced with BlindSpotConv2d")
#         else:
#             print(f"  -> Kept as regular Conv2d")
    
#     return model

# def main():
#     # Set device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
    
#     # Initialize model
#     model = Model(in_channels=3, contrastive=True).to(device)
#     model = replace_decoder_convs(model)
#     model.to(device)

#     # Create sample input tensors (assuming 256x256 input size based on model architecture)
#     batch_size = 2
#     x_noisy = torch.randn(batch_size, 3, 256, 256).to(device)
#     x_n2n = torch.randn(batch_size, 3, 256, 256).to(device)
    
#     # Test model in evaluation mode
#     model.eval()
#     with torch.no_grad():
#         try:
#             # Test with contrastive=True
#             output, contrastive_features = model(x_noisy, x_n2n)
            
#             # Print shapes
#             print("\nOutput shapes with contrastive=True:")
#             print(f"Main output shape: {output.shape}")
#             print("\nContrastive feature shapes:")
#             for i, (f1, f2) in enumerate(contrastive_features):
#                 print(f"Level {i}:")
#                 print(f"  f1 shape: {f1.shape}")
#                 print(f"  f2 shape: {f2.shape}")
            
#             # Test bypass mode
#             model.bypass = True
#             output_bypass, contrastive_features_bypass = model(x_noisy, x_n2n)
#             print("\nBypass mode output shape:", output_bypass.shape)
            
#             # Verify output ranges (should be between -1 and 1 due to Tanh)
#             print("\nOutput statistics:")
#             print(f"Min value: {output.min().item():.3f}")
#             print(f"Max value: {output.max().item():.3f}")
            
#             # Test model with contrastive=False
#             model_no_contrastive = Model(in_channels=3, contrastive=False).to(device)
#             output_no_contrastive = model_no_contrastive(x_noisy, x_n2n)
#             print("\nOutput shape with contrastive=False:", output_no_contrastive.shape)
            
#             # Additional checks
#             print("\nModel structure validation:")
#             print(f"Number of Swin blocks: {len(model.swin_blocks)}")
#             print(f"Number of contrastive heads: {len(model.contrastive_heads1)}")
#             print(f"Encoder channels: {model.encoder1.out_channels}")
            
#         except Exception as e:
#             print(f"\nError during model execution: {str(e)}")
#             raise e

# if __name__ == "__main__":
#     main()



import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class LightweightStudent(nn.Module):
    def __init__(self, in_channels=3, contrastive=True, bypass=False):
        super().__init__()
        self.bypass = bypass

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

        self.encoder1 = self.unet1.encoder
        self.encoder2 = self.unet2.encoder
        self.decoder = self.unet1.decoder
        
        # Get encoder channels for feature matching
        encoder_channels = self.encoder1.out_channels

        # Lightweight channel attention for bottleneck
        self.bottleneck_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(encoder_channels[-1], encoder_channels[-1] // 4, 1),
            nn.ReLU(),
            nn.Conv2d(encoder_channels[-1] // 4, encoder_channels[-1], 1),
            nn.Sigmoid()
        )

        # Contrastive heads only for final encoder features
        self.contrastive = contrastive
        if contrastive:
            self.contrastive_head1 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_features=encoder_channels[-1], out_features=256),  # Reduced from 512
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(in_features=256, out_features=64),
                nn.BatchNorm1d(64),
            )
            self.contrastive_head2 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_features=encoder_channels[-1], out_features=256),  # Reduced from 512
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(in_features=256, out_features=64),
                nn.BatchNorm1d(64),
            )

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
            tuple: (output, f1, f2) where f1, f2 are contrastive features if contrastive=True
        """
        # Get features from first encoder
        features1 = list(self.encoder1(x_noisy))
        
        # Get features from second encoder only if not bypassing
        if not self.bypass:
            features2 = list(self.encoder2(x_n2n))
        else:
            features2 = features1  # Use same features in bypass mode

        # Process features at each level
        processed_features = []
        for i in range(len(features1)):
            processed_feat = self.process_features(features1[i], features2[i])
            processed_features.append(processed_feat)

        # Apply attention to bottleneck
        bottleneck = processed_features[-1] * self.bottleneck_attention(processed_features[-1])

        # Decode
        decoder_features = processed_features[:-1]
        decoder_output = self.decoder(*decoder_features, bottleneck)

        # Final processing
        output = self.final(decoder_output)

        if self.contrastive:
            # Only compute contrastive features for final encoder level
            f1 = self.contrastive_head1(features1[-1])
            if self.bypass:
                f2 = self.contrastive_head2(features1[-1])
            else:
                f2 = self.contrastive_head2(features2[-1])
            return output, f1, f2
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
    output_normal, f1_normal, f2_normal = model_normal(x_noisy, x_n2n)
    print(f"\nNormal mode:")
    print(f"Output shape: {output_normal.shape}")
    print(f"Contrastive feature shapes: {f1_normal.shape}, {f2_normal.shape}")
    
    # Test bypass mode
    output_bypass, f1_bypass, f2_bypass = model_bypass(x_noisy, x_n2n)
    print(f"\nBypass mode:")
    print(f"Output shape: {output_bypass.shape}")
    print(f"Contrastive feature shapes: {f1_bypass.shape}, {f2_bypass.shape}")

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


# if __name__ == "__main__":
#     # Set the device
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     # Test both modes
#     model_normal = Model(in_channels=3).to(device)
#     model_bypass = Model(in_channels=3).to(device)
    
#     batch_size = 2
#     dummy_input = torch.randn(batch_size, 3, 256, 256).to(device)
#     dummy_n2n = torch.randn(batch_size, 3, 256, 256).to(device)

    # # Print model summary using torchsummary for two inputs
    # print("\nModel Summary (Normal Mode):")
    # summary(model_normal, input_size=[(3, 256, 256), (3, 256, 256)], device=device)

    # print("\nModel Summary (Bypass Mode):")
    # summary(model_bypass, input_size=[(3, 256, 256), (3, 256, 256)], device=device)

    # Test both modes
    # output_normal = model_normal(dummy_input, dummy_n2n)
    # output_bypass = model_bypass(dummy_input, dummy_n2n)

    # if isinstance(output_normal, tuple):
    #     print(f"Normal mode output shape: {output_normal[0].shape}")
    #     print(f"Normal mode contrastive feature shapes: {output_normal[1].shape}, {output_normal[2].shape}")
    
    # if isinstance(output_bypass, tuple):
    #     print(f"Bypass mode output shape: {output_bypass[0].shape}")
    #     print(f"Bypass mode contrastive feature shapes: {output_bypass[1].shape}, {output_bypass[2].shape}")