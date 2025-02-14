import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torchsummary import summary
# from utils.model.archs.SwinBlocks import SwinTransformerBlock
# from utils.model.archs.AttentionModules import SimpleChannelAttention, SqueezeExcitationBlock
# from archs.SwinBlocks import SwinTransformerBlock
# from archs.AttentionModules import SimpleChannelAttention, SqueezeExcitationBlock
from typing import Optional, Union, List
from segmentation_models_pytorch.base.modules import Activation
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder


# def initialize_decoder(module):
#     for m in module.modules():

#         if isinstance(m, nn.Conv2d):
#             nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)

#         elif isinstance(m, nn.BatchNorm2d):
#             nn.init.constant_(m.weight, 1)
#             nn.init.constant_(m.bias, 0)

#         elif isinstance(m, nn.Linear):
#             nn.init.xavier_uniform_(m.weight)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)

# def initialize_head(module):
#     for m in module.modules():
#         if isinstance(m, (nn.Linear, nn.Conv2d)):
#             nn.init.xavier_uniform_(m.weight)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
                
# class SegmentationHead(nn.Sequential):
#     def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
#         conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
#         upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
#         activation = Activation(activation)
#         super().__init__(conv2d, upsampling, activation)
                         
# class SegmentationModel(torch.nn.Module):
#     def initialize(self):
#         initialize_decoder(self.decoder)
#         initialize_head(self.segmentation_head)

#     def check_input_shape(self, x):

#         h, w = x.shape[-2:]
#         output_stride = self.encoder.output_stride
#         if h % output_stride != 0 or w % output_stride != 0:
#             new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
#             new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
#             raise RuntimeError(
#                 f"Wrong input shape height={h}, width={w}. Expected image height and width "
#                 f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
#             )

#     def forward(self, x,y=None):
#         """Sequentially pass `x` trough model`s encoder, decoder and heads"""

#         self.check_input_shape(x)

#         features = self.encoder(x)
#         if self.fusion == True:
#             features1 = self.encoder2(y)
            
#             f1 = features[-1]
#             f2 = features1[-1]
            
#             for ind in range(len(features)):
#                 # features[ind] = (features[ind]+features1[ind])/2
#                 # features[ind] = features1[ind]
#                 features[ind] = torch.maximum(features[ind],features1[ind])
#                 # features[ind] = torch.cat((features[ind],features1[ind]),1)
    
#         decoder_output = self.decoder(*features)

#         masks = self.segmentation_head(decoder_output)

#         if self.contrastive_head1 is not None:
#             f1= self.contrastive_head1(f1)
#             f2= self.contrastive_head2(f2)
#             return masks, f1,  f2
#         return masks

#     @torch.no_grad()
#     def predict(self, x, y=None):
#         if self.training:
#             self.eval()
#         if self.contrastive_head1 is not None:
#             x, _, _ = self.forward(x,y)
#             return x
#         if y is not None:
#             x = self.forward(x,y)
#             return x
#         x = self.forward(x)

#         return x
    
# class Unet(SegmentationModel):
#     def __init__(
#         self,
#         encoder_name: str = "resnet34",
#         encoder_depth: int = 5,
#         encoder_weights: Optional[str] = "imagenet",
#         fusion: bool = True,
#         bypass: bool = False,  # Added bypass parameter
#         decoder_use_batchnorm: bool = True,
#         decoder_channels: List[int] = (256, 128, 64, 32, 16),
#         decoder_attention_type: Optional[str] = None,
#         in_channels: int = 3,
#         classes: int = 1,
#         activation: Optional[Union[str, callable]] = None,
#         contrastive: bool = False,
#     ):
#         super().__init__()
#                 # Initialize contrastive heads as None by default
#         self.contrastive_head1 = None
#         self.contrastive_head2 = None
#         self.fusion = fusion
#         self.bypass = bypass  # Store bypass parameter
#         self.encoder = get_encoder(
#             encoder_name,
#             in_channels=in_channels,
#             depth=encoder_depth,
#             weights=encoder_weights,
#         )
#         self.encoder2 = get_encoder(
#             encoder_name,
#             in_channels=in_channels,
#             depth=encoder_depth,
#             weights=encoder_weights,
#         )

#         self.decoder = UnetDecoder(
#             encoder_channels=(self.encoder.out_channels),
#             # encoder_channels=tuple([2*item for item in self.encoder.out_channels]),
#             decoder_channels=decoder_channels,
#             # decoder_channels=tuple([2*item for item in decoder_channels]),
#             n_blocks=encoder_depth,
#             use_batchnorm=decoder_use_batchnorm,
#             center=True if encoder_name.startswith("vgg") else False,
#             attention_type=decoder_attention_type,
#         )

#         self.segmentation_head = SegmentationHead(
#             in_channels=decoder_channels[-1],
#             out_channels=classes,
#             activation=activation,
#             kernel_size=3,
#         )

#         if contrastive:
#             self.contrastive_head1 = nn.Sequential(
#                 nn.AdaptiveAvgPool2d(1),
#                 nn.Flatten(),
#                 nn.Linear(in_features=self.encoder.out_channels[-1], out_features=512),
#                 nn.BatchNorm1d(512),
#                 nn.ReLU(),
#                 nn.Linear(in_features=512, out_features=64),
#                 nn.BatchNorm1d(64),
#             )
#             self.contrastive_head2 = nn.Sequential(
#                 nn.AdaptiveAvgPool2d(1),
#                 nn.Flatten(),
#                 nn.Linear(in_features=self.encoder.out_channels[-1], out_features=512),
#                 nn.BatchNorm1d(512),
#                 nn.ReLU(),
#                 nn.Linear(in_features=512, out_features=64),
#                 nn.BatchNorm1d(64),
#             )

#         self.name = "u-{}".format(encoder_name)
#         self.initialize()

#     def forward(self, x, y=None):
#         """Sequentially pass `x` through model's encoder, decoder and heads"""
#         self.check_input_shape(x)

#         features1 = self.encoder(x)
        
#         if self.fusion == True:
#             if self.bypass:
#                 # When bypassing, use features1 for both paths
#                 features2 = features1
#                 f1 = features1[-1]
#                 f2 = features1[-1]
#             else:
#                 # Normal fusion behavior
#                 features2 = self.encoder2(y)
#                 f1 = features1[-1]
#                 f2 = features2[-1]
            
#             for ind in range(len(features1)):
#                 # features1[ind] = (features1[ind]+features2[ind])/2
#                 # features1[ind] = features2[ind]
#                 features1[ind] = torch.maximum(features1[ind], features2[ind])
#                 # features1[ind] = torch.cat((features1[ind],features2[ind]),1)

#         decoder_output = self.decoder(*features1)
#         masks = self.segmentation_head(decoder_output)

#         if self.contrastive_head1 is not None:
#             f1 = self.contrastive_head1(f1)
#             f2 = self.contrastive_head2(f2)
#             return masks, f1, f2
#         return masks
    
from typing import Optional, Union, List
import torch
import torch.nn as nn
from segmentation_models_pytorch.base.modules import Activation
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder

def initialize_decoder(module):
    for m in module.modules():

        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def initialize_head(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)
                         
class SegmentationModel(torch.nn.Module):
    def initialize(self):
        initialize_decoder(self.decoder)
        initialize_head(self.segmentation_head)

    def check_input_shape(self, x):

        h, w = x.shape[-2:]
        output_stride = self.encoder.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
            new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )

    def forward(self, x,y=None):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)

        features = self.encoder(x)
        if self.fusion == True:
            features1 = self.encoder2(y)
            
            f1 = features[-1]
            f2 = features1[-1]
            
            for ind in range(len(features)):
                # features[ind] = (features[ind]+features1[ind])/2
                # features[ind] = features1[ind]
                features[ind] = torch.maximum(features[ind],features1[ind])
                # features[ind] = torch.cat((features[ind],features1[ind]),1)
    
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.contrastive_head1 is not None:
            f1= self.contrastive_head1(f1)
            f2= self.contrastive_head2(f2)
            return masks, f1,  f2
        return masks

    @torch.no_grad()
    def predict(self, x, y=None):
        if self.training:
            self.eval()
        if self.contrastive_head1 is not None:
            x, _, _ = self.forward(x,y)
            return x
        if y is not None:
            x = self.forward(x,y)
            return x
        x = self.forward(x)

        return x
                         
class Unet(SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        fusion:bool=True,
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        contrastive: bool = False,
    ):
        super().__init__()
        self.fusion=fusion
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        self.encoder2 = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetDecoder(
            encoder_channels=(self.encoder.out_channels),
            # encoder_channels=tuple([2*item for item in self.encoder.out_channels]),
            decoder_channels=decoder_channels,
            # decoder_channels=tuple([2*item for item in decoder_channels]),
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if contrastive:
            self.contrastive_head1= nn.Sequential(
                                       nn.AdaptiveAvgPool2d(1),
                                       nn.Flatten(),
                                       nn.Linear(in_features=self.encoder.out_channels[-1], out_features=512),
                                       nn.BatchNorm1d(512),
                                       nn.ReLU(),
                                       nn.Linear(in_features=512, out_features=64),
                                       nn.BatchNorm1d(64),
                                   )
            self.contrastive_head2= nn.Sequential(
                                       nn.AdaptiveAvgPool2d(1),
                                       nn.Flatten(),
                                       nn.Linear(in_features=self.encoder.out_channels[-1], out_features=512),
                                       nn.BatchNorm1d(512),
                                       nn.ReLU(),
                                       nn.Linear(in_features=512, out_features=64),
                                       nn.BatchNorm1d(64),
                                   )

        self.name = "u-{}".format(encoder_name)
        self.initialize()
# ----------> BELOW IS THE IMPORTANT MODEL <-----------
    
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

#     # Print model summary using torchsummary for two inputs
#     # print("\nModel Summary (Normal Mode):")
#     # summary(model_normal, input_size=[(3, 256, 256), (3, 256, 256)], device=device)

#     # print("\nModel Summary (Bypass Mode):")
#     # summary(model_bypass, input_size=[(3, 256, 256), (3, 256, 256)], device=device)

#     output_normal = model_normal(dummy_input, dummy_n2n)
#     output_bypass = model_bypass(dummy_input, dummy_n2n)

#     if isinstance(output_normal, tuple):
#         print(f"Normal mode output shape: {output_normal[0].shape}")
#         print(f"Normal mode contrastive feature shapes: {output_normal[1].shape}, {output_normal[2].shape}")
    
#     if isinstance(output_bypass, tuple):
#         print(f"Bypass mode output shape: {output_bypass[0].shape}")
#         print(f"Bypass mode contrastive feature shapes: {output_bypass[1].shape}, {output_bypass[2].shape}")


# def test_model():
#     # Test device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # Initialize model with different configurations
#     # 1. Basic model without fusion or contrastive
#     model_basic = Unet(
#         encoder_name="resnet18",
#         fusion=False,
#         bypass=False,
#         contrastive=False
#     ).to(device)
    
#     # 2. Model with fusion
#     model_fusion = Unet(
#         encoder_name="resnet18",
#         fusion=True,
#         bypass=False,
#         contrastive=False
#     ).to(device)
    
#     # 3. Model with fusion and bypass
#     model_bypass = Unet(
#         encoder_name="resnet18",
#         fusion=True,
#         bypass=True,
#         contrastive=False
#     ).to(device)
    
#     # 4. Model with contrastive learning
#     model_contrastive = Unet(
#         encoder_name="resnet18",
#         fusion=True,
#         bypass=False,
#         contrastive=True
#     ).to(device)
    
#     # Create dummy inputs (adjust size according to your needs)
#     batch_size = 2
#     channels = 3
#     height = 256
#     width = 256
    
#     x = torch.randn(batch_size, channels, height, width).to(device)
#     y = torch.randn(batch_size, channels, height, width).to(device)
    
#     print("\nTesting different model configurations:")
    
#     # Test basic model
#     print("\n1. Testing basic model (no fusion, no contrastive):")
#     model_basic.eval()
#     with torch.no_grad():
#         output_basic = model_basic(x)
#         print(f"Output shape: {output_basic.shape}")
    
#     # Test fusion model
#     print("\n2. Testing fusion model:")
#     model_fusion.eval()
#     with torch.no_grad():
#         output_fusion = model_fusion(x, y)
#         print(f"Output shape: {output_fusion.shape}")
    
#     # Test bypass model
#     print("\n3. Testing bypass model:")
#     model_bypass.eval()
#     with torch.no_grad():
#         output_bypass = model_bypass(x, y)
#         print(f"Output shape: {output_bypass.shape}")
    
#     # Test contrastive model
#     print("\n4. Testing contrastive model:")
#     model_contrastive.eval()
#     with torch.no_grad():
#         output_masks, f1, f2 = model_contrastive(x, y)
#         print(f"Masks shape: {output_masks.shape}")
#         print(f"Contrastive feature 1 shape: {f1.shape}")
#         print(f"Contrastive feature 2 shape: {f2.shape}")
    
#     # Test predict method
#     print("\n5. Testing predict method:")
#     with torch.no_grad():
#         pred_output = model_fusion.predict(x, y)
#         print(f"Prediction output shape: {pred_output.shape}")

#     print("\nAll tests completed successfully!")

# def test_model_training():
#     # Initialize model
#     model = Unet(
#         encoder_name="resnet18",
#         fusion=True,
#         contrastive=True
#     ).to('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # Create dummy training data
#     batch_size = 2
#     x = torch.randn(batch_size, 3, 256, 256).to('cuda' if torch.cuda.is_available() else 'cpu')
#     y = torch.randn(batch_size, 3, 256, 256).to('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # Define loss functions
#     segmentation_criterion = torch.nn.MSELoss()  # or your preferred segmentation loss
#     contrastive_criterion = torch.nn.CosineSimilarity(dim=1)  # for contrastive loss
    
#     # Optimizer
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
#     # Training loop
#     model.train()
#     optimizer.zero_grad()
    
#     # Forward pass
#     masks, f1, f2 = model(x, y)
    
#     # Calculate losses
#     seg_loss = segmentation_criterion(masks, torch.randn_like(masks))  # replace with your target
#     contrastive_loss = -contrastive_criterion(f1, f2).mean()  # maximize similarity
    
#     # Combined loss
#     total_loss = seg_loss + 0.1 * contrastive_loss  # adjust weight as needed
    
#     # Backward pass
#     total_loss.backward()
#     optimizer.step()
    
#     print(f"Training test completed. Total loss: {total_loss.item()}")

# if __name__ == "__main__":
#     print("Starting model tests...")
#     test_model()
#     print("\nStarting training test...")
#     test_model_training()



def test_model():
    # Test device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test configurations
    test_configs = [
        {
            "name": "Basic UNet",
            "params": {
                "encoder_name": "resnet34",
                "fusion": False,
                "contrastive": False
            }
        },
        {
            "name": "Fusion UNet",
            "params": {
                "encoder_name": "resnet34",
                "fusion": True,
                "contrastive": False
            }
        },
        {
            "name": "Contrastive UNet",
            "params": {
                "encoder_name": "resnet34",
                "fusion": True,
                "contrastive": True
            }
        }
    ]
    
    # Create dummy inputs
    batch_size = 2
    channels = 3
    height = 256  # Make sure it's divisible by output_stride
    width = 256   # Make sure it's divisible by output_stride
    
    x = torch.randn(batch_size, channels, height, width).to(device)
    y = torch.randn(batch_size, channels, height, width).to(device)
    
    # Test each configuration
    for config in test_configs:
        print(f"\nTesting {config['name']}:")
        try:
            # Initialize model
            model = Unet(**config['params']).to(device)
            model.eval()
            
            with torch.no_grad():
                # Test without fusion input
                if not config['params']['fusion']:
                    output = model(x)
                    print(f"Forward pass (no fusion) output shape: {output.shape}")
                
                # Test with fusion input
                if config['params']['fusion']:
                    output = model(x, y)
                    if config['params']['contrastive']:
                        masks, f1, f2 = output
                        print(f"Forward pass output shapes:")
                        print(f"- Masks: {masks.shape}")
                        print(f"- Contrastive feature 1: {f1.shape}")
                        print(f"- Contrastive feature 2: {f2.shape}")
                    else:
                        print(f"Forward pass (with fusion) output shape: {output.shape}")
                
                # Test predict method
                pred_output = model.predict(x, y if config['params']['fusion'] else None)
                print(f"Predict method output shape: {pred_output.shape}")
                
            print(f"{config['name']} tests passed!")
            
        except Exception as e:
            print(f"Error testing {config['name']}: {str(e)}")

def test_training():
    print("\nTesting training functionality:")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model with contrastive learning
    model = Unet(
        encoder_name="resnet34",
        fusion=True,
        contrastive=True
    ).to(device)
    
    # Create dummy data
    batch_size = 2
    x = torch.randn(batch_size, 3, 256, 256).to(device)
    y = torch.randn(batch_size, 3, 256, 256).to(device)
    target = torch.randn(batch_size, 1, 256, 256).to(device)  # Assuming binary segmentation
    
    # Define loss functions
    segmentation_criterion = torch.nn.MSELoss()
    contrastive_criterion = torch.nn.CosineSimilarity(dim=1)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    try:
        # Training iteration
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        masks, f1, f2 = model(x, y)
        
        # Calculate losses
        seg_loss = segmentation_criterion(masks, target)
        contrastive_loss = -contrastive_criterion(f1, f2).mean()  # Maximize similarity
        total_loss = seg_loss + 0.1 * contrastive_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        print(f"Training test completed successfully!")
        print(f"- Segmentation loss: {seg_loss.item():.4f}")
        print(f"- Contrastive loss: {contrastive_loss.item():.4f}")
        print(f"- Total loss: {total_loss.item():.4f}")
        
    except Exception as e:
        print(f"Error in training test: {str(e)}")

def test_different_input_sizes():
    print("\nTesting different input sizes:")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = Unet(encoder_name="resnet34", fusion=True).to(device)
    model.eval()
    
    # Test different input sizes
    sizes = [(224, 224), (384, 384), (512, 512)]
    
    for h, w in sizes:
        try:
            print(f"\nTesting input size: {h}x{w}")
            x = torch.randn(1, 3, h, w).to(device)
            y = torch.randn(1, 3, h, w).to(device)
            
            with torch.no_grad():
                output = model(x, y)
                print(f"Output shape: {output.shape}")
                
        except Exception as e:
            print(f"Error with input size {h}x{w}: {str(e)}")

if __name__ == "__main__":
    print("Starting model tests...")
    test_model()
    test_training()
    test_different_input_sizes()