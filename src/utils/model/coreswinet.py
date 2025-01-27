import torch
import torch.nn as nn
from einops import rearrange, repeat
# from quaternion_layers import QuaternionTransposeConv,QuaternionConv, QuaternionLinearAutograd

class Linear(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True):
        super(Linear, self).__init__()

        self.Linear = nn.Linear(in_channel, out_channel, bias=bias)

    def forward(self, x):
        out = self.Linear(x)
        return out

# Input Projection
class InputProj(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1, norm_layer=None,act_layer=nn.LeakyReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
            act_layer(inplace=True)
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        drop_rate = 0.
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.pos_drop = nn.Dropout(p=drop_rate)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        if self.norm is not None:
            x = self.norm(x)
        return self.pos_drop(x)

class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3, stride=1, norm_layer=None,act_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
        )
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x, H, W):
        B, L, C = x.shape
        # import pdb;pdb.set_trace()
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.conv(x).flatten(2).transpose(1,2).contiguous()  # B H*W C
        return out


class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.deconv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out

class DHA(nn.Module):#LCSA_Layer
    def __init__(self, channel_num, reduction=16):
        super(DHA, self).__init__()

        self.L0 = Linear(channel_num,channel_num)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.Hash = nn.Sequential(
                nn.Conv2d(channel_num, channel_num // reduction, 1, padding=0, bias=True),
                nn.GELU(),
                nn.Conv2d(channel_num // reduction, channel_num, 1, padding=0, bias=True),
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
    '''Frequency-Hierarchy module'''

    def __init__(self, channel_num):
        super(DF, self).__init__()

        self.C0 = nn.Sequential(
             nn.Conv2d(channel_num, channel_num//3, groups=channel_num//3, kernel_size=3, stride=1, padding=1),
             nn.LeakyReLU(inplace=True))
             
        self.C1 = nn.Sequential(
             nn.Conv2d(channel_num, channel_num//3, groups=channel_num//3, kernel_size=3, stride=1, padding=2, dilation = 2),
             nn.LeakyReLU(inplace=True))    

        self.C2 = nn.Sequential(
             nn.Conv2d(channel_num, channel_num//3, groups=channel_num//3, kernel_size=3, stride=1, padding=3, dilation = 3),
             nn.LeakyReLU(inplace=True))

        self.R = nn.GELU()

    def forward(self, x):
        l = self.R(self.C2(x))
        m = self.R(self.C1(x) - l)
        h = self.R(self.C0(x) - self.C1(x))
        return l, m, h   

class EFF(nn.Module):
    '''Frequency enhancement module'''

    def __init__(self, dim=32, out_dim=128):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, dim),
                                     nn.GELU())
        self.DF = DF(dim)
        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, groups=dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True))
        self.linear2 = nn.Sequential(nn.Linear(dim, out_dim))

    def forward(self, x, H, W):
        # bs x hw x c
#        short = x
        bs, hw, c = x.size()
        x = self.linear1(x)
        
        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=H, w=W)
        sht = x
        # bs,hidden_dim,32x32
        l, m, h = self.DF(x)
        x = torch.cat((l, m, h), dim = 1)
        x = self.dwconv(x)
        x = x + sht

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h=H, w=W)

        x = self.linear2(x)

        return x
        
class AFEBlock(nn.Module):
    def __init__(self, dim=3, out_dim = 6):
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
        sht = x

        # EFF
        x = self.EFF(self.LN2(x)+ sht, H, W)
        return x
    
class Model(nn.Module):
    def __init__(self, in_channel=3, dim=54, contrastive=True, bypass=False):
        super().__init__()
        self.contrastive = contrastive
        self.bypass = bypass

        # First encoder path
        self.In1 = InputProj(in_channel, dim)
        self.FETBlock1_1 = AFEBlock(dim, dim)
        self.D1_1 = Downsample(dim, dim)
        self.FETBlock2_1 = AFEBlock(dim, dim*2)
        self.D2_1 = Downsample(dim*2, dim*2)
        self.FETBlock3_1 = AFEBlock(dim*2, dim*4)
        self.D3_1 = Downsample(dim*4, dim*4)
        self.FETBlock4_1 = AFEBlock(dim*4, dim*8)
        self.D4_1 = Downsample(dim*8, dim*8)
        self.FETBlock5_1 = AFEBlock(dim*8, dim*16)
        self.D5_1 = Downsample(dim*16, dim*16)

        # Second encoder path
        self.In2 = InputProj(in_channel, dim)
        self.FETBlock1_2 = AFEBlock(dim, dim)
        self.D1_2 = Downsample(dim, dim)
        self.FETBlock2_2 = AFEBlock(dim, dim*2)
        self.D2_2 = Downsample(dim*2, dim*2)
        self.FETBlock3_2 = AFEBlock(dim*2, dim*4)
        self.D3_2 = Downsample(dim*4, dim*4)
        self.FETBlock4_2 = AFEBlock(dim*4, dim*8)
        self.D4_2 = Downsample(dim*8, dim*8)
        self.FETBlock5_2 = AFEBlock(dim*8, dim*16)
        self.D5_2 = Downsample(dim*16, dim*16)

        # Feature processing blocks
        self.process1 = AFEBlock(dim, dim)
        self.process2 = AFEBlock(dim*2, dim*2)
        self.process3 = AFEBlock(dim*4, dim*4)
        self.process4 = AFEBlock(dim*8, dim*8)
        self.process5 = AFEBlock(dim*16, dim*16)

        # Bottleneck
        self.BNeck = AFEBlock(dim*16, dim*32)

        # Contrastive heads
        if contrastive:
            self.contrastive_head1 = nn.Sequential(
                nn.LayerNorm(dim*16),
                nn.Linear(dim*16, 512),
                nn.GELU(),
                nn.Linear(512, 64),
                nn.LayerNorm(64)
            )
            self.contrastive_head2 = nn.Sequential(
                nn.LayerNorm(dim*16),
                nn.Linear(dim*16, 512),
                nn.GELU(),
                nn.Linear(512, 64),
                nn.LayerNorm(64)
            )

        # Decoder path
        self.U6 = Upsample(dim*32, dim*16)
        self.FETBlock6 = AFEBlock(dim*32, dim*16)
        
        self.U7 = Upsample(dim*16, dim*8)
        self.FETBlock7 = AFEBlock(dim*16, dim*8)
        
        self.U8 = Upsample(dim*8, dim*4)
        self.FETBlock8 = AFEBlock(dim*8, dim*4)
        
        self.U9 = Upsample(dim*4, dim*2)
        self.FETBlock9 = AFEBlock(dim*4, dim*2)
        
        self.U10 = Upsample(dim*2, dim)
        self.FETBlock10 = AFEBlock(dim*2, dim)

        self.Out = OutputProj(dim, in_channel)

    def process_features(self, feat1, feat2, process_block, H, W):
        if self.bypass:
            # Skip element-wise max and directly process feat1
            return process_block(feat1, H, W)
        else:
            # Original processing with element-wise maximum
            max_feat = torch.maximum(feat1, feat2)
            return process_block(max_feat, H, W)

    def forward(self, x1, x2):
        H, W = x1.shape[2:]
        short_x = x1

        # First encoder path
        x1 = self.In1(x1)
        conv1_1 = self.FETBlock1_1(x1, H, W)
        pool1_1 = self.D1_1(conv1_1, H, W)
        
        conv2_1 = self.FETBlock2_1(pool1_1, H//2, W//2)
        pool2_1 = self.D2_1(conv2_1, H//2, W//2)
        
        conv3_1 = self.FETBlock3_1(pool2_1, H//4, W//4)
        pool3_1 = self.D3_1(conv3_1, H//4, W//4)
        
        conv4_1 = self.FETBlock4_1(pool3_1, H//8, W//8)
        pool4_1 = self.D4_1(conv4_1, H//8, W//8)
        
        conv5_1 = self.FETBlock5_1(pool4_1, H//16, W//16)
        pool5_1 = self.D5_1(conv5_1, H//16, W//16)

        # Second encoder path
        x2 = self.In2(x2)
        conv1_2 = self.FETBlock1_2(x2, H, W)
        pool1_2 = self.D1_2(conv1_2, H, W)
        
        conv2_2 = self.FETBlock2_2(pool1_2, H//2, W//2)
        pool2_2 = self.D2_2(conv2_2, H//2, W//2)
        
        conv3_2 = self.FETBlock3_2(pool2_2, H//4, W//4)
        pool3_2 = self.D3_2(conv3_2, H//4, W//4)
        
        conv4_2 = self.FETBlock4_2(pool3_2, H//8, W//8)
        pool4_2 = self.D4_2(conv4_2, H//8, W//8)
        
        conv5_2 = self.FETBlock5_2(pool4_2, H//16, W//16)
        pool5_2 = self.D5_2(conv5_2, H//16, W//16)

        # Process features at each level through AFE blocks
        skip1 = self.process_features(conv1_1, conv1_2, self.process1, H, W)
        skip2 = self.process_features(conv2_1, conv2_2, self.process2, H//2, W//2)
        skip3 = self.process_features(conv3_1, conv3_2, self.process3, H//4, W//4)
        skip4 = self.process_features(conv4_1, conv4_2, self.process4, H//8, W//8)
        skip5 = self.process_features(conv5_1, conv5_2, self.process5, H//16, W//16)

        # Process final bottleneck
        bottleneck = self.process_features(pool5_1, pool5_2, self.BNeck, H//32, W//32)

        # Calculate contrastive features if needed
        if self.contrastive:
            f1 = self.contrastive_head1(pool5_1.mean(dim=1))  # Global average pooling
            f2 = self.contrastive_head2(pool5_2.mean(dim=1))

        # Decoder path with processed skip connections
        up6 = self.U6(bottleneck, H//32, W//32)
        up6 = torch.cat([up6, skip5], 2)
        conv6 = self.FETBlock6(up6, H//16, W//16)

        up7 = self.U7(conv6, H//16, W//16)
        up7 = torch.cat([up7, skip4], 2)
        conv7 = self.FETBlock7(up7, H//8, W//8)

        up8 = self.U8(conv7, H//8, W//8)
        up8 = torch.cat([up8, skip3], 2)
        conv8 = self.FETBlock8(up8, H//4, W//4)

        up9 = self.U9(conv8, H//4, W//4)
        up9 = torch.cat([up9, skip2], 2)
        conv9 = self.FETBlock9(up9, H//2, W//2)

        up10 = self.U10(conv9, H//2, W//2)
        up10 = torch.cat([up10, skip1], 2)
        conv10 = self.FETBlock10(up10, H, W)

        output = self.Out(conv10, H, W) + short_x

        if self.contrastive:
            return output, f1, f2
        return output

if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    
    # Test normal forward pass
    x1 = torch.randn(1, 3, 128, 128)
    x2 = torch.randn(1, 3, 128, 128)
    M = Model(dim=54, contrastive=True)
    y, f1, f2 = M(x1, x2)
    print(f"Output shape: {y.shape}")
    print(f"Contrastive feature shapes: {f1.shape}, {f2.shape}")

    # Custom input function for complexity analysis
    def prepare_input(resolution):
        return {"x1": torch.randn(1, 3, *resolution), 
                "x2": torch.randn(1, 3, *resolution)}

    flops, params = get_model_complexity_info(
        M, 
        (256, 256), 
        input_constructor=prepare_input,
        as_strings=True, 
        print_per_layer_stat=False
    )
    print('Flops:  ' + flops)
    print('Params: ' + params)
