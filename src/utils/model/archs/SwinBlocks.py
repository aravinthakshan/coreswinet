import torch.nn as nn
import torch.nn.functional as F

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Define projection layers
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads=8, window_size=7, 
                 shift_size=0, mlp_ratio=4., qkv_bias=True, 
                 drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = min(window_size, min(input_resolution))  
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # Layer Norm
        self.norm1 = nn.LayerNorm(dim)
        
        # Window Attention
        self.attn = WindowAttention(
            dim, window_size=self.window_size, 
            num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=drop
        )

        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        self.norm2 = nn.LayerNorm(dim)

        # Drop path
        self.drop_path = nn.Identity() if drop_path == 0 else nn.DropPath(drop_path)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x

        # Layer Norm
        x = self.norm1(x)

        # Reshape
        x = x.reshape(B, H, W, C)

        # Perform window partitioning and attention
        x_windows, padding = self.window_partition(x)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # Perform attention
        attn_windows = self.attn(x_windows)
        
        # Reconstruct
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = self.window_reverse(attn_windows, H, W, padding)
        x = x.reshape(B, H * W, C)

        # Residual connection
        x = shortcut + self.drop_path(x)

        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def window_partition(self, x):
        """
        Partition into windows with padding if needed
        """
        B, H, W, C = x.shape
        
        # Compute padding
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        
        # Pad if necessary
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            B, H, W, C = x.shape
        
        # Compute number of windows
        patch_h = H // self.window_size
        patch_w = W // self.window_size
        
        # Partition windows
        x = x.view(B, patch_h, self.window_size, patch_w, self.window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size, self.window_size, C)
        
        return windows, (pad_h, pad_w)

    def window_reverse(self, windows, H, W, padding):
        """
        Reverse window partitioning and remove padding
        """
        pad_h, pad_w = padding
        B = int(windows.shape[0] / ((H + pad_h) / self.window_size * (W + pad_w) / self.window_size))
        
        x = windows.view(B, (H + pad_h) // self.window_size, (W + pad_w) // self.window_size, 
                         self.window_size, self.window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H + pad_h, W + pad_w, -1)
        
        # Remove padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :]
        
        return x