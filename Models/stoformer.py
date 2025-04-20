import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MLP(nn.Module):
    """Multi-Layer Perceptron module"""
    def __init__(self, dim, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or dim
        hidden_features = hidden_features or dim
        self.fc1 = nn.Linear(dim, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size, shift_h=0, shift_w=0):
    """
    Partition the input feature map into windows with stochastic shift
    Args:
        x: (B, H, W, C)
        window_size: window size
        shift_h, shift_w: shift in h and w directions
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    
    # Pad feature map to handle shift and ensure full window coverage
    pad_h = (window_size - (H + shift_h) % window_size) % window_size
    pad_w = (window_size - (W + shift_w) % window_size) % window_size
    
    if shift_h > 0 or shift_w > 0 or pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, shift_w, pad_w, shift_h, pad_h))
    
    _, Hp, Wp, _ = x.shape
    
    # Calculate number of windows in each dimension
    num_h = Hp // window_size
    num_w = Wp // window_size
    
    # Partition windows
    x = x.view(B, num_h, window_size, num_w, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    
    return windows, (num_h, num_w)

def window_reverse(windows, window_size, H, W, window_info=None):
    """
    Reverse window partition
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H, W: Original feature map size
        window_info: (num_h, num_w) tuple with window counts
    Returns:
        x: (B, H, W, C)
    """
    # Get dimensions from windows
    total_windows = windows.shape[0]
    C = windows.shape[-1]
    
    # Get window counts from provided info or calculate
    if window_info:
        num_h, num_w = window_info
    else:
        # Calculate padded dimensions that would be created by window_partition
        H_padded = math.ceil(H / window_size) * window_size
        W_padded = math.ceil(W / window_size) * window_size
        num_h = H_padded // window_size
        num_w = W_padded // window_size
    
    # Calculate batch size
    B = total_windows // (num_h * num_w)
    
    # Reshape the windows tensor back to original format
    x = windows.view(B, num_h, num_w, window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    
    # Calculate padded dimensions
    H_padded = num_h * window_size
    W_padded = num_w * window_size
    
    # Reshape to the padded dimensions
    x = x.view(B, H_padded, W_padded, C)
    
    # Crop to the original dimensions if needed
    if H_padded > H or W_padded > W:
        x = x[:, :H, :W, :].contiguous()
    
    return x

class StoWinMSA(nn.Module):
    """Stochastic Window Multi-head Self Attention module"""
    def __init__(self, dim, window_size=8, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        Args:
            x: input features with shape (B, H, W, C)
        """
        B, H, W, C = x.shape
        
        # For training: use stochastic window
        if self.training:
            # Sample random shifts for stochastic window partition
            shift_h = torch.randint(0, self.window_size, (1,), device=x.device)
            shift_w = torch.randint(0, self.window_size, (1,), device=x.device)
            
            # Window partition with stochastic shift - also get window counts
            x_windows, window_info = window_partition(x, self.window_size, shift_h, shift_w)
            
            # Compute attention within each window
            x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
            
            # Multi-head self-attention
            qkv = self.qkv(x_windows).reshape(-1, self.window_size * self.window_size, 3, self.num_heads, C // self.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            
            x_windows = (attn @ v).transpose(1, 2).reshape(-1, self.window_size * self.window_size, C)
            x_windows = self.proj(x_windows)
            x_windows = self.proj_drop(x_windows)
            
            # Reverse window partition - pass along window_info
            x_windows = x_windows.view(-1, self.window_size, self.window_size, C)
            x = window_reverse(x_windows, self.window_size, H, W, window_info)
            
            return x
            
        # For inference: use layer expectation propagation
        else:
            # Number of window partitions to average (simplified version)
            num_expectation = min(self.window_size * self.window_size, 8)  # Reduce for efficiency
            
            output = torch.zeros_like(x)
            count = torch.zeros((B, H, W, 1), device=x.device)
            
            # Compute expectation over a subset of window partitions
            for i in range(num_expectation):
                shift_h = (i // int(math.sqrt(num_expectation))) * (self.window_size // int(math.sqrt(num_expectation)))
                shift_w = (i % int(math.sqrt(num_expectation))) * (self.window_size // int(math.sqrt(num_expectation)))
                
                # Window partition with current shift
                x_windows, window_info = window_partition(x, self.window_size, shift_h, shift_w)
                x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
                
                # Multi-head self-attention
                qkv = self.qkv(x_windows).reshape(-1, self.window_size * self.window_size, 3, self.num_heads, C // self.num_heads)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                
                attn = (q @ k.transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                
                x_windows = (attn @ v).transpose(1, 2).reshape(-1, self.window_size * self.window_size, C)
                x_windows = self.proj(x_windows)
                x_windows = self.proj_drop(x_windows)
                
                # Reverse window partition
                x_windows = x_windows.view(-1, self.window_size, self.window_size, C)
                x_out = window_reverse(x_windows, self.window_size, H, W, window_info)
                
                # Make a binary mask for valid regions
                mask = torch.ones((B, H, W, 1), device=x.device)
                if shift_h > 0 or shift_w > 0:
                    if shift_h > 0:
                        mask[:, :shift_h, :, :] = 0
                    if shift_w > 0:
                        mask[:, :, :shift_w, :] = 0
                
                # Accumulate results and counts
                output = output + x_out * mask
                count = count + mask
            
            # Average the accumulated results
            output = output / (count + 1e-8)
            return output

class StoBlock(nn.Module):
    """Stochastic Window Transformer Block"""
    def __init__(self, dim, num_heads, window_size=8, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = StoWinMSA(
            dim, window_size=window_size, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)
        
        # Use drop path for stochastic depth if specified
        self.drop_path = nn.Identity() if drop_path == 0. else DropPath(drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # x is (B, H, W, C)
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)  # StoWin-MSA
        x = shortcut + self.drop_path(x)
        
        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.permute(0, 2, 3, 1)  # (B, H/patch_size, W/patch_size, embed_dim)
        x = self.norm(x)
        return x

class PatchMerging(nn.Module):
    """Patch Merging Layer - downsamples by 2x"""
    def __init__(self, input_dim, output_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = input_dim
        self.reduction = nn.Linear(4 * input_dim, output_dim, bias=False)
        self.norm = norm_layer(4 * input_dim)

    def forward(self, x):
        """
        x: (B, H, W, C)
        """
        B, H, W, C = x.shape
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x0 = x[:, 0::2, 0::2, :]  # (B, H/2, W/2, C)
        x1 = x[:, 1::2, 0::2, :]  # (B, H/2, W/2, C)
        x2 = x[:, 0::2, 1::2, :]  # (B, H/2, W/2, C)
        x3 = x[:, 1::2, 1::2, :]  # (B, H/2, W/2, C)
        x = torch.cat([x0, x1, x2, x3], -1)  # (B, H/2, W/2, 4*C)
        x = self.norm(x)
        x = self.reduction(x)  # (B, H/2, W/2, output_dim)

        return x

class PatchExpand(nn.Module):
    """Patch Expanding Layer - upsamples by 2x"""
    def __init__(self, input_dim, output_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = input_dim
        self.expand = nn.Linear(input_dim, output_dim * 2 * 2, bias=False)
        self.norm = norm_layer(output_dim)

    def forward(self, x):
        """
        x: (B, H, W, C)
        """
        B, H, W, C = x.shape
        x = self.expand(x)  # (B, H, W, 4*output_dim)
        x = x.view(B, H, W, 2, 2, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H * 2, W * 2, -1)  # (B, H*2, W*2, output_dim)
        x = self.norm(x)

        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # work with diff dim tensors, not just 2D ConvNets
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

class Stoformer(nn.Module):
    """Stochastic Window Transformer for Image Restoration"""
    def __init__(self, 
                 img_size=224,
                 in_chans=3, 
                 embed_dim=32,
                 depths=[1, 2, 8, 8], 
                 num_heads=[1, 2, 4, 8],
                 window_size=8, 
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 drop_rate=0.,
                 attn_drop_rate=0., 
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, 
                 patch_norm=True,
                 **kwargs):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        
        # Input embedding - no patch partitioning, just a conv
        self.input_proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        
        dims = [embed_dim]
        for i in range(self.num_layers):
            dim = embed_dim * (2 ** i)
            dims.append(dim)
        
        # Encoder blocks
        for i_layer in range(self.num_layers):
            layer = nn.ModuleList()
            for i_block in range(depths[i_layer]):
                layer.append(
                    StoBlock(
                        dim=dims[i_layer],
                        num_heads=num_heads[i_layer],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=drop_path_rate,
                        norm_layer=norm_layer
                    )
                )
            self.encoder_layers.append(layer)
            
            # Add downsampling layer except for the last layer
            if i_layer < self.num_layers - 1:
                self.downsample_layers.append(
                    PatchMerging(dims[i_layer], dims[i_layer+1], norm_layer=norm_layer)
                )
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        
        # Upsample blocks
        for i_layer in range(self.num_layers - 1, -1, -1):
            # Add upsampling layer except for the last layer
            if i_layer > 0:
                self.upsample_layers.append(
                    PatchExpand(dims[i_layer], dims[i_layer-1], norm_layer=norm_layer)
                )
            
            # Add decoder blocks - mirrored from encoder
            layer = nn.ModuleList()
            for i_block in range(depths[i_layer]):
                layer.append(
                    StoBlock(
                        dim=dims[i_layer] if i_layer == self.num_layers - 1 else dims[i_layer],
                        num_heads=num_heads[i_layer],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=drop_path_rate,
                        norm_layer=norm_layer
                    )
                )
            self.decoder_layers.append(layer)
        
        # Output projection
        self.output_proj = nn.Conv2d(embed_dim, in_chans, kernel_size=3, stride=1, padding=1)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_encoder(self, x):
        """Encoder forward pass"""
        features = []
        
        # Initial embedding
        x = self.input_proj(x)  # (B, embed_dim, H, W)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, embed_dim)
        
        # Encoder
        for i_layer, layer_blocks in enumerate(self.encoder_layers):
            # Add all features for skip connections
            features.append(x)
            
            # Apply blocks in current layer
            for block in layer_blocks:
                x = block(x)
            
            # Downsample if not the last layer
            if i_layer < self.num_layers - 1:
                x = self.downsample_layers[i_layer](x)
        
        return x, features
    
    def forward_decoder(self, x, features):
        """Decoder forward pass"""
        # Decoder
        for i_layer, layer_blocks in enumerate(self.decoder_layers):
            # Upsample if not the first layer (which is the bottleneck)
            if i_layer > 0:
                x = self.upsample_layers[i_layer-1](x)
                
                # Skip connection - add features from corresponding encoder layer
                skip_idx = self.num_layers - i_layer - 1
                if skip_idx >= 0:
                    x = x + features[skip_idx]
            
            # Apply blocks in current layer
            for block in layer_blocks:
                x = block(x)
        
        return x
    
    def forward(self, x):
        """Forward function"""
        # Save original input for residual connection
        x_orig = x
        
        # Encoder
        x, features = self.forward_encoder(x)
        
        # Decoder
        x = self.forward_decoder(x, features)
        
        # Output projection
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        x = self.output_proj(x)
        
        # Residual connection
        x = x + x_orig
        
        return x

def build_stoformer(img_size=256, window_size=8):
    """Build Stoformer model for image restoration"""
    model = Stoformer(
        img_size=img_size,
        in_chans=3,
        embed_dim=32,
        depths=[1, 2, 8, 8],  # Number of StoBlocks at each level
        num_heads=[1, 2, 4, 8],  # Number of attention heads at each layer
        window_size=window_size,
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        patch_norm=True
    )
    return model 