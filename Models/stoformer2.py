import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from einops import rearrange, repeat
import math
import random
import argparse

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: [B, N, C]
        x = torch.transpose(x, 1, 2)  # [B, C, N]
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        x = x * y.expand_as(x)
        x = torch.transpose(x, 1, 2)  # [B, N, C]
        return x


######## Embedding for q,k,v ########

class LinearProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., bias=True, Train=True):
        super(LinearProjection, self).__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.train=Train
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape

        attn_kv = x if attn_kv is None else attn_kv
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1]
        return q, k, v


########### feed-forward network #############
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0.):
        super(LeFF, self).__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                     act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
            act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # bs x hw x c
        bs, hw, c = x.size()
        hh = round(math.sqrt(hw))
        ww = round(math.sqrt(hw))

        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=ww)
        # bs,hidden_dim,32x32

        x = self.dwconv(x)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=ww)

        x = self.linear2(x)

        return x


########### window operation#############
def window_partition(x, win_size, dilation_rate=1):
    B, H, W, C = x.shape
    
    # Calculate padding required to make dimensions multiple of window size
    pad_h = (win_size - H % win_size) % win_size
    pad_w = (win_size - W % win_size) % win_size
    
    # Apply padding if needed
    if pad_h > 0 or pad_w > 0:
        # Pad the input to make H and W multiples of win_size
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        # Update H and W after padding
        H = H + pad_h
        W = W + pad_w
    
    if dilation_rate != 1:
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size, dilation=dilation_rate, padding=4 * (dilation_rate - 1),
                     stride=win_size)  # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0, 2, 1).contiguous().view(-1, C, win_size, win_size)  # B' ,C ,Wh ,Ww
        windows = windows.permute(0, 2, 3, 1).contiguous()  # B' ,Wh ,Ww ,C
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C)  # B' ,Wh ,Ww ,C
    
    # Store original dimensions in a new attribute
    windows.orig_size = (B, H - pad_h, W - pad_w, C)
    return windows


def window_reverse(windows, win_size, H, W, dilation_rate=1):
    # B' ,Wh ,Ww ,C
    # Retrieve original dimensions if available
    if hasattr(windows, 'orig_size'):
        orig_B, orig_H, orig_W, orig_C = windows.orig_size
    else:
        orig_H, orig_W = H, W
        
    # Calculate padded dimensions (must be multiples of win_size)
    pad_H = (win_size - orig_H % win_size) % win_size
    pad_W = (win_size - orig_W % win_size) % win_size
    padded_H, padded_W = orig_H + pad_H, orig_W + pad_W
    
    B = int(windows.shape[0] / (padded_H * padded_W / win_size / win_size))
    
    x = windows.view(B, padded_H // win_size, padded_W // win_size, win_size, win_size, -1)
    if dilation_rate != 1:
        x = windows.permute(0, 5, 3, 4, 1, 2).contiguous()  # B, C*Wh*Ww, H/Wh*W/Ww
        x = F.fold(x, (padded_H, padded_W), kernel_size=win_size, dilation=dilation_rate, padding=4 * (dilation_rate - 1),
                  stride=win_size)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, padded_H, padded_W, -1)
    
    # Remove padding to restore original dimensions
    if pad_H > 0 or pad_W > 0:
        x = x[:, :orig_H, :orig_W, :]
        
    return x


# Downsample Block
class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        H = round(math.sqrt(L))
        W = round(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.conv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out


# Upsample Block
class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        H = round(math.sqrt(L))
        W = round(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.deconv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out


# Input Projection
class InputProj(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1, norm_layer=None, act_layer=nn.LeakyReLU):
        super(InputProj, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size // 2),
            act_layer(inplace=True)
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):

        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        if self.norm is not None:
            x = self.norm(x)
        return x


# Output Projection
class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3, stride=1, norm_layer=None, act_layer=None):
        super(OutputProj, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, 3, kernel_size=3, stride=1, padding=1)
        )
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        H = round(math.sqrt(L))
        W = round(math.sqrt(L))
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


########### StoTransformer #############
class StoTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, win_size=8,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., proj_drop=0.,drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, stride=1, token_mlp='leff',
                 se_layer=False):
        super(StoTransformerBlock, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.stride=stride
        self.mlp_ratio = mlp_ratio
        self.token_mlp = token_mlp
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size - 1) * (2 * win_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size) # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size) # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size- 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size - 1
        relative_coords[:, :, 0] *= 2 * self.win_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.attn_drop = nn.Dropout(attn_drop)

        self.norm1 = norm_layer(dim)

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.softmax = nn.Softmax(dim=-1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.proj = nn.Linear(dim, dim)
        self.se_layer = SELayer(dim) if se_layer else nn.Identity()
        self.proj_drop = nn.Dropout(proj_drop)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer,
                       drop=drop) if token_mlp == 'ffn' else LeFF(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def attention(self, q, k, v, attn_mask=None):
        B_, h, N_, C_ = q.shape

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size * self.win_size, self.win_size * self.win_size, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        ratio = attn.size(-1) // relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d=ratio)
        attn = attn + relative_position_bias.unsqueeze(0)

        if attn_mask is not None:
            nW = attn_mask.shape[0]  # [nW, N_, N_]
            mask = repeat(attn_mask, 'nW m n -> nW m (n d)', d=1)  # [nW, N_, N_]
            attn = attn.view(B_ // nW, nW, self.num_heads, N_, N_ * 1) + mask.unsqueeze(1).unsqueeze(
                0)  # [1, nW, 1, N_, N_]
            # [B, nW, nh, N_, N_]
            attn = attn.view(-1, self.num_heads, N_, N_ * 1)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        y = (attn @ v).transpose(1, 2).reshape(B_, N_, h*C_)
        y = self.proj(y)
        return y

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x, mask=None):
        B, L, C = x.shape

        H = round(math.sqrt(L))
        W = round(math.sqrt(L))

        shortcut = x
        x = self.norm1(x)
        q = self.to_q(x) #[B, L, C]
        kv = self.to_kv(x)

        # Reshape to 2D spatial layout
        q = rearrange(q, 'b (h w) c -> b h w c', h=H)
        kv = rearrange(kv, 'b (h w) c -> b h w c', h=H)

        # Convert to 4D tensor for window partitioning
        x = x.view(B, H, W, C)

        if self.training:
            if mask != None:
                input_mask = F.interpolate(mask, size=(H, W)).permute(0, 2, 3, 1)
                input_mask_windows = window_partition(input_mask, self.win_size)  # nW, win_size, win_size, 1
                attn_mask = input_mask_windows.view(-1, self.win_size * self.win_size)  # nW, win_size*win_size
                attn_mask = attn_mask.unsqueeze(2) * attn_mask.unsqueeze(1)  # nW, win_size*win_size, win_size*win_size
                attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            else:
                attn_mask = None

            ## Stochastic shift window
            H_offset = random.randint(0, self.win_size - 1)
            W_offset = random.randint(0, self.win_size - 1)

            shift_mask = torch.zeros((1, H, W, 1)).type_as(x)

            if H_offset > 0:
                h_slices = (slice(0, -self.win_size),
                          slice(-self.win_size, -H_offset),
                         slice(-H_offset, None))
            else:
                h_slices = (slice(0, None),)
            if W_offset > 0:
                w_slices = (slice(0, -self.win_size),
                          slice(-self.win_size, -W_offset),
                         slice(-W_offset, None))
            else:
                w_slices = (slice(0, None),)

            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    shift_mask[:, h, w, :] = cnt
                    cnt += 1

            shift_mask_windows = window_partition(shift_mask, self.win_size)  # nW, win_size, win_size, 1
            shift_mask_windows = shift_mask_windows.view(-1, self.win_size * self.win_size)  # nW, win_size*win_size
            shift_attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(
                        2)  # nW, win_size*win_size, win_size*win_size
            shift_attn_mask = shift_attn_mask.masked_fill(shift_attn_mask != 0, float(-100.0)).masked_fill(
                    shift_attn_mask == 0, float(0.0))
            attn_mask = attn_mask + shift_attn_mask if attn_mask is not None else shift_attn_mask #[nW, N_,N_]

            # cyclic shift
            shifted_q = torch.roll(q, shifts=(-H_offset, -W_offset), dims=(1, 2))
            shifted_kv = torch.roll(kv, shifts=(-H_offset, -W_offset), dims=(1, 2))

            # partition windows
            q_windows = window_partition(shifted_q, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
            q_windows = q_windows.view(-1, self.win_size * self.win_size, C)  # nW*B, win_size*win_size, C
            B_, N_, C_ = q_windows.shape
            q_windows = q_windows.reshape(B_, N_, self.num_heads, C_ // self.num_heads).permute(0, 2, 1, 3)

            kv_windows = window_partition(shifted_kv, self.win_size)  # nW*B, win_size, win_size, 2C
            kv_windows = kv_windows.view(-1, self.win_size * self.win_size, 2 * C)
            kv_windows = kv_windows.reshape(B_, N_, 2, self.num_heads, C_ // self.num_heads).permute(2, 0, 3, 1, 4)
            k_windows, v_windows = kv_windows[0], kv_windows[1]

            attn_windows = self.attention(q_windows, k_windows, v_windows, attn_mask)

            attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
            x = window_reverse(attn_windows, self.win_size, H, W)  # B H' W' C

            x = torch.roll(x, shifts=(H_offset, W_offset), dims=(1, 2))

            x = x.view(B, H * W, C)
            del attn_mask

        else:
            # Optimized inference - sample only specific window shifts
            # Use a strategic sampling approach - take corners and center for good coverage
            if hasattr(self, 'fast_inference') and self.fast_inference:
                # Sample only a small subset of offsets strategically
                # This reduces computation drastically while maintaining accuracy
                offsets = []
                # Add corners
                offsets.append((0, 0))
                offsets.append((0, self.win_size-1))
                offsets.append((self.win_size-1, 0))
                offsets.append((self.win_size-1, self.win_size-1))
                
                # Add center point
                mid = self.win_size // 2
                offsets.append((mid, mid))
                
                # Add a few more strategic points
                third = self.win_size // 3
                two_third = 2 * self.win_size // 3
                offsets.append((third, third))
                offsets.append((third, two_third))
                offsets.append((two_third, third))
                offsets.append((two_third, two_third))
            else:
                # Original full computation - all window shift combinations
                offsets = [(h, w) for h in range(self.win_size) for w in range(self.win_size)]
            
            # Create a tensor to store averaged results on the device where x is
            avg = torch.zeros((B, H*W, C), device=x.device)
            NUM = 0
            
            # Loop through selected offsets only
            for H_offset, W_offset in offsets:
                if mask != None:
                    input_mask = F.interpolate(mask, size=(H, W)).permute(0, 2, 3, 1)
                    input_mask_windows = window_partition(input_mask, self.win_size)  # nW, win_size, win_size, 1
                    attn_mask = input_mask_windows.view(-1, self.win_size * self.win_size)  # nW, win_size*win_size
                    attn_mask = attn_mask.unsqueeze(2) * attn_mask.unsqueeze(
                        1)  # nW, win_size*win_size, win_size*win_size
                    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0,
                                                                                            float(0.0))
                else:
                    attn_mask = None
                shift_mask = torch.zeros((1, H, W, 1), device=x.device)

                if H_offset > 0:
                    h_slices = (slice(0, -self.win_size),
                            slice(-self.win_size, -H_offset),
                            slice(-H_offset, None))
                else:
                    h_slices = (slice(0, None),)

                if W_offset > 0:
                    w_slices = (slice(0, -self.win_size),
                            slice(-self.win_size, -W_offset),
                            slice(-W_offset, None))
                else:
                    w_slices = (slice(0, None),)

                cnt = 0
                for h in h_slices:
                    for w in w_slices:
                        shift_mask[:, h, w, :] = cnt
                        cnt += 1

                shift_mask_windows = window_partition(shift_mask, self.win_size)  # nW, win_size, win_size, 1
                shift_mask_windows = shift_mask_windows.view(-1,
                                                             self.win_size * self.win_size)  # nW, win_size*win_size
                shift_attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(
                    2)  # nW, win_size*win_size, win_size*win_size
                shift_attn_mask = shift_attn_mask.masked_fill(shift_attn_mask != 0, float(-100.0)).masked_fill(
                    shift_attn_mask == 0, float(0.0))
                attn_mask = attn_mask + shift_attn_mask if attn_mask is not None else shift_attn_mask  # [nW, N_,N_]

                shifted_q = torch.roll(q, shifts=(-H_offset, -W_offset), dims=(1, 2))
                shifted_kv = torch.roll(kv, shifts=(-H_offset, -W_offset), dims=(1, 2))

                # partition windows
                q_windows = window_partition(shifted_q, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
                q_windows = q_windows.view(-1, self.win_size * self.win_size, C)  # nW*B, win_size*win_size, C
                B_, N_, C_ = q_windows.shape
                q_windows = q_windows.reshape(B_, N_, self.num_heads, C_ // self.num_heads).permute(0, 2, 1, 3)

                kv_windows = window_partition(shifted_kv, self.win_size)  # nW*B, win_size, win_size, 2C
                kv_windows = kv_windows.view(-1, self.win_size * self.win_size, 2*C)
                kv_windows =  kv_windows.reshape(B_, N_, 2, self.num_heads, C_ // self.num_heads).permute(2, 0, 3, 1, 4)
                k_windows, v_windows = kv_windows[0], kv_windows[1]

                attn_windows = self.attention(q_windows, k_windows, v_windows, attn_mask)

                attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
                shifted_x = window_reverse(attn_windows, self.win_size, H, W)  # B H' W' C
                # reverse cyclic shift
                y = torch.roll(shifted_x, shifts=(H_offset, W_offset), dims=(1, 2))

                y = y.view(B, H * W, C)
                avg = NUM/(NUM+1)*avg + y/(NUM+1)
                NUM += 1
                del attn_mask
            x = avg
            
        # Apply residual connection, normalization and MLP
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


########### Basic layer of Stoformer ################
class BasicStoformerLayer(nn.Module):
    def __init__(self, dim, output_dim, input_resolution, depth, num_heads, win_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
                 token_mlp='leff', se_layer=False):

        super(BasicStoformerLayer, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        # build blocks
        self.blocks = nn.ModuleList([
            StoTransformerBlock(dim=dim, input_resolution=input_resolution,
                                  num_heads=num_heads, win_size=win_size,
                                  mlp_ratio=mlp_ratio,
                                  qkv_bias=qkv_bias,
                                  drop=drop, attn_drop=attn_drop,
                                  drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                  norm_layer=norm_layer, token_mlp=token_mlp,
                                  se_layer=se_layer)
            for i in range(depth)])


    def forward(self, x, mask=None):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, mask)
        return x 

class Stoformer(nn.Module):
    def __init__(self, img_size=128, in_chans=3,
                 embed_dim=32, depths=[1, 2, 8, 8, 2, 8, 8, 2, 1], num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
                 win_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, token_mlp='leff', se_layer=False,
                 dowsample=Downsample, upsample=Upsample, **kwargs):
        super(Stoformer, self).__init__()

        self.num_enc_layers = len(depths) // 2
        self.num_dec_layers = len(depths) // 2
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.mlp = token_mlp
        self.win_size = win_size
        self.reso = img_size
        self.pos_drop = nn.Dropout(p=drop_rate)
        # stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))]
        conv_dpr = [drop_path_rate] * depths[4]
        dec_dpr = enc_dpr[::-1]
        # build layers

        # Input/Output
        self.input_proj = InputProj(in_channel=in_chans, out_channel=embed_dim, kernel_size=3, stride=1,
                                    act_layer=nn.LeakyReLU)
        self.output_proj = OutputProj(in_channel=2 * embed_dim, out_channel=in_chans, kernel_size=3, stride=1)

        # Encoder
        self.encoderlayer_0 = BasicStoformerLayer(dim=embed_dim,
                                                output_dim=embed_dim,
                                                input_resolution=(img_size,
                                                                  img_size),
                                                depth=depths[0],
                                                num_heads=num_heads[0],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=enc_dpr[sum(depths[:0]):sum(depths[:1])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_mlp=token_mlp,
                                                se_layer=se_layer)
        self.dowsample_0 = dowsample(embed_dim, embed_dim * 2)

        self.encoderlayer_1 = BasicStoformerLayer(dim=embed_dim * 2,
                                                output_dim=embed_dim * 2,
                                                input_resolution=(img_size // 2,
                                                                  img_size // 2),
                                                depth=depths[1],
                                                num_heads=num_heads[1],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_mlp=token_mlp,
                                                se_layer=se_layer)
        self.dowsample_1 = dowsample(embed_dim * 2, embed_dim * 4)

        self.encoderlayer_2 = BasicStoformerLayer(dim=embed_dim * 4,
                                                output_dim=embed_dim * 4,
                                                input_resolution=(img_size // (2 ** 2),
                                                                  img_size // (2 ** 2)),
                                                depth=depths[2],
                                                num_heads=num_heads[2],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=enc_dpr[sum(depths[:2]):sum(depths[:3])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_mlp=token_mlp,
                                                se_layer=se_layer)
        self.dowsample_2 = dowsample(embed_dim * 4, embed_dim * 8)

        self.encoderlayer_3 = BasicStoformerLayer(dim=embed_dim * 8,
                                                output_dim=embed_dim * 8,
                                                input_resolution=(img_size // (2 ** 3),
                                                                  img_size // (2 ** 3)),
                                                depth=depths[3],
                                                num_heads=num_heads[3],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=enc_dpr[sum(depths[:3]):sum(depths[:4])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_mlp=token_mlp,
                                                se_layer=se_layer)
        self.dowsample_3 = dowsample(embed_dim * 8, embed_dim * 16)
        # Bottleneck
        self.conv = BasicStoformerLayer(dim=embed_dim * 16,
                                      output_dim=embed_dim * 16,
                                      input_resolution=(img_size // (2 ** 4),
                                                        img_size // (2 ** 4)),
                                      depth=depths[4],
                                      num_heads=num_heads[4],
                                      win_size=win_size,
                                      mlp_ratio=self.mlp_ratio,
                                      qkv_bias=qkv_bias,
                                      drop=drop_rate, attn_drop=attn_drop_rate,
                                      drop_path=conv_dpr,
                                      norm_layer=norm_layer,
                                      use_checkpoint=use_checkpoint,
                                        token_mlp=token_mlp, se_layer=se_layer)
        # Decoder
        self.upsample_0 = upsample(embed_dim * 16, embed_dim * 8)
        self.decoderlayer_0 = BasicStoformerLayer(dim=embed_dim * 16,
                                                output_dim=embed_dim * 16,
                                                input_resolution=(img_size // (2 ** 3),
                                                                  img_size // (2 ** 3)),
                                                depth=depths[5],
                                                num_heads=num_heads[5],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=dec_dpr[:depths[5]],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                 token_mlp=token_mlp,
                                                se_layer=se_layer)

        self.upsample_1 = upsample(embed_dim * 16, embed_dim * 4)
        self.decoderlayer_1 = BasicStoformerLayer(dim=embed_dim * 8,
                                                output_dim=embed_dim * 8,
                                                input_resolution=(img_size // (2 ** 2),
                                                                  img_size // (2 ** 2)),
                                                depth=depths[6],
                                                num_heads=num_heads[6],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=dec_dpr[sum(depths[5:6]):sum(depths[5:7])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                 token_mlp=token_mlp,
                                                se_layer=se_layer)

        self.upsample_2 = upsample(embed_dim * 8, embed_dim * 2)
        self.decoderlayer_2 = BasicStoformerLayer(dim=embed_dim * 4,
                                                output_dim=embed_dim * 4,
                                                input_resolution=(img_size // 2,
                                                                  img_size // 2),
                                                depth=depths[7],
                                                num_heads=num_heads[7],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=dec_dpr[sum(depths[5:7]):sum(depths[5:8])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_mlp=token_mlp,
                                                se_layer=se_layer)

        self.upsample_3 = upsample(embed_dim * 4, embed_dim)
        self.decoderlayer_3 = BasicStoformerLayer(dim=embed_dim * 2,
                                                output_dim=embed_dim * 2,
                                                input_resolution=(img_size,
                                                                  img_size),
                                                depth=depths[8],
                                                num_heads=num_heads[8],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=dec_dpr[sum(depths[5:8]):sum(depths[5:9])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_mlp=token_mlp,
                                                se_layer=se_layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x, mask=None):
        # Store original input dimensions
        input_shape = x.shape
        
        # Input Projection
        y = self.input_proj(x)
        y = self.pos_drop(y)
        
        # Encoder
        conv0 = self.encoderlayer_0(y, mask=mask) #128x128  32
        pool0 = self.dowsample_0(conv0)
        conv1 = self.encoderlayer_1(pool0, mask=mask) #64x64 64
        pool1 = self.dowsample_1(conv1)
        conv2 = self.encoderlayer_2(pool1, mask=mask) #32x32 128
        pool2 = self.dowsample_2(conv2)
        conv3 = self.encoderlayer_3(pool2, mask=mask) #16x16 256
        pool3 = self.dowsample_3(conv3)

        # Bottleneck
        conv4 = self.conv(pool3, mask=mask) #8x8 512

        # Decoder
        up0 = self.upsample_0(conv4) #16x16 256
        deconv0 = torch.cat([up0, conv3], -1) #16x16 512
        deconv0 = self.decoderlayer_0(deconv0, mask=mask) #16x16 512

        up1 = self.upsample_1(deconv0) #32x32 128
        deconv1 = torch.cat([up1, conv2], -1) #32x32 256
        deconv1 = self.decoderlayer_1(deconv1, mask=mask) #32x32 256

        up2 = self.upsample_2(deconv1) #64x64 64
        deconv2 = torch.cat([up2, conv1], -1) #64x64 128
        deconv2 = self.decoderlayer_2(deconv2, mask=mask) #64x64 128

        up3 = self.upsample_3(deconv2) #128x128 32
        deconv3 = torch.cat([up3, conv0], -1) #128x128 64
        deconv3 = self.decoderlayer_3(deconv3, mask=mask)

        # Output Projection
        y = self.output_proj(deconv3)
        
        # Ensure output has the same dimensions as input for residual connection
        if y.shape != input_shape:
            y = F.interpolate(y, size=(input_shape[2], input_shape[3]), mode='bilinear', align_corners=False)
        
        return x + y 

class Options():
    """docstring for Options"""

    def __init__(self):
        pass

    def init(self, parser):
        parser.add_argument('--batch_size', type=int, default=8, help='batch size')
        parser.add_argument('--nepoch', type=int, default=250, help='training epochs')
        parser.add_argument('--train_workers', type=int, default=4, help='train_dataloader workers')
        parser.add_argument('--eval_workers', type=int, default=1, help='eval_dataloader workers')
        parser.add_argument('--optimizer', type=str, default='adam', help='optimizer for training')
        parser.add_argument('--lr_initial', type=float, default=0.0002, help='initial learning rate')
        parser.add_argument('--LR_MIN', type=float, default=1e-6)
        parser.add_argument('--thre', type=int, default=50)
        parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
        parser.add_argument('--gpu', type=str, default='0,1', help='GPUs')
        parser.add_argument('--arch', type=str, default='Stoformer', help='archtechture')

        parser.add_argument('--save_dir', type=str, default='', help='save dir')
        parser.add_argument('--save_images', action='store_true', default=False)
        parser.add_argument('--env', type=str, default='_', help='env')
        parser.add_argument('--checkpoint', type=int, default=50, help='checkpoint')

        parser.add_argument('--norm_layer', type=str, default='nn.LayerNorm', help='normalize layer in transformer')
        parser.add_argument('--embed_dim', type=int, default=32, help='dim of emdeding features')
        parser.add_argument('--win_size', type=int, default=8, help='window size of self-attention')
        parser.add_argument('--token_projection', type=str, default='linear', help='linear/conv token projection')
        parser.add_argument('--token_mlp', type=str, default='leff', help='ffn/leff token mlp')
        parser.add_argument('--att_se', action='store_true', default=False, help='se after sa')

        parser.add_argument('--noiselevel', type=float, default=50)
        parser.add_argument('--use_grad_clip', action='store_true', default=False)

        # args for training
        parser.add_argument('--train_ps', type=int, default=128, help='patch size of training sample')
        parser.add_argument('--train_dir', type=str, default='', help='dir of train data')
        parser.add_argument('--val_dir', type=str, default='', help='dir of train data')
        parser.add_argument('--random_start', type=int, default=0, help='epochs for random shift')

        # args for testing
        parser.add_argument('--weights', type=str, default='', help='Path to trained weights')
        parser.add_argument('--test_workers', type=int, default=1, help='number of test works')
        parser.add_argument('--input_dir', type=str, default='', help='Directory of validation images')
        parser.add_argument('--result_dir', type=str, default='', help='Directory for results')
        parser.add_argument('--crop_size', type=int, default=256, help='crop size for testing')
        parser.add_argument('--overlap_size', type=int, default=30, help='overlap size for testing')
        return parser


def set_fast_inference_mode(model, enable=True):
    """Enable or disable fast inference mode for all transformer blocks in the model.
    
    This significantly speeds up validation by using a smart sampling strategy
    for window shifts instead of computing all possible combinations.
    
    Args:
        model: The Stoformer model
        enable: Boolean indicating whether to enable (True) or disable (False) fast mode
    """
    # Set the fast_inference flag on all StoTransformerBlock instances
    for module in model.modules():
        if isinstance(module, StoTransformerBlock):
            module.fast_inference = enable
    return model

def build_stoformer2(img_size=256, window_size=8, compat_mode=False):
    """Build Stoformer model for image restoration with stochastic window attention"""
    model = Stoformer(
        img_size=img_size,
        in_chans=3,
        embed_dim=32,
        depths=[1, 2, 8, 8, 2, 8, 8, 2, 1],
        num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
        win_size=window_size,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        patch_norm=True,
        use_checkpoint=False,
        token_mlp='leff',
        se_layer=False
    )
    # Add a model_type attribute to help identify this model variant later
    model.model_type = 'stoformer2'
    return model 