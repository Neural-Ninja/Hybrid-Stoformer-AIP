import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import OrderedDict

# Import Stoformer2 components
from Models.stoformer2 import (
    StoTransformerBlock, 
    BasicStoformerLayer, 
    Upsample, 
    InputProj, 
    OutputProj,
    build_stoformer2,
    set_fast_inference_mode
)

class CNNFeatureExtractor(nn.Module):
    """CNN backbone for extracting multi-scale features."""
    
    def __init__(self, backbone='resnet34', pretrained=True):
        super(CNNFeatureExtractor, self).__init__()
        
        # Select backbone
        if backbone == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
            feature_dims = [64, 128, 256, 512]
        elif backbone == 'resnet34':
            base_model = models.resnet34(pretrained=pretrained)
            feature_dims = [64, 128, 256, 512]
        elif backbone == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
            feature_dims = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Extract layers
        self.conv1 = base_model.conv1  # 64 channels
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        
        self.layer1 = base_model.layer1  # 64/256 channels
        self.layer2 = base_model.layer2  # 128/512 channels
        self.layer3 = base_model.layer3  # 256/1024 channels
        self.layer4 = base_model.layer4  # 512/2048 channels
        
        self.feature_dims = feature_dims
        
    def forward(self, x):
        # Initial processing
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c0 = x  # First feature, 64 channels, 1/2 resolution
        
        x = self.maxpool(x)
        
        # Extract features at different scales
        c1 = self.layer1(x)      # 1/4 resolution
        c2 = self.layer2(c1)     # 1/8 resolution
        c3 = self.layer3(c2)     # 1/16 resolution
        c4 = self.layer4(c3)     # 1/32 resolution
        
        return [c0, c1, c2, c3, c4]

class FeatureFusionModule(nn.Module):
    """Module to fuse CNN features for transformer input."""
    
    def __init__(self, cnn_dims, transformer_dim):
        super(FeatureFusionModule, self).__init__()
        
        # For ResNet, the feature dimensions are for the four main layers
        # But we need to handle c0 (after initial conv) which has different dimensions
        # Typically 64 channels for most ResNet variants
        self.init_conv_proj = nn.Conv2d(64, transformer_dim, kernel_size=1)
        
        # Create projection layers for each layer's output
        self.layer1_proj = nn.Conv2d(cnn_dims[0], transformer_dim, kernel_size=1)  # typically 64 or 256
        self.layer2_proj = nn.Conv2d(cnn_dims[1], transformer_dim, kernel_size=1)  # typically 128 or 512
        self.layer3_proj = nn.Conv2d(cnn_dims[2], transformer_dim, kernel_size=1)  # typically 256 or 1024
        self.layer4_proj = nn.Conv2d(cnn_dims[3], transformer_dim, kernel_size=1)  # typically 512 or 2048
        
    def forward(self, cnn_features):
        # Project and resize all features to the same dimensions
        projected_features = []
        
        # Make sure we have all 5 features: c0, c1, c2, c3, c4
        if len(cnn_features) != 5:
            raise ValueError(f"Expected 5 CNN features, got {len(cnn_features)}")
            
        # Handle each feature individually with its correct projection 
        # c0 - Initial convolution output
        x0 = self.init_conv_proj(cnn_features[0])
        
        # c1 - Layer 1 output
        x1 = self.layer1_proj(cnn_features[1])
        
        # c2 - Layer 2 output
        x2 = self.layer2_proj(cnn_features[2])
        
        # c3 - Layer 3 output
        x3 = self.layer3_proj(cnn_features[3])
        
        # c4 - Layer 4 output
        x4 = self.layer4_proj(cnn_features[4])
        
        # Choose the smallest feature map (c4) as the target size
        target_size = cnn_features[4].shape[2:]
        
        # Resize all projected features to the target size
        p0 = F.interpolate(x0, size=target_size, mode='bilinear', align_corners=False)
        p1 = F.interpolate(x1, size=target_size, mode='bilinear', align_corners=False)
        p2 = F.interpolate(x2, size=target_size, mode='bilinear', align_corners=False)
        p3 = F.interpolate(x3, size=target_size, mode='bilinear', align_corners=False)
        p4 = x4  # Already at target size
        
        # Collect all projections
        projected_features = [p0, p1, p2, p3, p4]
            
        # Sum all features
        fused_feature = sum(projected_features)
        return fused_feature

class HybridStoformer(nn.Module):
    """Hybrid CNN-Stoformer model for image deblurring."""
    
    def __init__(self, 
                 in_chans=3,
                 embed_dim=32, 
                 depths=[1, 2, 8, 8, 2, 8, 8, 2, 1], 
                 num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
                 win_size=8, 
                 mlp_ratio=4.,
                 cnn_backbone='resnet34',
                 pretrained=True,
                 img_size=256):
        super(HybridStoformer, self).__init__()
        
        # CNN Feature Extractor
        self.cnn_extractor = CNNFeatureExtractor(backbone=cnn_backbone, pretrained=pretrained)
        cnn_feature_dims = self.cnn_extractor.feature_dims
        
        # Determine bottleneck dimensions based on input size
        bottleneck_dim = embed_dim * 16  # Same as Stoformer2
        
        # Feature Fusion Module
        self.feature_fusion = FeatureFusionModule(
            cnn_dims=cnn_feature_dims, 
            transformer_dim=bottleneck_dim
        )
        
        # Create Stoformer2 model without input projection (we'll use CNN features instead)
        stoformer = build_stoformer2(img_size=img_size, window_size=win_size)
        
        # Get Stoformer2 components we want to reuse
        # Skip input projection, start from conv layer
        self.conv = stoformer.conv  # bottleneck
        
        # Decoder components
        self.upsample_0 = stoformer.upsample_0
        self.decoderlayer_0 = stoformer.decoderlayer_0
        self.upsample_1 = stoformer.upsample_1
        self.decoderlayer_1 = stoformer.decoderlayer_1
        self.upsample_2 = stoformer.upsample_2
        self.decoderlayer_2 = stoformer.decoderlayer_2
        self.upsample_3 = stoformer.upsample_3
        self.decoderlayer_3 = stoformer.decoderlayer_3
        
        # Output projection
        self.output_proj = stoformer.output_proj
        
        # Skip connection handling - create convolutions that map CNN features to Stoformer2 expected dimensions
        # These dimensions must match exactly what the stoformer2 decoder expects for skip connections
        if cnn_backbone in ['resnet18', 'resnet34']:
            # The key insight is to map CNN features to the dimensions expected in Stoformer2's forward method
            # ResNet18/34 feature dimensions: [64 (c0), 64 (c1), 128 (c2), 256 (c3), 512 (c4)]
            # Map c3 to embed_dim*8 (256) for decoderlayer_0
            self.skip_conv3 = nn.Conv2d(256, embed_dim * 8, kernel_size=1)  
            # Map c2 to embed_dim*4 (128) for decoderlayer_1
            self.skip_conv2 = nn.Conv2d(128, embed_dim * 4, kernel_size=1)  
            # Map c1 to embed_dim*2 (64) for decoderlayer_2
            self.skip_conv1 = nn.Conv2d(64, embed_dim * 2, kernel_size=1)   
            # Map c0 to embed_dim (32) for decoderlayer_3
            self.skip_conv0 = nn.Conv2d(64, embed_dim, kernel_size=1)       
        elif cnn_backbone == 'resnet50':
            # ResNet50 feature dimensions: [64 (c0), 256 (c1), 512 (c2), 1024 (c3), 2048 (c4)]
            # Map c3 to embed_dim*8 (256) for decoderlayer_0
            self.skip_conv3 = nn.Conv2d(1024, embed_dim * 8, kernel_size=1) 
            # Map c2 to embed_dim*4 (128) for decoderlayer_1
            self.skip_conv2 = nn.Conv2d(512, embed_dim * 4, kernel_size=1)  
            # Map c1 to embed_dim*2 (64) for decoderlayer_2
            self.skip_conv1 = nn.Conv2d(256, embed_dim * 2, kernel_size=1)  
            # Map c0 to embed_dim (32) for decoderlayer_3
            self.skip_conv0 = nn.Conv2d(64, embed_dim, kernel_size=1)       
        
    def forward(self, x):
        # Store original input for residual connection
        input_img = x
        
        # Extract CNN features
        cnn_features = self.cnn_extractor(x)
        
        # Get spatial dimensions for each feature level
        B, _, _, _ = x.shape
        H_0, W_0 = cnn_features[0].shape[2:] # 1/2 of original (c0)
        H_1, W_1 = cnn_features[1].shape[2:] # 1/4 of original (c1)
        H_2, W_2 = cnn_features[2].shape[2:] # 1/8 of original (c2)
        H_3, W_3 = cnn_features[3].shape[2:] # 1/16 of original (c3)
        H_4, W_4 = cnn_features[4].shape[2:] # 1/32 of original (c4)
        
        # Fuse CNN features as bottleneck features
        bottleneck_feature = self.feature_fusion(cnn_features)
        
        # Store actual dimensions before flattening
        H, W = bottleneck_feature.shape[2:]
        
        # Convert to transformer format (B, C, H, W) -> (B, H*W, C)
        bottleneck_tokens = bottleneck_feature.flatten(2).transpose(1, 2)
        
        # Store the spatial dimensions as attributes directly on the tensor
        # This is critical! The transformer needs to know the 2D structure
        bottleneck_tokens.bottleneck_shape = (H, W)
        
        # Process with transformer bottleneck
        conv4 = self.conv(bottleneck_tokens)
        conv4.bottleneck_shape = bottleneck_tokens.bottleneck_shape
        
        # Prepare skip connections from CNN features
        # The key fix: Match dimensions to what the decoder layers expect
        # In stoformer2.py, decoderlayer_0 expects dim=embed_dim*16 with skip of embed_dim*8
        # decoderlayer_1 expects dim=embed_dim*8 with skip of embed_dim*4, etc.
        
        # These skip connections must match the dimensions expected by Stoformer2
        # The order must follow the same pattern as in the original Stoformer2
        skip3 = self.skip_conv3(cnn_features[3])  # 1/16 resolution - embed_dim*8 (256)
        skip2 = self.skip_conv2(cnn_features[2])  # 1/8 resolution - embed_dim*4 (128) 
        skip1 = self.skip_conv1(cnn_features[1])  # 1/4 resolution - embed_dim*2 (64)
        skip0 = self.skip_conv0(cnn_features[0])  # 1/2 resolution - embed_dim (32)
        
        # Convert skip connections to token format
        skip3_tokens = skip3.flatten(2).transpose(1, 2)  # For decoderlayer_0
        skip2_tokens = skip2.flatten(2).transpose(1, 2)  # For decoderlayer_1
        skip1_tokens = skip1.flatten(2).transpose(1, 2)  # For decoderlayer_2
        skip0_tokens = skip0.flatten(2).transpose(1, 2)  # For decoderlayer_3
        
        # Store the spatial dimensions for each token set - this is critical!
        skip3_tokens.bottleneck_shape = (H_3, W_3)
        skip2_tokens.bottleneck_shape = (H_2, W_2)
        skip1_tokens.bottleneck_shape = (H_1, W_1)
        skip0_tokens.bottleneck_shape = (H_0, W_0)
        
        # Decoder path (follows the same pattern as in stoformer2.py)
        up0 = self.upsample_0(conv4)  # 1/16 resolution, embed_dim*8 (256)
        up0.bottleneck_shape = skip3_tokens.bottleneck_shape  # Use skip3's shape
        
        deconv0 = torch.cat([up0, skip3_tokens], -1)  # concat to embed_dim*16 (512)
        deconv0.bottleneck_shape = skip3_tokens.bottleneck_shape
        
        deconv0 = self.decoderlayer_0(deconv0)  # Process concat result
        deconv0.bottleneck_shape = skip3_tokens.bottleneck_shape  # Preserve shape info
        
        up1 = self.upsample_1(deconv0)  # 1/8 resolution, embed_dim*4 (128)
        up1.bottleneck_shape = skip2_tokens.bottleneck_shape  # Use skip2's shape
        
        deconv1 = torch.cat([up1, skip2_tokens], -1)  # concat to embed_dim*8 (256)
        deconv1.bottleneck_shape = skip2_tokens.bottleneck_shape
        
        deconv1 = self.decoderlayer_1(deconv1)  # Process concat result
        deconv1.bottleneck_shape = skip2_tokens.bottleneck_shape
        
        up2 = self.upsample_2(deconv1)  # 1/4 resolution, embed_dim*2 (64)
        up2.bottleneck_shape = skip1_tokens.bottleneck_shape  # Use skip1's shape
        
        deconv2 = torch.cat([up2, skip1_tokens], -1)  # concat to embed_dim*4 (128)
        deconv2.bottleneck_shape = skip1_tokens.bottleneck_shape
        
        deconv2 = self.decoderlayer_2(deconv2)  # Process concat result
        deconv2.bottleneck_shape = skip1_tokens.bottleneck_shape
        
        up3 = self.upsample_3(deconv2)  # 1/2 resolution, embed_dim (32)
        up3.bottleneck_shape = skip0_tokens.bottleneck_shape  # Use skip0's shape
        
        deconv3 = torch.cat([up3, skip0_tokens], -1)  # concat to embed_dim*2 (64)
        deconv3.bottleneck_shape = skip0_tokens.bottleneck_shape
        
        deconv3 = self.decoderlayer_3(deconv3)  # Process concat result
        deconv3.bottleneck_shape = skip0_tokens.bottleneck_shape  # Add this line
        
        # Output projection
        y = self.output_proj(deconv3)
        
        # Ensure output has the same dimensions as input for residual connection
        if y.shape != input_img.shape:
            y = F.interpolate(y, size=(input_img.shape[2], input_img.shape[3]), 
                             mode='bilinear', align_corners=False)
        
        return input_img + y
    
    def set_fast_inference_mode(self, enable=True):
        """Enable or disable fast inference mode for all transformer blocks.
        
        Fast inference mode skips the calculation of certain spatial shifts, significantly
        speeding up inference (7-10x faster) with minimal quality impact.
        
        Args:
            enable (bool): Whether to enable fast inference mode
        """
        # Iterate through all modules and set fast_inference flag on StoTransformerBlock instances
        for module in self.modules():
            if hasattr(module, 'fast_inference'):
                module.fast_inference = enable
                
def build_hybrid_stoformer(img_size=256, window_size=8, cnn_backbone='resnet34', pretrained=True):
    """Build Hybrid CNN-Stoformer model for image restoration."""
    model = HybridStoformer(
        in_chans=3,
        embed_dim=32,
        depths=[1, 2, 8, 8, 2, 8, 8, 2, 1],
        num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
        win_size=window_size,
        mlp_ratio=4.,
        cnn_backbone=cnn_backbone,
        pretrained=pretrained,
        img_size=img_size
    )
    # Add model type attribute to help identify this model variant
    model.model_type = 'hybrid_stoformer'
    return model 