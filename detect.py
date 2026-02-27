"""
Grounded Iceberg Detection Inference Script
This script performs automated iceberg detection on Sentinel-1 SAR imagery using a pre-trained ResUNet model.
"""

import os
import sys
import gc
import glob
import re
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import rasterio
from rasterio.enums import Resampling
from scipy.ndimage import binary_dilation

# ==========================================
# 1. Model Architecture Definition
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super(ChannelAttention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_pool = self.global_avg_pool(x)
        max_pool = self.global_max_pool(x)
        avg_out = self.fc2(self.relu(self.fc1(avg_pool)))
        max_out = self.fc2(self.relu(self.fc1(max_pool)))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return x * out

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_pool, max_pool], dim=1)
        out = self.conv1(concat)
        out = self.sigmoid(out)
        return x * out

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention()
        
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class UNetWithResNet152(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNetWithResNet152, self).__init__()
        # Load pre-trained ResNet152
        resnet152 = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
        
        # Modify the first convolutional layer to accommodate the input channel count (e.g., single-channel SAR)
        resnet152.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # Extract the encoder portion
        self.encoder = nn.Sequential(*list(resnet152.children())[:-2])
        
        # Freeze the first few layers of the encoder to maintain feature stability
        for param in list(self.encoder.children())[:4]:
            param.requires_grad = False
            
        # Convolutional Block Attention Modules (CBAM)
        self.cbam1 = CBAM(64, reduction=32)
        self.cbam2 = CBAM(256, reduction=32)
        self.cbam3 = CBAM(512, reduction=32)
        self.cbam4 = CBAM(1024, reduction=32)
        self.cbam_center = CBAM(2048, reduction=32)
        
        # Decoder section
        self.decoder4 = self.conv_block(2048 + 1024, 512)
        self.decoder3 = self.conv_block(512 + 512, 256)
        self.decoder2 = self.conv_block(256 + 256, 128)
        self.decoder1 = self.conv_block(128 + 64, 64)
        
        # Output layer
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Encoder path
        enc1 = self.encoder[0:4](x)
        enc2 = self.encoder[4:5](enc1)
        enc3 = self.encoder[5:6](enc2)
        enc4 = self.encoder[6:7](enc3)
        center = self.encoder[7:8](enc4)
        
        # Apply attention mechanisms
        enc1 = self.cbam1(enc1)
        enc2 = self.cbam2(enc2)
        enc3 = self.cbam3(enc3)
        enc4 = self.cbam4(enc4)
        center = self.cbam_center(center)
        
        # Decoder path (Upsampling + Concatenation + Convolution)
        dec4 = self.decoder4(torch.cat([F.interpolate(center, size=enc4.shape[2:], mode='nearest'), enc4], dim=1))
        dec3 = self.decoder3(torch.cat([F.interpolate(dec4, size=enc3.shape[2:], mode='nearest'), enc3], dim=1))
        dec2 = self.decoder2(torch.cat([F.interpolate(dec3, size=enc2.shape[2:], mode='nearest'), enc2], dim=1))
        dec1 = self.decoder1(torch.cat([F.interpolate(dec2, size=enc1.shape[2:], mode='nearest'), enc1], dim=1))
        
        # Output and resize to original dimensions
        out = self.output(dec1)
        out = F.interpolate(out, size=(x.size(2), x.size(3)), mode='nearest')
        return out

# ==========================================
# 2. Data Processing Functions
# ==========================================
def resample_patch(patch, transform, patch_width, patch_height, new_width=1536, new_height=1536):
    """Upsample a single patch and its mask while preserving geographic coordinates."""
    mask = (~np.isnan(patch) & ~np.isinf(patch)).astype(np.float32)
    profile = {
        'driver': 'GTiff',
        'height': patch_height,
        'width': patch_width,
        'count': 2,
        'dtype': patch.dtype,
        'transform': transform,
        'crs': None,
        'nodata': None
    }
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**profile) as dst:
            dst.write(patch, 1)
            dst.write(mask, 2)
        with memfile.open() as src:
            data_resampled = src.read(1, out_shape=(new_height, new_width), resampling=Resampling.bilinear)
            mask_resampled = src.read(2, out_shape=(new_height, new_width), resampling=Resampling.nearest)
            new_transform = src.transform
    return data_resampled, mask_resampled, new_transform

def split_image(image, transform, ROWS, COLS):
    """Split the image into smaller patches, retaining geographic coordinates and NaN boundaries."""
    height, width = image.shape
    patch_height = height // ROWS
    patch_width = width // COLS
    patches, patch_transforms, nan_boundaries = [], [], []

    for i in range(ROWS):
        for j in range(COLS):
            y_start = i * patch_height
            y_end = height if i == ROWS - 1 else (i + 1) * patch_height
            x_start = j * patch_width
            x_end = width if j == COLS - 1 else (j + 1) * patch_width
            
            patch = image[y_start:y_end, x_start:x_end]
            patch_transform = transform * rasterio.transform.Affine.translation(x_start, y_start)

            # Calculate NaN boundaries (1 pixel around NaN values)
            nan_mask = np.isnan(patch) | np.isinf(patch)
            nan_boundary = binary_dilation(nan_mask, structure=np.ones((3, 3))) & ~nan_mask

            patches.append(patch)
            patch_transforms.append(patch_transform)
            nan_boundaries.append(nan_boundary.astype(np.uint8))

    return patches, patch_transforms, patch_height, patch_width, nan_boundaries

def stitch_patches(predictions, height, width, patch_height, patch_width, ROWS, COLS):
    """Stitch the predicted patches back into the original image dimensions."""
    full_mask = np.zeros((height, width), dtype=np.uint8)
    for idx, pred_np in enumerate(predictions):
        row = idx // COLS
        col = idx % COLS
        y_start = row * patch_height
        y_end = height if row == ROWS - 1 else (row + 1) * patch_height
        x_start = col * patch_width  
        x_end = width if col == COLS - 1 else (col + 1) * patch_width
        full_mask[y_start:y_end, x_start:x_end] = pred_np
    return full_mask

def predict_patches(model, patches, patch_transforms, nan_boundaries, device, image_height, image_width, patch_height, patch_width, ROWS, COLS, batch_size):
    """Perform batch prediction on patches, mask null regions, and handle out-of-boundary pixels."""
    model.eval()
    predictions = []
    transform = transforms.ToTensor()
    
    for batch_start in range(0, len(patches), batch_size):
        batch_end = min(batch_start + batch_size, len(patches))
        batch_patches = patches[batch_start:batch_end]
        batch_transforms = patch_transforms[batch_start:batch_end]
        batch_nan_boundaries = nan_boundaries[batch_start:batch_end]
        
        # Prepare batched inputs
        batch_tensors, batch_masks = [], []
        for patch, patch_transform in zip(batch_patches, batch_transforms):
            patch_resampled, mask_resampled, _ = resample_patch(patch, patch_transform, patch.shape[1], patch.shape[0])
            patch_resampled = np.nan_to_num(patch_resampled, nan=0.0, posinf=0.0, neginf=0.0)
            batch_tensors.append(transform(patch_resampled))
            batch_masks.append(mask_resampled)
        
        batch_tensors = torch.stack(batch_tensors).to(device)
        
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda'):
                # Batch inference
                preds = model(batch_tensors)
                preds = (torch.sigmoid(preds) > 0.5).float()
        
        # Process batched outputs
        for idx, (pred, mask_resampled, nan_boundary) in enumerate(zip(preds, batch_masks, batch_nan_boundaries)):
            # Convert to pixel values: 0 = background, 255 = iceberg
            pred_np = pred.squeeze().cpu().numpy().astype(np.uint8) * 255
            pred_np = pred_np * mask_resampled.astype(np.uint8)
            # Downsample back to original patch dimensions
            pred_np = np.array(Image.fromarray(pred_np).resize((patches[batch_start + idx].shape[1], patches[batch_start + idx].shape[0]), resample=Image.NEAREST))
            # Set pixels outside the NaN boundary to 0
            boundary_outer = binary_dilation(nan_boundary, structure=np.ones((3, 3)))
            pred_np[boundary_outer] = 0
            predictions.append(pred_np)
        
        print(f"Processed batch {batch_start//batch_size + 1}/{len(patches)//batch_size + 1}")
        
        # Clear VRAM for the batch
        del batch_tensors, preds
        torch.cuda.empty_cache()
    
    return stitch_patches(predictions, image_height, image_width, patch_height, patch_width, ROWS, COLS)

# ==========================================
# 3. Main Execution
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Automated Iceberg Detection using ResUNet")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input Sentinel-1 TIFF files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated binary masks")
    parser.add_argument("--model_path", type=str, default="models/unet_model.pth", help="Path to the pre-trained model weights")
    parser.add_argument("--patch_size", type=int, default=250, help="Target pixel size for image patching")
    parser.add_argument("--batch_size", type=int, default=35, help="Batch size for model inference")
    args = parser.parse_args()

    # Validate model weights
    if not os.path.exists(args.model_path):
        print(f"Error: Pre-trained weights not found at '{args.model_path}'")
        print("Please download the weights from our GitHub Releases page and place them in the correct directory.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | GPUs available: {torch.cuda.device_count()}")
    
    # Initialise and load model
    model = UNetWithResNet152(in_channels=1, out_channels=1)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    tiff_files = glob.glob(os.path.join(args.input_dir, "**", "*.tif"), recursive=True)
    if not tiff_files:
        print(f"No TIFF files found in {args.input_dir}")
        return

    for input_tif in tiff_files:
        filename = os.path.basename(input_tif)
        match = re.search(r"(\d{8}T\d{6})", filename)
        if not match:
            continue
        timestamp = match.group(1)
    
        rel_dir = os.path.relpath(os.path.dirname(input_tif), args.input_dir)
        output_subdir = os.path.join(args.output_dir, rel_dir)
        os.makedirs(output_subdir, exist_ok=True)
        output_path = os.path.join(output_subdir, f"{timestamp}_mask.tif")
    
        if os.path.exists(output_path):
            print(f"Skipping {filename}: Output already exists")
            continue
       
        # Read the raw imagery
        with rasterio.open(input_tif) as src:
            image = src.read(1)
            transform_geo = src.transform
            crs = src.crs
            image_height, image_width = image.shape
        
        # Automatically calculate row and column patch counts
        ROWS = max(1, int(np.ceil(image_height / args.patch_size)))
        COLS = max(1, int(np.ceil(image_width / args.patch_size)))
        print(f"\nProcessing {filename}: {ROWS} × {COLS} patches")
        
        patches, patch_transforms, patch_height, patch_width, nan_boundaries = split_image(image, transform_geo, ROWS, COLS)
        full_mask = predict_patches(model, patches, patch_transforms, nan_boundaries, device, image_height, image_width, patch_height, patch_width, ROWS, COLS, args.batch_size)
        
        # Save as GeoTIFF
        with rasterio.open(
            output_path, 'w', driver='GTiff',
            height=full_mask.shape[0], width=full_mask.shape[1], count=1,
            dtype=full_mask.dtype, crs=crs, transform=transform_geo
        ) as dst:
            dst.write(full_mask, 1)
        
        print(f"Saved mask to {output_path}")
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()