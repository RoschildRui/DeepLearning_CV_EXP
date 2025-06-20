import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def load_and_preprocess_image(img_path, size=(224, 224)):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(size)
    img_np = np.array(img, dtype=np.float32) / 255.0
    # transformer to (batch_size, channels, height, width)
    img_tensor = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0)
    return img_tensor, img_np

def create_feature_maps(input_tensor, num_channels=8):
    """
    create multi-channel feature maps, simulate the output of the network intermediate layer
    use different convolution kernels to generate different feature maps
    """
    batch_size, in_channels, height, width = input_tensor.shape
    
    # create multiple different convolution kernels to generate feature maps
    conv_layers = []
    for i in range(num_channels):
        conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, bias=False)
        if i % 4 == 0:  # horizontal edge detection
            conv.weight.data = torch.tensor([[[[-1, -1, -1], [2, 2, 2], [-1, -1, -1]]]], dtype=torch.float32).repeat(1, in_channels, 1, 1)
        elif i % 4 == 1:  # vertical edge detection
            conv.weight.data = torch.tensor([[[[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]]]], dtype=torch.float32).repeat(1, in_channels, 1, 1)
        elif i % 4 == 2:  # diagonal edge detection
            conv.weight.data = torch.tensor([[[[2, -1, -1], [-1, 2, -1], [-1, -1, 2]]]], dtype=torch.float32).repeat(1, in_channels, 1, 1)
        else:  # blur/smooth
            conv.weight.data = torch.tensor([[[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]], dtype=torch.float32).repeat(1, in_channels, 1, 1) / 9
        conv_layers.append(conv)
    
    feature_maps = []
    with torch.no_grad():
        for conv in conv_layers:
            feature_map = conv(input_tensor)
            feature_maps.append(feature_map)
    
    multi_channel_features = torch.cat(feature_maps, dim=1)
    return multi_channel_features

def apply_1x1_conv_channel_reduction(input_tensor, reduction_factor=2):
    """
    use 1x1 convolution kernel to reduce the number of channels
    """
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels = in_channels // reduction_factor
    
    print(f"input feature map shape: {input_tensor.shape}")
    print(f"input channels: {in_channels}")
    print(f"target output channels: {out_channels}")
    
    conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
    nn.init.xavier_uniform_(conv1x1.weight)
    with torch.no_grad():
        output_tensor = conv1x1(input_tensor)
    
    print(f"output feature map shape: {output_tensor.shape}")
    print(f"output channels: {output_tensor.shape[1]}")
    print(f"channel reduction ratio: {in_channels}/{output_tensor.shape[1]} = {in_channels/output_tensor.shape[1]:.1f}x")
    
    return output_tensor, conv1x1

def visualize_channel_reduction(original_img, input_features, output_features, save_path):
    batch_size, in_channels, height, width = input_features.shape
    _, out_channels, _, _ = output_features.shape
    
    max_display_channels = 8
    in_display_channels = min(in_channels, max_display_channels)
    out_display_channels = min(out_channels, max_display_channels)
    
    fig, axes = plt.subplots(3, max_display_channels, figsize=(20, 12))
    for i in range(max_display_channels):
        if i == 0:
            axes[0, i].imshow(original_img)
            axes[0, i].set_title('Original Image', fontsize=10)
        else:
            axes[0, i].axis('off')
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
    
    # display input feature maps (before reduction)
    for i in range(max_display_channels):
        if i < in_display_channels:
            feature_map = input_features[0, i].numpy()
            axes[1, i].imshow(feature_map, cmap='viridis')
            axes[1, i].set_title(f'Input Ch.{i+1}\n({in_channels} channels)', fontsize=10)
        else:
            axes[1, i].axis('off')
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
    
    # display output feature maps (after reduction)
    for i in range(max_display_channels):
        if i < out_display_channels:
            feature_map = output_features[0, i].numpy()
            axes[2, i].imshow(feature_map, cmap='viridis')
            axes[2, i].set_title(f'Output Ch.{i+1}\n({out_channels} channels)', fontsize=10)
        else:
            axes[2, i].axis('off')
        axes[2, i].set_xticks([])
        axes[2, i].set_yticks([])
    
    fig.text(0.02, 0.83, 'Original', rotation=90, va='center', fontsize=12, fontweight='bold')
    fig.text(0.02, 0.5, f'Before\n(1×1 Conv)', rotation=90, va='center', fontsize=12, fontweight='bold')
    fig.text(0.02, 0.17, f'After\n(1×1 Conv)', rotation=90, va='center', fontsize=12, fontweight='bold')
    
    plt.suptitle(f'1×1 Convolution Channel Reduction: {in_channels} → {out_channels} channels', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, top=0.93)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    os.makedirs('results/exp3', exist_ok=True)
        
    img_path = 'love.jpg'
    input_tensor, original_img = load_and_preprocess_image(img_path)
    print(f"image shape: {input_tensor.shape}")
    
    multi_channel_features = create_feature_maps(input_tensor, num_channels=8)
    print(f"generated feature map shape: {multi_channel_features.shape}")
    
    reduced_features, conv1x1_layer = apply_1x1_conv_channel_reduction(multi_channel_features, reduction_factor=2)
    
    save_path = 'results/exp3/task3_1x1_conv_channel_reduction.png'
    visualize_channel_reduction(original_img, multi_channel_features, reduced_features, save_path)
    print(f"visualization results saved to: {save_path}")
    print(f"\nexp3 completed! results saved to results/exp3/")

if __name__ == "__main__":
    main() 