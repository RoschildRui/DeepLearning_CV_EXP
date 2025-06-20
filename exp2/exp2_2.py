import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def calculate_output_shape(input_size, kernel_size, padding, stride):
    """
    calculate the theoretical formula for the output shape of the convolution layer
    output size = (input size + 2×padding - kernel size) / stride + 1
    """
    output_size = (input_size + 2 * padding - kernel_size) // stride + 1
    return output_size

def verify_conv_shapes(img_path):
    input_height, input_width = 28, 28  # input image size
    kernel_size = 3                     # kernel size
    in_channels = 1                     # input channels
    out_channels = 1                    # output channels
    
    img = Image.open(img_path).convert('L').resize((input_width, input_height))
    img_np = np.array(img, dtype=np.float32) / 255.0
    # (batch_size, channels, height, width)
    input_tensor = torch.tensor(img_np).unsqueeze(0).unsqueeze(0)
    print(f"loaded image path: {img_path}")
    print(f"input tensor shape: {input_tensor.shape}")
    print(f"input image size: {input_height} × {input_width}")
    print(f"kernel size: {kernel_size} × {kernel_size}")
    print()
    
    combinations = [
        {"padding": 0, "stride": 1, "name": "combination 1: no padding, stride 1"},
        {"padding": 1, "stride": 2, "name": "combination 2: padding 1, stride 2"},
        {"padding": 2, "stride": 3, "name": "combination 3: padding 2, stride 3"}
    ]
    
    results = []
    
    for i, combo in enumerate(combinations, 1):
        padding = combo["padding"]
        stride = combo["stride"]
        name = combo["name"]
        
        print(f"{name}")
        print(f"parameters: padding={padding}, stride={stride}")
        
        theoretical_height = calculate_output_shape(input_height, kernel_size, padding, stride)
        theoretical_width = calculate_output_shape(input_width, kernel_size, padding, stride)
        
        print(f"theoretical calculation:")
        print(f"  output height = ({input_height} + 2×{padding} - {kernel_size}) ÷ {stride} + 1 = {theoretical_height}")
        print(f"  output width = ({input_width} + 2×{padding} - {kernel_size}) ÷ {stride} + 1 = {theoretical_width}")
        print(f"  theoretical output shape: {theoretical_height} × {theoretical_width}")
        
        conv_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride
        )
        
        with torch.no_grad():
            output_tensor = conv_layer(input_tensor)
        
        actual_shape = output_tensor.shape
        actual_height = actual_shape[2]
        actual_width = actual_shape[3]
        
        print(f"actual validation:")
        print(f"  actual output tensor shape: {actual_shape}")
        print(f"  actual output shape: {actual_height} × {actual_width}")
        
        height_match = theoretical_height == actual_height
        width_match = theoretical_width == actual_width
        
        print(f"validation result:")
        print(f"  height match: {height_match} (theoretical: {theoretical_height}, actual: {actual_height})")
        print(f"  width match: {width_match} (theoretical: {theoretical_width}, actual: {actual_width})")
        print(f"  overall match: {height_match and width_match}")
        
        results.append({
            "combination": name,
            "padding": padding,
            "stride": stride,
            "theoretical": (theoretical_height, theoretical_width),
            "actual": (actual_height, actual_width),
            "match": height_match and width_match
        })
        
        print("-" * 50)
    
    print("\n" + "=" * 60)
    print("experiment results summary")
    print("=" * 60)
    
    for result in results:
        status = "✓ match" if result["match"] else "✗ not match"
        print(f"{result['combination']}: {status}")
        print(f"  theoretical: {result['theoretical']}, actual: {result['actual']}")
    
    all_match = all(result["match"] for result in results)
    print(f"\nall combinations validation result: {'✓ all match' if all_match else '✗ some not match'}")
    
    if all_match:
        print("conclusion: the formula for calculating the output shape of the convolution layer is correct!")
    else:
        print("conclusion: there are calculation errors, need to check further.")
    
    return results

def visualize_conv_effects(img_path):
    print("\n" + "=" * 60)
    print("visualize the effects of different padding and stride")
    print("=" * 60)
    
    input_size = 512
    img = Image.open(img_path).convert('L').resize((input_size, input_size))
    img_np = np.array(img, dtype=np.float32) / 255.0
    input_tensor = torch.tensor(img_np).unsqueeze(0).unsqueeze(0)
    input_image = input_tensor
    # input_image = torch.zeros(1, 1, input_size, input_size)
    # input_image[0, 0, 2:6, 2:6] = 1.0  # center square
    # input_image[0, 0, 3:5, 3:5] = 2.0  # inner square
    # kernel = torch.tensor([[-1, -1, -1],
    #                       [-1,  8, -1],
    #                       [-1, -1, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    # 3×3 sharpen kernel, can enhance edges and preserve image brightness
    kernel = torch.tensor([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]], dtype=torch.float32,).unsqueeze(0).unsqueeze(0)
    
    combinations = [
        {"padding": 0, "stride": 1},
        {"padding": 1, "stride": 2},
        {"padding": 2, "stride": 3}
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    axes[0, 0].imshow(input_image[0, 0].numpy(), cmap='gray')
    axes[0, 0].set_title(f'input image\n{input_size}×{input_size}')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(kernel[0, 0].numpy(), cmap='gray')
    axes[1, 0].set_title('kernel\n3×3')
    axes[1, 0].axis('off')
    
    for i, combo in enumerate(combinations):
        padding = combo["padding"]
        stride = combo["stride"]
        
        conv = nn.Conv2d(1, 1, 3, padding=padding, stride=stride, bias=False)
        conv.weight.data = kernel
        
        with torch.no_grad():
            output = conv(input_image)
        
        theoretical_size = calculate_output_shape(input_size, 3, padding, stride)
        actual_size = output.shape[2]
        
        axes[0, i+1].imshow(output[0, 0].numpy(), cmap='gray')
        axes[0, i+1].set_title(f'combination {i+1}\npadding={padding}, stride={stride}')
        axes[0, i+1].axis('off')
        
        axes[1, i+1].text(0.5, 0.5, f'theoretical: {theoretical_size}×{theoretical_size}\nactual: {actual_size}×{actual_size}\nmatch: {"✓" if theoretical_size == actual_size else "✗"}', 
                         ha='center', va='center', transform=axes[1, i+1].transAxes, fontsize=12)
        axes[1, i+1].axis('off')
    
    plt.tight_layout()
    plt.savefig('exp2_padding_stride_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("visualization results saved to: exp2/results/exp2/exp2_padding_stride_visualization.png")

if __name__ == "__main__":
    import os
    img_path = 'love.jpg'
    os.makedirs('code/exp2/results/exp2', exist_ok=True)
    results = verify_conv_shapes(img_path)
    visualize_conv_effects(img_path)
    print("\nexp2 completed!") 