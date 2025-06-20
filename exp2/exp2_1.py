import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"cannot load image: {image_path}")
    return image

def create_edge_kernels():
    """创建不同方向的边缘检测卷积核"""
    # 水平边缘检测核（检测水平方向的边缘）
    horizontal_kernel = torch.tensor([
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1]
    ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # 竖直边缘检测核（检测竖直方向的边缘）
    vertical_kernel = torch.tensor([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # 对角边缘检测核1（左上到右下方向）
    diagonal_kernel1 = torch.tensor([
        [-1, -1,  0],
        [-1,  0,  1],
        [ 0,  1,  1]
    ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # 对角边缘检测核2（右上到左下方向）
    diagonal_kernel2 = torch.tensor([
        [ 0, -1, -1],
        [ 1,  0, -1],
        [ 1,  1,  0]
    ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    return {
        'horizontal': horizontal_kernel,
        'vertical': vertical_kernel,
        'diagonal1': diagonal_kernel1,
        'diagonal2': diagonal_kernel2
    }

def apply_convolution(image, kernel):
    image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    output = F.conv2d(image_tensor, kernel, padding=1)
    output_np = output.squeeze().detach().numpy()
    return output_np

def normalize_output(output):
    output = np.abs(output)
    output = (output / output.max() * 255).astype(np.uint8)
    return output

def main():
    image_path = 'love.jpg'
    print(f"loading image: {image_path}")
    try:
        original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        processed_image = load_image(image_path)
        print(f"image size: {processed_image.shape}")
    except ValueError as e:
        print(e)
        return
    
    kernels = create_edge_kernels()
    
    results = {}
    for kernel_name, kernel in kernels.items():
        print(f"applying {kernel_name} edge detection kernel...")
        output = apply_convolution(processed_image, kernel)
        results[kernel_name] = normalize_output(output)
    
    plt.figure(figsize=(15, 12))
    plt.subplot(2, 3, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('original image (RGB)')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(results['horizontal'], cmap='gray')
    plt.title('horizontal edge detection')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(results['vertical'], cmap='gray')
    plt.title('vertical edge detection')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(results['diagonal1'], cmap='gray')
    plt.title('diagonal edge detection 1\n(top-left to bottom-right)')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(results['diagonal2'], cmap='gray')
    plt.title('diagonal edge detection 2\n(top-right to bottom-left)')
    plt.axis('off')
    
    combined_edges = np.maximum.reduce([
        results['horizontal'], 
        results['vertical'], 
        results['diagonal1'], 
        results['diagonal2']
    ])
    
    plt.subplot(2, 3, 6)
    plt.imshow(combined_edges, cmap='gray')
    plt.title('combined edge detection')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('edge_detection_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nused kernels:")
    for kernel_name, kernel in kernels.items():
        print(f"\n{kernel_name} kernel:")
        print(kernel.squeeze().numpy())
    print("\nedge detection completed! results saved to 'edge_detection_results.png'")

if __name__ == "__main__":
    main()
