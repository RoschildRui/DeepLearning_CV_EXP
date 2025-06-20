import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch
from torch import nn, optim
from utils import get_device, seed_everything
from dataset import load_fashion_mnist
from model import SoftmaxRegression
from utils import train
from vis import plot_results, show_predictions

def main():
    seed = 20250520
    seed_everything(seed)
    device = get_device()
    print(f"Use device: {device}")
    
    train_loader, test_loader = load_fashion_mnist(batch_size=64)
    # Fashion-MNIST image size: 28x28
    input_size = 28 * 28  
    num_classes = 10
    
    model = SoftmaxRegression(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-2)
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    train_losses, test_accuracies = train(
        model, train_loader, test_loader, criterion, optimizer, epochs=20, device=device
    )
    
    plot_results(train_losses, test_accuracies,title='softmax regression')
    show_predictions(model, test_loader, num_images=5, device=device)

if __name__ == "__main__":
    main() 