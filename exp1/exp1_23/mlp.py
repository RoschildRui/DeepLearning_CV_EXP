import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch
from torch import nn, optim
from dataset import load_fashion_mnist
from model import MLP
from utils import train, get_device, seed_everything
from vis import plot_results, show_predictions

def compare_hidden_designs(input_size, hidden_sizes_list, lr_list, num_classes, train_loader, test_loader, device="cpu"):
    results = []
    
    for i, hidden_sizes in enumerate(hidden_sizes_list):
        print(f"\nTraining MLP model #{i+1} with hidden layer design: {hidden_sizes}")
        
        model = MLP(input_size, hidden_sizes, num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr_list[i])
        train_losses, test_accuracies = train(
            model, train_loader, test_loader, criterion, optimizer, epochs=5, device=device
        )
        
        results.append({
            'hidden_layer_design': hidden_sizes,
            'learning_rate': lr_list[i],
            'final_accuracy': test_accuracies[-1],
            'train_losses': train_losses,
            'test_accuracies': test_accuracies
        })
    
    plt.figure(figsize=(12, 6))
    for result in results:
        plt.plot(result['test_accuracies'], label=f"hidden layer design: {result['hidden_layer_design']}")
    
    plt.title('Comparison of model accuracy with different hidden layer designs')
    plt.xlabel('epoch')
    plt.ylabel('accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print("\nfinal test accuracy:")
    for result in results:
        print(f"hidden layer design: {result['hidden_layer_design']}, learning rate: {result['learning_rate']}, accuracy: {result['final_accuracy']:.2f}%")

def main():
    seed = 42
    seed_everything(seed)
    device = get_device()
    print(f"Use device: {device}")
    
    train_loader, test_loader = load_fashion_mnist(batch_size=64)
    
    # Fashion-MNIST image size: 28x28
    input_size = 28 * 28  
    # 3 hidden layers
    hidden_sizes = [512, 256, 128] 
    num_classes = 10
    model = MLP(input_size, hidden_sizes, num_classes)
    print(model)

    learning_rate = 1e-3
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    # train_losses, test_accuracies = train(
    #     model, train_loader, test_loader, criterion, optimizer, epochs=20, device=device
    # )
    
    # plot_results(train_losses, test_accuracies,title='mlp')
    
    # show_predictions(model, test_loader, num_images=5, device=device)
    
    hidden_sizes_list = [
        [128],                 # 1 hidden layer
        [256, 128],            # 2 hidden layers
        [512, 256, 128],       # 3 hidden layers
        [1024, 512, 256, 128], # 4 hidden layers
        [512, 1024, 256, 128]  # Inversion layer design
    ]
    learning_rate_list = [1e-3, 1e-3, 1e-3, 1e-3, 1e-3]
    compare_hidden_designs(input_size, hidden_sizes_list, learning_rate_list, num_classes, train_loader, test_loader, device)

if __name__ == "__main__":
    main() 