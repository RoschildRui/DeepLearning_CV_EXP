import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from collections import OrderedDict
from utils import generate_report4exp2_4 as generate_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"use device: {device}")

class LeNet(nn.Module):
    def __init__(self, kernel_size=5, padding=0, stride=1):
        super(LeNet, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        
        self.conv1 = nn.Conv2d(1, 6, kernel_size=kernel_size, padding=padding, stride=stride)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=kernel_size, padding=padding, stride=stride)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc_input_size = self._get_fc_input_size()
        
        self.fc1 = nn.Linear(self.fc_input_size, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
    
    def _get_fc_input_size(self):
        # calculate the size of the feature map after convolution and pooling
        # input size: 28x28
        x = 28
        
        # first convolution layer
        x = (x + 2 * self.padding - self.kernel_size) // self.stride + 1
        # first pooling layer
        x = x // 2
        # second convolution layer
        x = (x + 2 * self.padding - self.kernel_size) // self.stride + 1
        # second pooling layer
        x = x // 2
        
        return 16 * x * x
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # flatten
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x

def load_fashion_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = torchvision.datasets.FashionMNIST(
        root='../exp1/data',
        train=True,
        transform=transform,
        download=False
    )
    
    test_dataset = torchvision.datasets.FashionMNIST(
        root='../exp1/data',
        train=False,
        transform=transform,
        download=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        test_acc = test_model(model, test_loader)
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}, '
              f'Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
    
    return train_losses, train_accuracies, test_accuracies

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy

def experiment_with_different_configs(configs):
    train_loader, test_loader = load_fashion_mnist()
    results = {}
    for config in configs:
        print(f"\n{'='*50}")
        print(f"test config: {config['name']}")
        print(f"kernel size: {config['kernel_size']}, padding: {config['padding']}, stride: {config['stride']}")
        print(f"{'='*50}")
        
        try:
            model = LeNet(
                kernel_size=config['kernel_size'],
                padding=config['padding'],
                stride=config['stride']
            ).to(device)
            print(f"fc input size: {model.fc_input_size}")
            start_time = time.time()
            train_losses, train_accs, test_accs = train_model(model, train_loader, test_loader, epochs=5)
            training_time = time.time() - start_time

            results[config['name']] = {
                'config': config,
                'train_losses': train_losses,
                'train_accuracies': train_accs,
                'test_accuracies': test_accs,
                'final_test_accuracy': test_accs[-1],
                'training_time': training_time,
                'fc_input_size': model.fc_input_size
            }
            
            print(f"training time: {training_time:.2f}s")
            print(f"final test accuracy: {test_accs[-1]:.2f}%")
            
        except Exception as e:
            print(f"config {config['name']} failed: {str(e)}")
            results[config['name']] = {
                'config': config,
                'error': str(e)
            }
    
    return results

def plot_results(results):
    os.makedirs('results/exp2_4', exist_ok=True)
    successful_results = {k: v for k, v in results.items() if 'error' not in v}
    if not successful_results:
        print("no successful results can be visualized")
        return
    
    plt.figure(figsize=(15, 10))
    
    # subplot 1: test accuracy during training
    plt.subplot(2, 2, 1)
    for name, result in successful_results.items():
        epochs = range(1, len(result['test_accuracies']) + 1)
        plt.plot(epochs, result['test_accuracies'], marker='o', label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy Change with Different Configurations')
    plt.legend()
    plt.grid(True)
    
    # subplot 2: final test accuracy comparison
    plt.subplot(2, 2, 2)
    names = list(successful_results.keys())
    final_accs = [result['final_test_accuracy'] for result in successful_results.values()]
    bars = plt.bar(range(len(names)), final_accs)
    plt.xlabel('Configuration')
    plt.ylabel('Final Test Accuracy (%)')
    plt.title('Final Test Accuracy Comparison with Different Configurations')
    plt.xticks(range(len(names)), [name.replace('_', '\n') for name in names], rotation=0)
    
    for bar, acc in zip(bars, final_accs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{acc:.1f}%', ha='center', va='bottom')
    
    # subplot 3: training time comparison
    plt.subplot(2, 2, 3)
    training_times = [result['training_time'] for result in successful_results.values()]
    bars = plt.bar(range(len(names)), training_times)
    plt.xlabel('Configuration')
    plt.ylabel('Training Time (s)')
    plt.title('Training Time Comparison with Different Configurations')
    plt.xticks(range(len(names)), [name.replace('_', '\n') for name in names], rotation=0)
    
    for bar, time_val in zip(bars, training_times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{time_val:.1f}s', ha='center', va='bottom')
    
    # subplot 4: fc input size comparison
    plt.subplot(2, 2, 4)
    fc_sizes = [result['fc_input_size'] for result in successful_results.values()]
    bars = plt.bar(range(len(names)), fc_sizes)
    plt.xlabel('Configuration')
    plt.ylabel('FC Input Size')
    plt.title('FC Input Size Comparison with Different Configurations')
    plt.xticks(range(len(names)), [name.replace('_', '\n') for name in names], rotation=0)
    
    for bar, size in zip(bars, fc_sizes):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{size}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/exp2_4/lenet_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    generate_report(results)



def main():
    configs = [
        {'kernel_size': 3, 'padding': 0, 'stride': 1, 'name': 'kernel_3x3_pad_0_stride_1'},
        {'kernel_size': 5, 'padding': 0, 'stride': 1, 'name': 'kernel_5x5_pad_0_stride_1'},
        {'kernel_size': 7, 'padding': 0, 'stride': 1, 'name': 'kernel_7x7_pad_0_stride_1'},
        {'kernel_size': 5, 'padding': 2, 'stride': 1, 'name': 'kernel_5x5_pad_2_stride_1'},
        # {'kernel_size': 5, 'padding': 0, 'stride': 2, 'name': 'kernel_5x5_pad_0_stride_2'},
        {'kernel_size': 3, 'padding': 1, 'stride': 1, 'name': 'kernel_3x3_pad_1_stride_1'},
    ]
    results = experiment_with_different_configs(configs=configs)
    plot_results(results)
    print("\nexp2_4 completed!")
    print("results saved to results/exp2_4/")

if __name__ == "__main__":
    main() 