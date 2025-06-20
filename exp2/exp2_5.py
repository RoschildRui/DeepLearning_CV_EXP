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
from utils import generate_report4exp2_5 as generate_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"use device: {device}")

class AlexNet(nn.Module):
    def __init__(self, num_classes=10, dropout=0.5):
        super(AlexNet, self).__init__()
        
        self.features = nn.Sequential(
            # first convolution layer: input 1x28x28 -> output 96x6x6
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # second convolution layer: input 96x6x6 -> output 256x6x6
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # third convolution layer: input 256x6x6 -> output 384x6x6
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # fourth convolution layer: input 384x6x6 -> output 384x6x6
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # fifth convolution layer: input 384x6x6 -> output 256x6x6
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # adaptive average pooling, ensure output size is 6x6
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # classifier layer
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class ModifiedAlexNet(nn.Module):
    """optimized AlexNet for 28x28 images"""
    def __init__(self, num_classes=10, dropout=0.5):
        super(ModifiedAlexNet, self).__init__()
        
        self.features = nn.Sequential(
            # first layer: 1x28x28 -> 64x14x14
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # second layer: 64x14x14 -> 128x7x7
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # third layer: 128x7x7 -> 256x7x7
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # fourth layer: 256x7x7 -> 256x7x7
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # fifth layer: 256x7x7 -> 256x3x3
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256 * 3 * 3, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def load_fashion_mnist(batch_size=128, augment=True):
    """load FashionMNIST dataset, support data augmentation"""
    if augment:
        # use data augmentation for training set
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))  # FashionMNIST statistics
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    train_dataset = torchvision.datasets.FashionMNIST(
        root='../exp1/data',
        train=True,
        transform=train_transform,
        download=False
    )
    
    test_dataset = torchvision.datasets.FashionMNIST(
        root='../exp1/data',
        train=False,
        transform=test_transform,
        download=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, epochs=20, lr=0.001, weight_decay=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    learning_rates = []
    
    best_test_acc = 0.0
    best_model_state = None
    
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
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        test_acc = test_model(model, test_loader)
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_state = model.state_dict().copy()
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        learning_rates.append(optimizer.param_groups[0]['lr'])
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}, '
              f'Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        scheduler.step()
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'load best model, test accuracy: {best_test_acc:.2f}%')
    
    return train_losses, train_accuracies, test_accuracies, learning_rates, best_test_acc

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
    results = {}
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"test config: {config['name']}")
        print(f"{'='*60}")
        
        try:
            train_loader, test_loader = load_fashion_mnist(
                batch_size=config['batch_size'], 
                augment=config['augment']
            )
            model = config['model_class'](dropout=config['dropout']).to(device)
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"total params: {total_params:,}")
            print(f"trainable params: {trainable_params:,}")
            
            start_time = time.time()
            train_losses, train_accs, test_accs, lrs, best_acc = train_model(
                model, train_loader, test_loader,
                epochs=config['epochs'],
                lr=config['lr'],
                weight_decay=config['weight_decay']
            )
            training_time = time.time() - start_time
            
            results[config['name']] = {
                'config': config,
                'train_losses': train_losses,
                'train_accuracies': train_accs,
                'test_accuracies': test_accs,
                'learning_rates': lrs,
                'best_test_accuracy': best_acc,
                'final_test_accuracy': test_accs[-1],
                'training_time': training_time,
                'total_params': total_params,
                'trainable_params': trainable_params
            }
            
            print(f"training time: {training_time:.2f}s")
            print(f"best test accuracy: {best_acc:.2f}%")
            print(f"final test accuracy: {test_accs[-1]:.2f}%")
            
        except Exception as e:
            print(f"config {config['name']} failed: {str(e)}")
            results[config['name']] = {
                'config': config,
                'error': str(e)
            }
    
    return results

def plot_results(results):
    os.makedirs('results/exp2_5', exist_ok=True)
    
    successful_results = {k: v for k, v in results.items() if 'error' not in v}
    if not successful_results:
        print("no successful results can be visualized")
        return
    
    plt.figure(figsize=(20, 15))
    
    # subplot 1: test accuracy change
    plt.subplot(2, 3, 1)
    for name, result in successful_results.items():
        epochs = range(1, len(result['test_accuracies']) + 1)
        plt.plot(epochs, result['test_accuracies'], marker='o', label=name, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy Change')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # subplot 2: training loss change
    plt.subplot(2, 3, 2)
    for name, result in successful_results.items():
        epochs = range(1, len(result['train_losses']) + 1)
        plt.plot(epochs, result['train_losses'], marker='s', label=name, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Change')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # subplot 3: best accuracy comparison
    plt.subplot(2, 3, 3)
    names = list(successful_results.keys())
    best_accs = [result['best_test_accuracy'] for result in successful_results.values()]
    bars = plt.bar(range(len(names)), best_accs, color='skyblue', edgecolor='navy', alpha=0.7)
    plt.xlabel('Configuration')
    plt.ylabel('Best Test Accuracy (%)')
    plt.title('Best Test Accuracy Comparison')
    plt.xticks(range(len(names)), [name.replace('_', '\n') for name in names], rotation=0)
    
    for bar, acc in zip(bars, best_accs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # subplot 4: training time comparison
    plt.subplot(2, 3, 4)
    training_times = [result['training_time']/60 for result in successful_results.values()]
    bars = plt.bar(range(len(names)), training_times, color='lightcoral', edgecolor='darkred', alpha=0.7)
    plt.xlabel('Configuration')
    plt.ylabel('Training Time (min)')
    plt.title('Training Time Comparison')
    plt.xticks(range(len(names)), [name.replace('_', '\n') for name in names], rotation=0)
    
    for bar, time_val in zip(bars, training_times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{time_val:.1f}min', ha='center', va='bottom', fontweight='bold')
    
    # subplot 5: parameter number comparison
    plt.subplot(2, 3, 5)
    param_counts = [result['total_params']/1e6 for result in successful_results.values()]
    bars = plt.bar(range(len(names)), param_counts, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
    plt.xlabel('Configuration')
    plt.ylabel('Parameter Number (Million)')
    plt.title('Model Parameter Number Comparison')
    plt.xticks(range(len(names)), [name.replace('_', '\n') for name in names], rotation=0)
    
    for bar, count in zip(bars, param_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{count:.2f}M', ha='center', va='bottom', fontweight='bold')
    
    # subplot 6: learning rate change
    plt.subplot(2, 3, 6)
    best_config = max(successful_results.items(), key=lambda x: x[1]['best_test_accuracy'])
    epochs = range(1, len(best_config[1]['learning_rates']) + 1)
    plt.plot(epochs, best_config[1]['learning_rates'], marker='d', color='purple', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title(f'Learning Rate Change - {best_config[0]}')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/exp2_5/alexnet_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    generate_report(results)

def main():
    print(f"use device: {device}")
    
    configs = [
        {
            'name': 'AlexNet_Original',
            'model_class': AlexNet,
            'batch_size': 128,
            'lr': 0.001,
            'epochs': 30,
            'dropout': 0.5,
            'augment': True,
            'weight_decay': 1e-4
        },
        {
            'name': 'ModifiedAlexNet_v1',
            'model_class': ModifiedAlexNet,
            'batch_size': 128,
            'lr': 0.001,
            'epochs': 30,
            'dropout': 0.5,
            'augment': True,
            'weight_decay': 1e-4
        },
        {
            'name': 'ModifiedAlexNet_v2',
            'model_class': ModifiedAlexNet,
            'batch_size': 64,
            'lr': 0.0005,
            'epochs': 30,
            'dropout': 0.3,
            'augment': True,
            'weight_decay': 5e-4
        },
        {
            'name': 'ModifiedAlexNet_v3',
            'model_class': ModifiedAlexNet,
            'batch_size': 128,
            'lr': 0.002,
            'epochs': 30,
            'dropout': 0.4,
            'augment': True,
            'weight_decay': 1e-3
        }
    ]
    results = experiment_with_different_configs(configs=configs)
    
    plot_results(results)
    
    successful_results = {k: v for k, v in results.items() if 'error' not in v}
    if successful_results:
        best_config = max(successful_results.items(), key=lambda x: x[1]['best_test_accuracy'])
        print(f"\nexp2_5 completed!")
        print(f"best config: {best_config[0]}")
        print(f"best accuracy: {best_config[1]['best_test_accuracy']:.2f}%")
        print(f"training time: {best_config[1]['training_time']/60:.1f}min")
    print("\nall results saved to results/exp2_5/")

if __name__ == "__main__":
    main()