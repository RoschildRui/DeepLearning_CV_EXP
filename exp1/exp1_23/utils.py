import torch
import random
import numpy as np
from tqdm import tqdm

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def seed_everything(seed=20250520):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def train(model, train_loader, test_loader, criterion, optimizer, epochs=10, device="cpu"):
    model.to(device)
    train_losses = []
    test_accuracies = []
    
    for epoch in tqdm(range(epochs), desc="training progress"):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_accuracy = 100 * correct / total
        test_accuracies.append(test_accuracy)
        
        print(f'epoch [{epoch+1}/{epochs}], loss: {epoch_loss:.4f}, test accuracy: {test_accuracy:.2f}%')
    
    return train_losses, test_accuracies

