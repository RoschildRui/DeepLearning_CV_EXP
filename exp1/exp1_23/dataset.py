from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_fashion_mnist(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.FashionMNIST(
        root='code\exp1\data', 
        train=True, 
        download=False, 
        transform=transform
    )
    
    test_dataset = datasets.FashionMNIST(
        root='code\exp1\data', 
        train=False, 
        download=False, 
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, test_loader