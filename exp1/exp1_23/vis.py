import matplotlib.pyplot as plt
import torch
import warnings
warnings.filterwarnings("ignore")

def plot_results(train_losses, test_accuracies,title=''):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(train_losses)
    ax1.set_title(f'{title} training loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('loss')
    ax1.grid(True)
    
    ax2.plot(test_accuracies)
    ax2.set_title(f'{title} test accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('accuracy (%)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def show_predictions(model, test_loader, num_images=5, device="cpu"):
    model.eval()
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    with torch.no_grad():
        images_device = images[:num_images].to(device)
        outputs = model(images_device)
        _, predicted = torch.max(outputs, 1)
    
    fig = plt.figure(figsize=(12, 4))
    for idx in range(num_images):
        ax = fig.add_subplot(1, num_images, idx+1, xticks=[], yticks=[])
        img = images[idx].numpy().squeeze()
        ax.imshow(img, cmap='gray')
        
        # green for correct, blue for incorrect
        pred_class = class_names[predicted[idx]]
        true_class = class_names[labels[idx]]
        color = 'green' if predicted[idx] == labels[idx] else 'blue'
        
        ax.set_title(f'Predict: {pred_class}\nTrue: {true_class}', color=color)
    
    plt.tight_layout()
    plt.show()

