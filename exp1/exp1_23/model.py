import torch.nn as nn

class SoftmaxRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SoftmaxRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
        
    def forward(self, x):
        # x: [batch_size, 1(channel number), 28, 28]
        x = x.view(x.shape[0], -1)
        logits = self.linear(x)
        return logits
    
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.2):
        super(MLP, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))

        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.model(x)