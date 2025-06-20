import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import torch

RANDOME_SEED = 20250520

def seed_everything(seed=RANDOME_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

class SimpleLinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for i in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            # l2 loss
            loss = np.mean((y_pred - y) ** 2)
            self.loss_history.append(loss)
            dw = (2/n_samples) * np.dot(X.T, (y_pred - y))
            db = (2/n_samples) * np.sum(y_pred - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            if (i+1) % 100 == 0:
                print(f'iteration {i+1}/{self.n_iterations}, loss: {loss:.4f}')
                
        return self
        
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def plot_loss(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history)
        plt.title('linear regression loss curve')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.grid(True)
        plt.show()

def generate_data():
    # ref: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html
    X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=RANDOME_SEED)
    return X, y

def main():
    seed_everything(seed=RANDOME_SEED)
    X, y = generate_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOME_SEED)
    
    model = SimpleLinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = np.mean((y_pred - y_test) ** 2)
    print(f"mse on test set: {mse:.4f}")
    model.plot_loss()

    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', label='actual data')
    plt.plot(X_test, y_pred, color='red', label='predicted data')
    plt.title('linear regression prediction result')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main() 