#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Optional
from house_price_dataset import HousePriceDataset

class HousePriceTrainer:
    def __init__(self, model, output_dir='checkpoints'):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)
        print(f"use device: {self.device}")
        
    def train(self, X_train, y_train, X_val, y_val, 
              batch_size=32, epochs=100, learning_rate=0.001,
              early_stopping_patience=10):
        train_dataset = HousePriceDataset(X_train, y_train)
        val_dataset = HousePriceDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # criterion = nn.MSELoss()
        criterion = nn.SmoothL1Loss()
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        train_losses = []
        val_losses = []
        val_r2_scores = []
        best_val_loss = float('inf')
        best_model_path = None
        early_stopping_counter = 0
        
        for epoch in range(epochs):
            start_time = time.time()
            self.model.train()
            epoch_loss = 0
            
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
                for X_batch, y_batch in pbar:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    
                    optimizer.zero_grad()
                    y_pred = self.model(X_batch)
                    loss = criterion(y_pred, y_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

                    epoch_loss += loss.item() * X_batch.size(0)
                    pbar.set_postfix({"loss": loss.item()})
            
            epoch_loss /= len(train_loader.dataset)
            train_losses.append(epoch_loss)
            
            val_loss, val_r2 = self.evaluate(val_loader, criterion)
            val_losses.append(val_loss)
            val_r2_scores.append(val_r2)

            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(self.output_dir, f'best_model_epoch_{epoch}.pt')
                torch.save(self.model.state_dict(), best_model_path)
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= early_stopping_patience:
                print(f"early stopping: {early_stopping_patience} epochs without improvement")
                break

            epoch_time = time.time() - start_time
            print(f"epoch {epoch+1}/{epochs} - "
                  f"train loss: {epoch_loss:.4f}, validation loss: {val_loss:.4f}, "
                  f"validation R²: {val_r2:.4f}, time: {epoch_time:.2f}s")
                
        if best_model_path:
            self.model.load_state_dict(torch.load(best_model_path))
        
        self.plot_training_curves(train_losses, val_losses, val_r2_scores)
        
        return train_losses, val_losses, val_r2_scores
    
    def evaluate(self, data_loader, criterion=None):
        if criterion is None:
            # criterion = nn.MSELoss()
            criterion = nn.SmoothL1Loss()
            
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch)
                total_loss += loss.item()
                
                all_preds.extend(y_pred.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
                
        avg_loss = total_loss / len(data_loader)
        all_preds = np.array(all_preds).flatten()
        all_targets = np.array(all_targets).flatten()
        r2 = r2_score(all_targets, all_preds)
        
        return avg_loss, r2
    
    def predict(self, X_test):
        test_dataset = HousePriceDataset(X_test)
        test_loader = DataLoader(test_dataset, batch_size=64)
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for X_batch in test_loader:
                X_batch = X_batch.to(self.device)
                y_pred = self.model(X_batch)
                predictions.extend(y_pred.cpu().numpy())
                
        return np.array(predictions)
    
    def plot_training_curves(self, train_losses, val_losses, val_r2_scores):
        plt.figure(figsize=(16, 8))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='train loss')
        plt.plot(val_losses, label='validation loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('train and validation loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(val_r2_scores, label='validation R²')
        plt.xlabel('epoch')
        plt.ylabel('R²')
        plt.title('validation R²')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_curves.png'))
        plt.close()

# --------------------------- Baseline MLP  --------------------------- #
class BaselineHousePriceModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x).squeeze(-1)


###############################################################################
# Enhanced model components                                                   #
###############################################################################


class ResidualMLPBlock(nn.Module):
    """
    a two-layer MLP block with residual connection.
    description: FC -> BN -> GELU -> Dropout -> FC -> BN -> Residual Add -> GELU -> Dropout
    if input and output dimension are not the same, use linear layer to project residual.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)

        self.proj = None
        if in_dim != out_dim:
            self.proj = nn.Linear(in_dim, out_dim)

        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        if self.proj is not None:
            nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x):
        identity = x if self.proj is None else self.proj(x)

        out = self.dropout(self.act(self.bn1(self.fc1(x))))
        out = self.bn2(self.fc2(out))

        out = out + identity
        out = self.act(out)
        out = self.dropout(out)
        return out 

class HousePriceModel(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dims: Optional[List[int]] = None,
                 dropout: float = 0.3):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 512, 256, 256, 128]

        blocks = []
        prev_dim = input_dim
        for dim in hidden_dims:
            blocks.append(ResidualMLPBlock(prev_dim, dim, dropout=dropout))
            prev_dim = dim

        self.feature_extractor = nn.Sequential(*blocks)
        self.regressor = nn.Linear(prev_dim, 1)
        nn.init.xavier_uniform_(self.regressor.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        return self.regressor(x).squeeze(-1) 