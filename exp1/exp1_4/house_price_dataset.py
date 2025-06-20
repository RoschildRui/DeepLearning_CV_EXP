import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split

class HousePriceDataset(Dataset):
    def __init__(self, X=None, y=None,
                 train_file=None,
                 test_file=None):
        self.X = X
        self.y = y
        self.scaler = None
        self.feature_names = None
        self.num_cols = None
        self.train_file = train_file if train_file is not None else '../data/house_price/kaggle_house_pred_train.csv'
        self.test_file = test_file if test_file is not None else '../data/house_price/kaggle_house_pred_test.csv'
        
    def __len__(self):
        if self.X is None:
            return 0
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.X is None:
            raise ValueError("Dataset not initialized with data. Please provide X data.")
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        if self.y is not None:
            y = torch.tensor(self.y[idx], dtype=torch.float32)
            return x, y
        return x
    
    def preprocess_data(self, test_size=0.2, random_state=42):

        df = pd.read_csv(self.train_file)
        y = np.log1p(df['SalePrice'].values)  # log(1+price)
        df = df.drop(['Id', 'SalePrice'], axis=1)
        
        columns_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
        existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        df = df.drop(columns=existing_columns_to_drop)

        df = pd.get_dummies(df, drop_first=True)
        self.num_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
        cat_cols = [col for col in df.columns if df[col].dtype in ['object', 'bool']]
        
        print(f"numerical feature dimension: {len(self.num_cols)}")
        print(f"categorical feature dimension: {len(cat_cols)}")

        imputer = KNNImputer(n_neighbors=10, weights="uniform")
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        if self.num_cols:
            df_imputed[self.num_cols] = np.log1p(df_imputed[self.num_cols])

        self.scaler = StandardScaler()
        df_normalized = df_imputed.copy()
        
        if self.num_cols:
            df_normalized[self.num_cols] = self.scaler.fit_transform(df_imputed[self.num_cols])
        self.feature_names = df_normalized.columns.tolist()

        X_train, X_val, y_train, y_val = train_test_split(
            df_normalized.values, y, test_size=test_size, random_state=random_state
        )

        print(f"preprocessed feature dimension: {X_train.shape[1]}")
        print(f"train set size: {X_train.shape[0]}")
        print(f"validation set size: {X_val.shape[0]}")
        
        return X_train, X_val, y_train, y_val
    
    def prepare_test_data(self):
        if self.scaler is None:
            raise ValueError("Please call preprocess_data to process training data first")
            
        test_df = pd.read_csv(self.test_file)
        test_ids = test_df['Id'].values
        test_df = test_df.drop(['Id'], axis=1)
        columns_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
        existing_columns_to_drop = [col for col in columns_to_drop if col in test_df.columns]
        test_df = test_df.drop(columns=existing_columns_to_drop)
        test_df = pd.get_dummies(test_df, drop_first=True)
        for col in self.feature_names:
            if col not in test_df.columns:
                test_df[col] = 0
        test_df = test_df[self.feature_names]
        imputer = KNNImputer(n_neighbors=10, weights="uniform")
        test_df_imputed = pd.DataFrame(imputer.fit_transform(test_df), columns=test_df.columns)
        if self.num_cols:
            num_cols_in_test = [col for col in self.num_cols if col in test_df_imputed.columns]
            test_df_imputed[num_cols_in_test] = np.log1p(test_df_imputed[num_cols_in_test])
        
        test_df_normalized = test_df_imputed.copy()
        if self.num_cols:
            num_cols_in_test = [col for col in self.num_cols if col in test_df_imputed.columns]
            test_df_normalized[num_cols_in_test] = self.scaler.transform(test_df_imputed[num_cols_in_test])
        
        return test_df_normalized.values, test_ids
    
    @staticmethod
    def create_data_loaders(X_train, y_train, X_val, y_val, batch_size=32):
        train_dataset = HousePriceDataset(X_train, y_train)
        val_dataset = HousePriceDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        return train_loader, val_loader

    def get_feature_names(self):
        if self.scaler is None:
            raise ValueError("Please call preprocess_data to train the scaler first")
        return self.feature_names 