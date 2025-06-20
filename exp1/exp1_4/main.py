import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from house_price_dataset import HousePriceDataset
from house_price_model import HousePriceModel, HousePriceTrainer

def main():
    os.makedirs('output', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    data_processor = HousePriceDataset(train_file='../data/house_price/kaggle_house_pred_train.csv',
                                       test_file='../data/house_price/kaggle_house_pred_test.csv')
    X_train, X_val, y_train, y_val = data_processor.preprocess_data()
    X_test, test_ids = data_processor.prepare_test_data()
    
    print(f"train set size: {X_train.shape}")
    print(f"validation set size: {X_val.shape}")
    print(f"test set size: {X_test.shape}")
    
    input_dim = X_train.shape[1]
    model = HousePriceModel(
        input_dim=input_dim,
        hidden_dims=[256, 64],
        dropout=0.5
    )
    trainer = HousePriceTrainer(model, output_dir='checkpoints')
    train_losses, val_losses, val_r2_scores = trainer.train(
        X_train, y_train, X_val, y_val,
        batch_size=64,
        epochs=300,
        learning_rate=1e-3,
        early_stopping_patience=50
    )
    
    test_predictions = np.expm1(trainer.predict(X_test))
    
    predictions_df = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': test_predictions
    })
    predictions_df.to_csv('output/predictions.csv', index=False)
    
    print("prediction completed, results saved to output/predictions.csv")

    plt.figure(figsize=(12, 4))
    # subplot 1: train loss and validation loss
    plt.subplot(1, 2, 1)
    epochs_range = range(1, len(train_losses) + 1)
    # plt.plot(epochs_range, train_losses, 'b-', label='train loss', linewidth=2)
    # plt.plot(epochs_range, val_losses, 'r-', label='validation loss', linewidth=2)
    # plt.title('loss change during training', fontsize=14, fontweight='bold')
    # plt.xlabel('epochs', fontsize=12)
    # plt.ylabel('loss', fontsize=12)
    # plt.legend(fontsize=11)
    # plt.grid(True, alpha=0.3)

    plt.plot(epochs_range, train_losses, 'b-', label='train loss', linewidth=2, alpha=0.8)
    plt.plot(epochs_range, val_losses, 'r-', label='validation loss', linewidth=2, alpha=0.8)
    plt.fill_between(epochs_range, train_losses, alpha=0.3, color='blue')
    plt.fill_between(epochs_range, val_losses, alpha=0.3, color='red')
    plt.title('loss change curve', fontsize=13, fontweight='bold')
    plt.xlabel('epochs', fontsize=11)
    plt.ylabel('loss', fontsize=11)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # subplot 2: validation R² score
    plt.subplot(1, 2, 2)
    filtered_indices = [i for i, r2 in enumerate(val_r2_scores) if -1 <= r2 <= 1]
    filtered_epochs = [i + 1 for i in filtered_indices]
    filtered_r2 = [val_r2_scores[i] for i in filtered_indices]
    plt.plot(filtered_epochs, filtered_r2, 'g-', label='filtered validation R² score', linewidth=2)
    plt.fill_between(filtered_epochs, filtered_r2, alpha=0.3, color='green')
    # plt.plot(epochs_range, val_r2_scores, 'g-', label='validation R² score', linewidth=2)
    plt.ylim(-1, 1)
    plt.title('validation R² score change (focus on [-1, 1])', fontsize=14, fontweight='bold')
    plt.xlabel('epochs', fontsize=12)
    plt.ylabel('R² score', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # # subplot 3: train loss and validation loss (log scale)
    # plt.subplot(1, 3, 3)
    # plt.semilogy(epochs_range, train_losses, 'b-', label='train loss', linewidth=2)
    # plt.semilogy(epochs_range, val_losses, 'r-', label='validation loss', linewidth=2)
    # plt.title('loss change (log scale)', fontsize=14, fontweight='bold')
    # plt.xlabel('epochs', fontsize=12)
    # plt.ylabel('loss (log scale)', fontsize=12)
    # plt.legend(fontsize=11)
    # plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/training_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize=(12, 4))
    
    # # subplot 1: train loss and validation loss (linear scale)
    # plt.subplot(2, 3, 1)
    # plt.plot(epochs_range, train_losses, 'b-', label='train loss', linewidth=2, alpha=0.8)
    # plt.plot(epochs_range, val_losses, 'r-', label='validation loss', linewidth=2, alpha=0.8)
    # plt.fill_between(epochs_range, train_losses, alpha=0.3, color='blue')
    # plt.fill_between(epochs_range, val_losses, alpha=0.3, color='red')
    # plt.title('loss change curve', fontsize=13, fontweight='bold')
    # plt.xlabel('epochs', fontsize=11)
    # plt.ylabel('loss', fontsize=11)
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    
    # subplot 2: validation R² score change
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, val_r2_scores, 'g-', linewidth=2, marker='o', markersize=4, alpha=0.8)
    plt.fill_between(epochs_range, val_r2_scores, alpha=0.3, color='green')
    plt.title('validation R² score change (full training)', fontsize=13, fontweight='bold')
    plt.xlabel('epochs', fontsize=11)
    plt.ylabel('R² score', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # subplot 3: loss difference (overfitting detection)
    plt.subplot(1, 2, 1)
    loss_diff = np.array(val_losses) - np.array(train_losses)
    plt.plot(epochs_range, loss_diff, 'purple', linewidth=2, alpha=0.8)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.fill_between(epochs_range, loss_diff, alpha=0.3, color='purple')
    plt.title('validation loss - train loss', fontsize=13, fontweight='bold')
    plt.xlabel('epochs', fontsize=11)
    plt.ylabel('loss difference', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    best_epoch = np.argmax(val_r2_scores) + 1
    best_r2 = max(val_r2_scores)
    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]
    final_r2 = val_r2_scores[-1]

    plt.tight_layout()
    plt.savefig('output/detailed_training_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    stats_text = (
        f"\ntraining statistics summary:\n"
        f"best R² score: {best_r2:.4f}\n"
        f"best epochs: {best_epoch}\n"
        f"final train loss: {final_train_loss:.4f}\n"
        f"final validation loss: {final_val_loss:.4f}\n"
        f"final R² score: {final_r2:.4f}\n"
        f"total epochs: {len(train_losses)}\n"
    )
    print(stats_text)

    plt.figure(figsize=(12, 6))
    plt.plot(test_predictions[:100])
    plt.title('predictions of the first 100 test samples')
    plt.xlabel('sample ID')
    plt.ylabel('predicted price')
    plt.savefig('output/test_predictions.png')
    plt.close()
    
    print("所有图表已保存到 output/ 目录:")
    print("- training_metrics.png: 训练指标变化概览")
    print("- detailed_training_analysis.png: 详细训练分析")
    print("- test_predictions.png: 测试预测结果")
    
if __name__ == "__main__":
    main() 