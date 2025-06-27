import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model_pp import PointNetPlusPlus
from modelnet10_dataset import ModelNet10Dataset
import matplotlib.pyplot as plt
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Train PointNet++ with different augmentation strategies')
    parser.add_argument('--augmentation', type=str, choices=['none', 'jitter', 'dropout', 'both'], 
                        default='none', help='Augmentation strategy to use')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_points', type=int, default=1024, help='Number of points per point cloud')
    parser.add_argument('--data_path', type=str, default='../data/ModelNet10', help='Path to ModelNet10 dataset')
    parser.add_argument('--save_prefix', type=str, default='', help='Prefix for saved model files')
    return parser.parse_args()


def get_augmentation_config(aug_type):
    """
    Returns augmentation configuration based on the specified type.
    Returns tuple: (use_jitter, use_dropout)
    """
    if aug_type == 'none':
        return False, False
    elif aug_type == 'jitter':
        return True, False
    elif aug_type == 'dropout':
        return False, True
    elif aug_type == 'both':
        return True, True
    else:
        raise ValueError(f"Unknown augmentation type: {aug_type}")


def train():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Augmentation strategy: {args.augmentation}")

    # Hyperparameters
    num_points = args.num_points
    num_classes = 10
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr

    # Get augmentation configuration
    use_jitter, use_dropout = get_augmentation_config(args.augmentation)
    
    # Create augmentation-specific parameters dict
    aug_params = {
        'use_jitter': use_jitter,
        'use_dropout': use_dropout
    }

    # Datasets and loaders
    # Note: Assuming your ModelNet10Dataset can accept augmentation parameters
    # If not, you'll need to modify the dataset class accordingly
    train_ds = ModelNet10Dataset(args.data_path, 'train', num_points, augment=args.augmentation != 'none', **aug_params)
    val_ds   = ModelNet10Dataset(args.data_path, 'test',  num_points, augment=False)
    test_ds  = ModelNet10Dataset(args.data_path, 'test',  num_points, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    # Model, optimizer, loss, scheduler
    model = PointNetPlusPlus(num_classes=num_classes).to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Logs
    train_losses, train_accs, val_accs = [], [], []

    # Create experiment name for saving files
    exp_name = f"{args.save_prefix}{args.augmentation}" if args.save_prefix else args.augmentation

    for epoch in range(1, epochs+1):
        # Training
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for points, labels in train_loader:
            points, labels = points.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(points)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * points.size(0)
            preds = logits.argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
        train_loss = running_loss / total
        train_acc = correct / total * 100

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for points, labels in val_loader:
                points, labels = points.to(device), labels.to(device)
                logits = model(points)
                preds = logits.argmax(dim=1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total * 100

        # Scheduler step
        scheduler.step()

        # Record
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Logging
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch}/{epochs} | LR: {current_lr:.4f} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Acc: {val_acc:.2f}%")

        # Checkpoint every 10 epochs
        if epoch % 10 == 0:
            ckpt_name = f"pointnetpp_{exp_name}_epoch{epoch}.pth"
            torch.save(model.state_dict(), ckpt_name)
            print(f"Saved checkpoint: {ckpt_name}")

    # Final test accuracy
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for points, labels in test_loader:
            points, labels = points.to(device), labels.to(device)
            logits = model(points)
            preds = logits.argmax(dim=1)
            test_correct += preds.eq(labels).sum().item()
            test_total += labels.size(0)
    test_acc = test_correct / test_total * 100
    print(f"Test Accuracy: {test_acc:.2f}%")

    # Plot and save training curves
    epochs_range = range(1, epochs+1)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss ({args.augmentation})')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs_range, train_accs, label='Train Acc')
    plt.plot(epochs_range, val_accs,   label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Training & Validation Accuracy ({args.augmentation})')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'training_curves_{exp_name}.png')
    plt.show()

    # Save final model
    final_model_name = f'{exp_name}_model.pth'
    torch.save(model.state_dict(), final_model_name)
    print(f"Saved final model: {final_model_name}")

    # Save training results summary
    results_summary = {
        'augmentation': args.augmentation,
        'final_test_accuracy': test_acc,
        'final_train_accuracy': train_accs[-1],
        'final_val_accuracy': val_accs[-1],
        'hyperparameters': {
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rate': lr,
            'num_points': num_points
        }
    }
    
    import json
    with open(f'results_{exp_name}.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"Saved results summary: results_{exp_name}.json")


if __name__ == '__main__':
    train()