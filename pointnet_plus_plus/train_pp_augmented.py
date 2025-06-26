import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model_pp import PointNetPlusPlus
from modelnet10_dataset import ModelNet10Dataset
import matplotlib.pyplot as plt


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    num_points = 1024
    num_classes = 10
    batch_size = 16
    epochs = 100
    lr = 0.001

    # Datasets and loaders (no augmentation for baseline)
    train_ds = ModelNet10Dataset('../data/ModelNet10', 'train', num_points, augment=True)
    val_ds   = ModelNet10Dataset('../data/ModelNet10', 'test',  num_points, augment=True)
    test_ds  = ModelNet10Dataset('../data/ModelNet10', 'test',  num_points, augment=True)

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
            ckpt_name = f"pointnetpp_epoch{epoch}.pth"
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
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs_range, train_accs, label='Train Acc')
    plt.plot(epochs_range, val_accs,   label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training & Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

    # Save final model
    torch.save(model.state_dict(), 'augmented_model.pth')
    print("Saved final model: augmented_model.pth")

if __name__ == '__main__':
    train()
