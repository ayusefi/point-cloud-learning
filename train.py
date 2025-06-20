import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from modelnet10_dataset import ModelNet10Dataset
from model import PointNetClassifier

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    num_points = 1024
    num_classes = 10
    batch_size = 16
    epochs = 10
    learning_rate = 0.001

    # Dataset
    train_dataset = ModelNet10Dataset(root_dir='data/ModelNet10', split='train', num_points=num_points)
    test_dataset = ModelNet10Dataset(root_dir='data/ModelNet10', split='test', num_points=num_points)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = PointNetClassifier(num_classes=num_classes).to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for points, labels in train_loader:
            points = points.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(points)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.max(1)[1]
            total_correct += preds.eq(labels).sum().item()
            total_samples += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        train_acc = total_correct / total_samples * 100
        train_losses.append(avg_loss)
        train_accuracies.append(train_acc)

        # Validation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for points, labels in test_loader:
                points = points.to(device)
                labels = labels.to(device)

                outputs = model(points)
                preds = outputs.max(1)[1]
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)

        test_acc = correct / total * 100
        test_accuracies.append(test_acc)

        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {avg_loss:.4f}, "
              f"Train Acc: {train_acc:.2f}%, "
              f"Test Acc: {test_acc:.2f}%")

        # Save checkpoint
        torch.save(model.state_dict(), f"pointnet_epoch{epoch+1}.pth")

    # Plot loss/accuracy curves
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(train_accuracies, label='Train Acc')
    plt.plot(test_accuracies, label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

if __name__ == "__main__":
    train()
