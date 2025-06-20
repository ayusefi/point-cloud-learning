import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from modelnet10_dataset import ModelNet10Dataset  # Adjust the name if different
from model import PointNetClassifier

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Parameters
    num_points = 1024
    num_classes = 10
    batch_size = 16
    epochs = 10
    learning_rate = 0.001

    # Dataset and DataLoader
    train_dataset = ModelNet10Dataset(root_dir='../data/archive/ModelNet10', split='train', num_points=num_points)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = PointNetClassifier(num_classes=num_classes).to(device)
    criterion = nn.NLLLoss()  # Because model uses log_softmax
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for points, labels in train_loader:
            points = points.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(points)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    train()
