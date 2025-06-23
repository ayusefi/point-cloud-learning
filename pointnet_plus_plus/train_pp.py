
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model_pp import PointNetPlusPlus
from modelnet10_dataset import ModelNet10Dataset


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size, num_points = 16, 1024
    lr, epochs = 0.001, 100

    # data
    train_ds = ModelNet10Dataset('../data/ModelNet10', 'train', num_points)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # model
    model = PointNetPlusPlus(num_classes=10, input_channels=0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for xyz, labels in train_loader:
            xyz, labels = xyz.to(device), labels.to(device)
            logits = model(xyz)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

if __name__ == '__main__':
    train()
