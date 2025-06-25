import argparse
import torch
from torch.utils.data import DataLoader
from model_pp import PointNetPlusPlus
from modelnet10_dataset import ModelNet10Dataset

def evaluate(model_path, data_dir, batch_size=16, num_points=1024, device=None):
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")

    # Load model
    model = PointNetPlusPlus(num_classes=10).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # Prepare test dataset and loader
    test_ds = ModelNet10Dataset(root_dir=data_dir, split='test', num_points=num_points, augment=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Run evaluation
    correct = 0
    total = 0
    with torch.no_grad():
        for points, labels in test_loader:
            points, labels = points.to(device), labels.to(device)
            outputs = model(points)
            preds = outputs.argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    acc = correct / total * 100
    print(f"Test Accuracy: {acc:.2f}% ({correct}/{total})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a PointNet++ model on ModelNet10')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--data', type=str, required=True, help='Path to ModelNet10 data directory')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_points', type=int, default=1024)
    args = parser.parse_args()
    evaluate(args.model, args.data, batch_size=args.batch_size, num_points=args.num_points)
