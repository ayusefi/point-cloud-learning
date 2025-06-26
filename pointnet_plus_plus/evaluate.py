import argparse
import torch
from torch.utils.data import DataLoader
from model_pp import PointNetPlusPlus
from modelnet10_dataset import ModelNet10Dataset
import numpy as np
from data_augmentation import jitter, random_dropout
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

def evaluate(model_path, data_dir, batch_size=16, num_points=1024,
             jitter_sigma=0.01, dropout_p=0.2, device=None, n_vis=3):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")

    # Load model
    model = PointNetPlusPlus(num_classes=10).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    # Dataset + loader
    test_ds = ModelNet10Dataset(root_dir=data_dir, split='test',
                                num_points=num_points, augment=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, drop_last=False)

    correct = 0
    total = 0

    with torch.no_grad():
        for points, labels in test_loader:
            # Apply jitter and dropout in numpy
            points_np = points.cpu().numpy()
            for i in range(points_np.shape[0]):
                pts = jitter(points_np[i], sigma=jitter_sigma)
                pts = random_dropout(pts, p=dropout_p)
                # Pad back to num_points if needed
                if pts.shape[0] < num_points:
                    pad_size = num_points - pts.shape[0]
                    pad_pts = np.zeros((pad_size, 3), dtype=np.float32)
                    pts = np.vstack((pts, pad_pts))
                points_np[i] = pts

            points_tensor = torch.tensor(points_np, dtype=torch.float32).to(device)
            labels = labels.to(device)

            outputs = model(points_tensor)
            preds = outputs.argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    acc = correct / total * 100
    print(f"Corrupted Test Accuracy: {acc:.2f}% ({correct}/{total})")

    # --- Visualization of random samples ---
    vis_indices = random.sample(range(len(test_ds)), k=min(n_vis, len(test_ds)))
    fig = plt.figure(figsize=(5 * len(vis_indices), 5))
    for i, idx in enumerate(vis_indices):
        # load and corrupt
        pts, true_label = test_ds[idx]
        pts = pts.numpy()
        pts = jitter(pts, sigma=jitter_sigma)
        pts = random_dropout(pts, p=dropout_p)
        if pts.shape[0] < num_points:
            pad_size = num_points - pts.shape[0]
            pad_pts = np.zeros((pad_size, 3), dtype=np.float32)
            pts = np.vstack((pts, pad_pts))
        pts_tensor = torch.tensor(pts, dtype=torch.float32).unsqueeze(0).to(device)

        # predict
        with torch.no_grad():
            out = model(pts_tensor)
            pred_label = out.argmax(dim=1).item()

        # plot
        ax = fig.add_subplot(1, len(vis_indices), i+1, projection='3d')
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=5)
        ax.set_title(f"True: {true_label}, Pred: {pred_label}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("evaluated_samples.png")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate a PointNet++ model on a corrupted ModelNet10 test set')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to ModelNet10 data directory')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_points', type=int, default=1024)
    parser.add_argument('--jitter_sigma', type=float, default=0.01,
                        help='Standard deviation for jitter')
    parser.add_argument('--dropout_p', type=float, default=0.2,
                        help='Dropout probability')
    parser.add_argument('--n_vis', type=int, default=3,
                        help='Number of random samples to visualize')
    args = parser.parse_args()

    evaluate(args.model, args.data,
             batch_size=args.batch_size,
             num_points=args.num_points,
             jitter_sigma=args.jitter_sigma,
             dropout_p=args.dropout_p)
