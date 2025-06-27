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
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json

def evaluate(model_path, data_dir, batch_size=16, num_points=1024,
             jitter_sigma=0.01, dropout_p=0.2, device=None, n_vis=3,
             apply_corruption=True, vis_indices=None):
    """
    Evaluate a PointNet++ model on the ModelNet10 test set.

    Args:
        model_path (str): Path to the model checkpoint file.
        data_dir (str): Directory containing ModelNet10 data.
        batch_size (int): Number of samples per batch.
        num_points (int): Number of points per point cloud.
        jitter_sigma (float): Standard deviation for jitter corruption.
        dropout_p (float): Probability for random dropout corruption.
        device (torch.device): Device to run the evaluation on (CPU/GPU).
        n_vis (int): Number of samples to visualize if vis_indices is not provided.
        apply_corruption (bool): Whether to apply data corruption during evaluation.
        vis_indices (list): List of specific indices to visualize (overrides n_vis).
    """
    # Device setup
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")
    
    # Corruption status
    corruption_str = "with corruption" if apply_corruption else "without corruption"
    print(f"Evaluation mode: {corruption_str}")
    if apply_corruption:
        print(f"Corruption parameters: jitter_sigma={jitter_sigma}, dropout_p={dropout_p}")

    # Load model
    model = PointNetPlusPlus(num_classes=10).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    # Load dataset
    test_ds = ModelNet10Dataset(root_dir=data_dir, split='test',
                                num_points=num_points, augment=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, drop_last=False)

    # Class names
    class_names = sorted(list(test_ds.class_map.keys()))
    print(f"Classes: {class_names}")

    # Evaluation loop
    all_predictions = []
    all_labels = []
    correct = 0
    total = 0

    print("Evaluating model...")
    with torch.no_grad():
        for batch_idx, (points, labels) in enumerate(test_loader):
            if apply_corruption:
                points_np = points.cpu().numpy()
                for i in range(points_np.shape[0]):
                    pts = jitter(points_np[i], sigma=jitter_sigma)
                    pts = random_dropout(pts, p=dropout_p)
                    if pts.shape[0] < num_points:
                        pad_size = num_points - pts.shape[0]
                        pad_pts = np.zeros((pad_size, 3), dtype=np.float32)
                        pts = np.vstack((pts, pad_pts))
                    elif pts.shape[0] > num_points:
                        pts = pts[:num_points]
                    points_np[i] = pts
                points_tensor = torch.tensor(points_np, dtype=torch.float32).to(device)
            else:
                points_tensor = points.to(device)
            
            labels = labels.to(device)
            outputs = model(points_tensor)
            preds = outputs.argmax(dim=1)
            
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(test_loader)} batches")

    # Calculate accuracy
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    acc = correct / total * 100
    corruption_prefix = "Corrupted " if apply_corruption else "Clean "
    print(f"\n{corruption_prefix}Test Accuracy: {acc:.2f}% ({correct}/{total})")

    # Classification report
    report = classification_report(all_labels, all_predictions, 
                                   target_names=class_names, digits=4, zero_division=0)
    print("\n" + "="*80)
    print("DETAILED CLASSIFICATION METRICS")
    print("="*80)
    print(report)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title(f'{corruption_prefix}Test Set Confusion Matrix\nAccuracy: {acc:.2f}%')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    corruption_suffix = "_corrupted" if apply_corruption else "_clean"
    plt.savefig(f"confusion_matrix{corruption_suffix}.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Visualization
    if vis_indices is not None:
        print(f"\nGenerating visualization of specified samples (indices: {vis_indices})...")
        if any(idx < 0 or idx >= len(test_ds) for idx in vis_indices):
            raise ValueError("Some indices are out of range")
    else:
        vis_indices = random.sample(range(len(test_ds)), k=min(n_vis, len(test_ds)))
        print(f"\nGenerating visualization of {len(vis_indices)} random samples...")

    fig = plt.figure(figsize=(5 * len(vis_indices), 6))
    for i, idx in enumerate(vis_indices):
        pts, true_label = test_ds[idx]
        pts = pts.numpy()
        
        if apply_corruption:
            pts = jitter(pts, sigma=jitter_sigma)
            pts = random_dropout(pts, p=dropout_p)
            if pts.shape[0] < num_points:
                pad_size = num_points - pts.shape[0]
                pad_pts = np.zeros((pad_size, 3), dtype=np.float32)
                pts = np.vstack((pts, pad_pts))
            elif pts.shape[0] > num_points:
                pts = pts[:num_points]
        
        pts_tensor = torch.tensor(pts, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(pts_tensor)
            pred_label = out.argmax(dim=1).item()

        ax = fig.add_subplot(1, len(vis_indices), i+1, projection='3d')
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=5, alpha=0.6)
        
        true_class = class_names[true_label]
        pred_class = class_names[pred_label]
        color = 'green' if true_label == pred_label else 'red'
        
        ax.set_title(f"Index: {idx}\nTrue: {true_class}\nPred: {pred_class}", 
                     color=color, fontsize=10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        max_range = np.array([pts[:,0].max()-pts[:,0].min(),
                              pts[:,1].max()-pts[:,1].min(),
                              pts[:,2].max()-pts[:,2].min()]).max() / 2.0
        mid_x = (pts[:,0].max()+pts[:,0].min()) * 0.5
        mid_y = (pts[:,1].max()+pts[:,1].min()) * 0.5
        mid_z = (pts[:,2].max()+pts[:,2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.savefig(f"evaluated_samples{corruption_suffix}.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Save results
    results = {
        'overall_accuracy': acc,
        'total_samples': total,
        'correct_predictions': correct,
    }
    results_filename = f"evaluation_results{corruption_suffix}.json"
    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {results_filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a PointNet++ model on ModelNet10 test set')
    parser.add_argument('--model', type=str, required=True, 
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--data', type=str, required=True, 
                        help='Path to ModelNet10 data directory')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Batch size for evaluation')
    parser.add_argument('--num_points', type=int, default=1024, 
                        help='Number of points per point cloud')
    parser.add_argument('--jitter_sigma', type=float, default=0.01, 
                        help='Standard deviation for jitter')
    parser.add_argument('--dropout_p', type=float, default=0.2, 
                        help='Dropout probability')
    parser.add_argument('--n_vis', type=int, default=3,
                        help='Number of random samples to visualize if --vis_indices is not provided')
    parser.add_argument('--vis_indices', type=str, default=None,
                        help='Space-separated list of indices to visualize (e.g., "0 1 2"). '
                             'If provided, overrides --n_vis')
    parser.add_argument('--no_corruption', action='store_true', 
                        help='Evaluate without corruption')
    args = parser.parse_args()

    # Parse vis_indices
    if args.vis_indices is not None:
        try:
            vis_indices = list(map(int, args.vis_indices.split()))
        except ValueError:
            raise ValueError("Invalid vis_indices: must be space-separated integers")
    else:
        vis_indices = None

    # Run evaluation
    evaluate(args.model, args.data,
             batch_size=args.batch_size,
             num_points=args.num_points,
             jitter_sigma=args.jitter_sigma,
             dropout_p=args.dropout_p,
             n_vis=args.n_vis,
             apply_corruption=not args.no_corruption,
             vis_indices=vis_indices)