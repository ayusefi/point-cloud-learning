import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from data_augmentation import jitter, random_dropout

class ModelNet10Dataset(Dataset):
    def __init__(self, root_dir, split='train', num_points=1024, classes=None, augment=False):
        """
        Args:
            root_dir: Path to ModelNet10 directory
            split: 'train' or 'test'
            num_points: Number of points to sample from mesh
            classes: Optional list of class names (if you want a subset)
            augment: Boolean flag to apply data augmentation
        """
        self.root_dir = root_dir
        self.split = split
        self.num_points = num_points
        self.augment = augment
        self.files = []
        self.class_map = {}

        # Determine which classes to use
        all_classes = sorted(os.listdir(root_dir))
        if classes is None:
            classes = all_classes

        # Build list of (file_path, label_index)
        for idx, cls in enumerate(classes):
            self.class_map[cls] = idx
            pattern = os.path.join(root_dir, cls, split, '*.off')
            for fpath in glob.glob(pattern):
                self.files.append((fpath, idx))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load and sample points
        file_path, label = self.files[idx]
        verts = self.load_off(file_path)            # (V, 3)
        verts = self.sample_points(verts)           # (num_points, 3)

        if self.augment:
            # 1) Jitter: add Gaussian noise
            verts = jitter(verts, sigma=0.01)

            # 2) Dropout: randomly remove points
            verts = random_dropout(verts, p=0.2)

            # 3) Pad or trim to ensure fixed size
            N = verts.shape[0]
            if N < self.num_points:
                # Sample with replacement to pad up
                pad_idx = np.random.choice(N, self.num_points - N, replace=True)
                pad_pts = verts[pad_idx, :]
                verts = np.vstack([verts, pad_pts])
            elif N > self.num_points:
                # Trim excess points
                verts = verts[:self.num_points, :]

        # Return tensor of shape (num_points, 3) and label
        return torch.tensor(verts, dtype=torch.float32), label

    def load_off(self, file_path):
        """Load vertices from an OFF file."""
        with open(file_path, 'r') as f:
            header = f.readline().strip()
            if header != 'OFF':
                raise ValueError('Not a valid OFF file')
            n_verts, n_faces, _ = map(int, f.readline().strip().split())
            verts = []
            for _ in range(n_verts):
                verts.append(list(map(float, f.readline().strip().split())))
        return np.array(verts, dtype=np.float32)

    def sample_points(self, verts):
        """Uniformly sample or duplicate points to get exactly num_points."""
        V = verts.shape[0]
        if V >= self.num_points:
            indices = np.random.choice(V, self.num_points, replace=False)
        else:
            indices = np.random.choice(V, self.num_points, replace=True)
        return verts[indices]

if __name__ == "__main__":
    # Quick sanity check
    dataset = ModelNet10Dataset(root_dir='../data/ModelNet10',
                                split='train',
                                num_points=1024,
                                augment=True)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    for pts, labels in loader:
        print("Points batch shape:", pts.shape)   # Expected: (8, 1024, 3)
        print("Labels batch shape:", labels.shape) # Expected: (8,)
        print("labels:", labels)
        break
