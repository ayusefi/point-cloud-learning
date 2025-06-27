import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from data_augmentation import jitter, random_dropout

class ModelNet10Dataset(Dataset):
    def __init__(self, root_dir, split='train', num_points=1024, classes=None, augment=False, 
                 use_jitter=False, use_dropout=False, jitter_sigma=0.01, dropout_p=0.2):
        """
        Args:
            root_dir: Path to ModelNet10 directory
            split: 'train' or 'test'
            num_points: Number of points to sample from mesh
            classes: Optional list of class names (if you want a subset)
            augment: Boolean flag to apply data augmentation (legacy parameter)
            use_jitter: Boolean flag to apply jitter augmentation
            use_dropout: Boolean flag to apply dropout augmentation
            jitter_sigma: Standard deviation for Gaussian noise in jitter
            dropout_p: Probability of dropping each point in dropout
        """
        self.root_dir = root_dir
        self.split = split
        self.num_points = num_points
        self.augment = augment
        self.use_jitter = use_jitter
        self.use_dropout = use_dropout
        self.jitter_sigma = jitter_sigma
        self.dropout_p = dropout_p
        self.files = []
        self.class_map = {}

        # Print augmentation configuration for debugging
        if split == 'train':
            aug_methods = []
            if self.use_jitter:
                aug_methods.append(f"jitter(σ={self.jitter_sigma})")
            if self.use_dropout:
                aug_methods.append(f"dropout(p={self.dropout_p})")
            
            if aug_methods:
                print(f"Dataset ({split}): Using augmentations: {', '.join(aug_methods)}")
            else:
                print(f"Dataset ({split}): No augmentations")

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

        print(f"Loaded {len(self.files)} {split} samples from {len(classes)} classes")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load and sample points
        file_path, label = self.files[idx]
        verts = self.load_off(file_path)            # (V, 3)
        verts = self.sample_points(verts)           # (num_points, 3)

        # Apply augmentations based on configuration
        if self.split == 'train' and (self.augment or self.use_jitter or self.use_dropout):
            
            # Apply jitter if enabled
            if self.use_jitter or (self.augment and not self.use_dropout):
                # If augment=True but specific flags not set, apply jitter for backward compatibility
                verts = jitter(verts, sigma=self.jitter_sigma)

            # Apply dropout if enabled
            if self.use_dropout or (self.augment and not self.use_jitter):
                # If augment=True but specific flags not set, apply dropout for backward compatibility
                verts = random_dropout(verts, p=self.dropout_p)

            # Ensure we still have the correct number of points after dropout
            verts = self._ensure_num_points(verts)

        # Return tensor of shape (num_points, 3) and label
        return torch.tensor(verts, dtype=torch.float32), label

    def _ensure_num_points(self, verts):
        """Ensure the point cloud has exactly num_points after augmentation."""
        N = verts.shape[0]
        if N < self.num_points:
            # Sample with replacement to pad up
            pad_idx = np.random.choice(N, self.num_points - N, replace=True)
            pad_pts = verts[pad_idx, :]
            verts = np.vstack([verts, pad_pts])
        elif N > self.num_points:
            # Trim excess points
            verts = verts[:self.num_points, :]
        return verts

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

    def get_augmentation_info(self):
        """Return information about current augmentation configuration."""
        info = {
            'augment': self.augment,
            'use_jitter': self.use_jitter,
            'use_dropout': self.use_dropout,
            'jitter_sigma': self.jitter_sigma,
            'dropout_p': self.dropout_p
        }
        return info

if __name__ == "__main__":
    # Test different augmentation configurations
    print("Testing different augmentation configurations:")
    
    # Test 1: No augmentation
    print("\n1. No augmentation:")
    dataset1 = ModelNet10Dataset(root_dir='../data/ModelNet10',
                                split='train',
                                num_points=1024,
                                use_jitter=False,
                                use_dropout=False)
    
    # Test 2: Jitter only
    print("\n2. Jitter only:")
    dataset2 = ModelNet10Dataset(root_dir='../data/ModelNet10',
                                split='train',
                                num_points=1024,
                                use_jitter=True,
                                use_dropout=False)
    
    # Test 3: Dropout only
    print("\n3. Dropout only:")
    dataset3 = ModelNet10Dataset(root_dir='../data/ModelNet10',
                                split='train',
                                num_points=1024,
                                use_jitter=False,
                                use_dropout=True)
    
    # Test 4: Both augmentations
    print("\n4. Both augmentations:")
    dataset4 = ModelNet10Dataset(root_dir='../data/ModelNet10',
                                split='train',
                                num_points=1024,
                                use_jitter=True,
                                use_dropout=True)
    
    # Test data loading
    print("\n5. Testing data loader:")
    loader = DataLoader(dataset4, batch_size=8, shuffle=True)
    for pts, labels in loader:
        print("Points batch shape:", pts.shape)   # Expected: (8, 1024, 3)
        print("Labels batch shape:", labels.shape) # Expected: (8,)
        print("labels:", labels)
        break