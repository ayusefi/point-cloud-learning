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
            augment: Boolean, apply data augmentation if True
        """
        self.root_dir = root_dir
        self.split = split
        self.num_points = num_points
        self.augment = augment
        self.files = []
        self.class_map = {}
        all_classes = sorted(os.listdir(root_dir))
        if classes is None:
            classes = all_classes
        for idx, cls in enumerate(classes):
            self.class_map[cls] = idx
            pattern = os.path.join(root_dir, cls, split, '*.off')
            for f in glob.glob(pattern):
                self.files.append((f, idx))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path, label = self.files[idx]
        verts = self.load_off(file_path)
        verts = self.sample_points(verts)
        if self.augment:
            verts = jitter(verts)
            verts = random_dropout(verts)
        return torch.tensor(verts, dtype=torch.float32), label

    def load_off(self, file):
        with open(file, 'r') as f:
            if 'OFF' != f.readline().strip():
                raise ValueError('Not a valid OFF file')
            n_verts, n_faces, _ = map(int, f.readline().strip().split())
            verts = [list(map(float, f.readline().strip().split())) for _ in range(n_verts)]
            verts = np.array(verts)
            return verts

    def sample_points(self, verts):
        if len(verts) >= self.num_points:
            indices = np.random.choice(len(verts), self.num_points, replace=False)
        else:
            indices = np.random.choice(len(verts), self.num_points, replace=True)
        return verts[indices]

if __name__ == "__main__":
    dataset = ModelNet10Dataset(root_dir='data/ModelNet10', split='train', num_points=1024, augment=True)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    for points, labels in dataloader:
        print("Points batch shape:", points.shape)  # Expected: (8, 1024, 3)
        print("Labels batch shape:", labels.shape)  # Expected: (8,)
        break
