import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from data_augmentation import jitter, random_dropout

class ModelNet40Dataset(Dataset):
    def __init__(self, root_dir, split='train', num_points=1024, classes=None, augment=False, 
                 use_jitter=False, use_dropout=False, jitter_sigma=0.01, dropout_p=0.2):
        """
        Args:
            root_dir: Path to ModelNet40 directory
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
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        if not lines:
            raise ValueError(f'Empty file: {file_path}')
        
        line_idx = 0
        
        # Handle different OFF header formats
        first_line = lines[line_idx]
        
        if first_line == 'OFF':
            # Standard OFF format: OFF on separate line
            line_idx += 1
            if line_idx >= len(lines):
                raise ValueError(f'Incomplete OFF file: {file_path}')
            
            try:
                n_verts, n_faces, n_edges = map(int, lines[line_idx].split())
                line_idx += 1
            except ValueError:
                # Maybe the counts line contains floats (vertex data)
                # Try to read all remaining lines as vertices
                return self._read_vertices_from_lines(lines[line_idx:], file_path)
                
        elif first_line.startswith('OFF'):
            # Compact OFF format: OFF with counts on same line
            parts = first_line.split()
            if len(parts) >= 4:
                try:
                    n_verts, n_faces, n_edges = map(int, parts[1:4])
                    line_idx += 1
                except ValueError:
                    # If parsing fails, treat all lines as vertices
                    return self._read_vertices_from_lines(lines, file_path)
            else:
                # OFF without counts, read next line or treat all as vertices
                line_idx += 1
                if line_idx < len(lines):
                    try:
                        n_verts, n_faces, n_edges = map(int, lines[line_idx].split())
                        line_idx += 1
                    except ValueError:
                        return self._read_vertices_from_lines(lines[line_idx:], file_path)
                else:
                    return self._read_vertices_from_lines([], file_path)
        else:
            # No OFF header, try to parse first line as counts or as vertex data
            try:
                n_verts, n_faces, n_edges = map(int, first_line.split())
                line_idx += 1
            except ValueError:
                # First line is vertex data, read all lines as vertices
                return self._read_vertices_from_lines(lines, file_path)
        
        # Read vertices based on n_verts
        verts = []
        for i in range(n_verts):
            if line_idx + i >= len(lines):
                break
            line = lines[line_idx + i]
            try:
                vertex = list(map(float, line.split()[:3]))
                if len(vertex) >= 3:
                    verts.append(vertex[:3])
            except ValueError:
                print(f"Warning: Skipping invalid vertex line in {file_path}: {line}")
                continue
        
        if len(verts) == 0:
            raise ValueError(f'No valid vertices found in {file_path}')
        
        return np.array(verts, dtype=np.float32)
    
    def _read_vertices_from_lines(self, lines, file_path):
        """Helper method to read vertices from lines without count information."""
        verts = []
        for line in lines:
            if not line:
                continue
            try:
                # Try to parse as vertex coordinates
                coords = line.split()
                if len(coords) >= 3:
                    vertex = list(map(float, coords[:3]))
                    verts.append(vertex)
            except ValueError:
                # Skip lines that can't be parsed as floats
                continue
        
        if len(verts) == 0:
            raise ValueError(f'No valid vertices found in {file_path}')
        
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

    def get_class_names(self):
        """Return the list of class names."""
        return list(self.class_map.keys())

    def get_num_classes(self):
        """Return the number of classes."""
        return len(self.class_map)

if __name__ == "__main__":
    # Test different augmentation configurations
    print("Testing different augmentation configurations:")
    
    # Test 1: No augmentation
    print("\n1. No augmentation:")
    dataset1 = ModelNet40Dataset(root_dir='../data/ModelNet40',
                                split='train',
                                num_points=1024,
                                use_jitter=False,
                                use_dropout=False)
    
    # Test 2: Jitter only
    print("\n2. Jitter only:")
    dataset2 = ModelNet40Dataset(root_dir='../data/ModelNet40',
                                split='train',
                                num_points=1024,
                                use_jitter=True,
                                use_dropout=False)
    
    # Test 3: Dropout only
    print("\n3. Dropout only:")
    dataset3 = ModelNet40Dataset(root_dir='../data/ModelNet40',
                                split='train',
                                num_points=1024,
                                use_jitter=False,
                                use_dropout=True)
    
    # Test 4: Both augmentations
    print("\n4. Both augmentations:")
    dataset4 = ModelNet40Dataset(root_dir='../data/ModelNet40',
                                split='train',
                                num_points=1024,
                                use_jitter=True,
                                use_dropout=True)
    
    # Print dataset info
    print(f"\nDataset info:")
    print(f"Number of classes: {dataset4.get_num_classes()}")
    print(f"Class names: {dataset4.get_class_names()}")
    
    # Test data loading
    print("\n5. Testing data loader:")
    loader = DataLoader(dataset4, batch_size=8, shuffle=True)
    for pts, labels in loader:
        print("Points batch shape:", pts.shape)   # Expected: (8, 1024, 3)
        print("Labels batch shape:", labels.shape) # Expected: (8,)
        print("labels:", labels)
        break