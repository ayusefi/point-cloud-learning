# model_pp.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def index_points(points, idx):
    """
    Utility for indexing points given indices.
    """
    B = points.shape[0]
    idx_shape = idx.shape
    offset = torch.arange(B, dtype=idx.dtype, device=idx.device).view(B, *([1] * (idx.dim()-1))) * points.shape[1]
    flat_points = points.reshape(B * points.shape[1], -1)
    flat_idx = (idx + offset).reshape(-1)
    new_points = flat_points[flat_idx]
    return new_points.view(*idx_shape, -1)


def farthest_point_sampling(xyz, npoint):
    """Iterative FPS to sample npoint indices from xyz (B,N,3)."""
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.full((B, N), float('inf'), device=xyz.device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=xyz.device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(B), farthest].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def square_distance(src, dst):
    """Compute pairwise squared distances between src and dst."""
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def query_ball_point(radius, nsample, xyz, new_xyz):
    """Group local regions within radius for each centroid in new_xyz."""
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]
    sqrdists = square_distance(new_xyz, xyz)
    group_idx = torch.zeros(B, S, nsample, dtype=torch.long, device=xyz.device)
    for b in range(B):
        for i in range(S):
            dist = sqrdists[b, i]
            within = torch.nonzero(dist <= radius**2).squeeze(-1)
            if within.numel() == 0:
                idx = torch.zeros(nsample, dtype=torch.long, device=xyz.device)
            elif within.numel() >= nsample:
                idx = within[torch.randperm(within.numel(), device=xyz.device)[:nsample]]
            else:
                pad = within[torch.randint(0, within.numel(), (nsample-within.numel(),), device=xyz.device)]
                idx = torch.cat([within, pad], dim=0)
            group_idx[b, i] = idx
    return group_idx


class PointNetSetAbstraction(nn.Module):
    """Single Set Abstraction layer: sample, group, learn, pool."""
    def __init__(self, npoint, radius, nsample, in_channel, mlp_channels):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        layers = []
        last_ch = in_channel + 3
        for out_ch in mlp_channels:
            layers += [nn.Conv2d(last_ch, out_ch, 1, bias=False),
                       nn.BatchNorm2d(out_ch),
                       nn.ReLU(inplace=True)]
            last_ch = out_ch
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, points=None):
        B, N, _ = xyz.shape
        # 1) FPS sample
        idx = farthest_point_sampling(xyz, self.npoint)
        new_xyz = index_points(xyz, idx)
        # 2) Group
        group_idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, group_idx) - new_xyz.unsqueeze(2)
        # 3) Concat features
        if points is not None:
            grouped_pts = index_points(points.permute(0,2,1), group_idx)
            grouped_pts = grouped_pts.permute(0,3,1,2)
            feats = torch.cat([grouped_xyz.permute(0,3,1,2), grouped_pts], dim=1)
        else:
            feats = grouped_xyz.permute(0,3,1,2)
        # 4) Shared MLP + max pool
        feats = self.mlp(feats)
        feats = torch.max(feats, -1)[0]
        return new_xyz, feats


class PointNetPlusPlus(nn.Module):
    """Complete PointNet++ for classification, with optional feature transform stub."""
    def __init__(self, num_classes=10, input_channels=0, feature_transform=False):
        super().__init__()
        self.feature_transform = feature_transform
        # SA layers
        self.sa1 = PointNetSetAbstraction(512, 0.2, 32, input_channels, [64,64,128])
        self.sa2 = PointNetSetAbstraction(128, 0.4, 64, 128, [128,128,256])
        # Global feature: MLP then max-pool across all points
        self.global_mlp = nn.Sequential(
            nn.Conv1d(256, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 1024, 1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True)
        )
        # Classifier head
        self.fc_layers = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, xyz, features=None):
        # Forward through SA layers
        l1_xyz, l1_feats = self.sa1(xyz, features)
        l2_xyz, l2_feats = self.sa2(l1_xyz, l1_feats)
        # Global MLP + max-pool
        x = self.global_mlp(l2_feats)
        x = torch.max(x, 2)[0]
        # Classifier
        logits = self.fc_layers(x)
        if self.feature_transform:
            # Stub: return identity transform for regularizer
            B = xyz.size(0)
            trans_feat = torch.eye(3, device=xyz.device).unsqueeze(0).repeat(B, 1, 1)
            return logits, trans_feat
        else:
            return logits