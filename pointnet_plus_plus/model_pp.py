# model_pp.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def index_points(points, idx):
    """
    Utility for indexing points given indices.
    points: (B, N, C), idx: (B, S) or (B, S, nsample)
    returns: new points (B, ...) including C dim
    """
    B = points.size(0)
    idx_shape = idx.shape
    offset = torch.arange(B, device=points.device).view(B, *([1] * (idx.dim()-1))) * points.size(1)
    flat_points = points.reshape(-1, points.size(-1))
    flat_idx = (idx + offset).reshape(-1)
    new_points = flat_points[flat_idx]
    return new_points.view(*idx_shape, -1)


def farthest_point_sampling(xyz, npoint):
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.full((B, N), 1e10, device=xyz.device)
    farthest = torch.randint(0, N, (B,), device=xyz.device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(B), farthest].unsqueeze(1)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0,2,1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def query_ball_point(radius, nsample, xyz, new_xyz):
    B, N, _ = xyz.shape
    S = new_xyz.size(1)
    sqrdists = square_distance(new_xyz, xyz)
    group_idx = torch.zeros(B, S, nsample, dtype=torch.long, device=xyz.device)
    for b in range(B):
        for i in range(S):
            dist = sqrdists[b,i]
            idx = torch.where(dist <= radius**2)[0]
            if idx.numel() == 0:
                idx = torch.zeros(nsample, dtype=torch.long, device=xyz.device)
            elif idx.numel() > nsample:
                idx = idx[torch.randperm(idx.numel(), device=xyz.device)[:nsample]]
            else:
                pad = idx[torch.randint(0, idx.numel(), (nsample-idx.numel(),), device=xyz.device)]
                idx = torch.cat([idx, pad], dim=0)
            group_idx[b,i] = idx
    return group_idx


class TNet(nn.Module):
    """
    Spatial transformer network, as in PointNet.
    """
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):  # x: (B, k, N)
        B = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        iden = torch.eye(self.k, device=x.device).view(1, self.k*self.k).repeat(B,1)
        x = x + iden
        return x.view(B, self.k, self.k)


class PointNetSetAbstraction(nn.Module):
    """Single Set Abstraction layer: sample, group, learn, pool."""
    def __init__(self, npoint, radius, nsample, in_channel, mlp_channels):
        super().__init__()
        self.npoint, self.radius, self.nsample = npoint, radius, nsample
        layers=[]
        last = in_channel+3
        for out in mlp_channels:
            layers += [nn.Conv2d(last, out,1,bias=False), nn.BatchNorm2d(out), nn.ReLU()]
            last = out
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, points=None):
        # xyz: (B,N,3), points: (B,C,N) or None
        fps_idx = farthest_point_sampling(xyz, self.npoint)
        new_xyz = index_points(xyz, fps_idx)
        idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx) - new_xyz.unsqueeze(2)
        if points is not None:
            grouped_pts = index_points(points.permute(0,2,1), idx).permute(0,3,1,2)
            feats = torch.cat([grouped_xyz.permute(0,3,1,2), grouped_pts], dim=1)
        else:
            feats = grouped_xyz.permute(0,3,1,2)
        feats = self.mlp(feats)
        return new_xyz, torch.max(feats, -1)[0]


class PointNetPlusPlus(nn.Module):
    """PointNet++ with input & feature transforms and proper global normalization."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.input_tnet = TNet(3)
        self.sa1 = PointNetSetAbstraction(512, 0.2, 32, 0, [64,64,128])
        self.sa2 = PointNetSetAbstraction(128, 0.4, 64, 128, [128,128,256])
        self.global_mlp = nn.Sequential(
            nn.Conv1d(256,512,1,bias=False), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Conv1d(512,1024,1,bias=False), nn.BatchNorm1d(1024), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(1024,512,bias=False), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512,256,bias=False), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256,num_classes)
        )

    def forward(self, xyz, _=None):
        # normalize per-cloud
        centroid = xyz.mean(1, keepdim=True)
        xyz = xyz - centroid
        furthest = torch.max(torch.norm(xyz, dim=2, keepdim=True), dim=1, keepdim=True)[0]
        xyz = xyz / furthest
        # input transform
        trans = self.input_tnet(xyz.permute(0,2,1))
        xyz = torch.bmm(trans, xyz.permute(0,2,1)).permute(0,2,1)
        # SA layers
        l1_xyz, l1_feats = self.sa1(xyz)
        l2_xyz, l2_feats = self.sa2(l1_xyz, l1_feats)
        # global
        x = self.global_mlp(l2_feats)
        x = torch.max(x,2)[0]
        # classifier
        logits = self.fc(x)
        return F.log_softmax(logits, dim=1)
