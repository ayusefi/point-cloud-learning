import numpy as np

def jitter(points, sigma=0.01):
    """
    Add Gaussian noise to the point cloud.
    Args:
        points (numpy array): An array of shape (N, 3) representing point coordinates.
        sigma (float): Standard deviation of the Gaussian noise.
    Returns:
        numpy array: The jittered point cloud.
    """
    noise = np.random.normal(0, sigma, points.shape)
    return points + noise

def random_dropout(points, p=0.2):
    """
    Randomly drop a fraction of points from the point cloud.
    Args:
        points (numpy array): An array of shape (N, 3) representing point coordinates.
        p (float): Fraction of points to drop.
    Returns:
        numpy array: The point cloud after random dropout.
    """
    N = points.shape[0]
    keep_mask = np.random.rand(N) >= p
    return points[keep_mask]
