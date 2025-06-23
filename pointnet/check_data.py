import numpy as np
import matplotlib.pyplot as plt

def load_off(file):
    """Load a .off file and return vertices and faces."""
    with open(file, 'r') as f:
        if 'OFF' != f.readline().strip():
            raise ValueError('Not a valid OFF header')
        
        n_verts, n_faces, _ = tuple(map(int, f.readline().strip().split()))
        verts = []
        for _ in range(n_verts):
            verts.append(list(map(float, f.readline().strip().split())))
        verts = np.array(verts)
        
        # Faces (not used for point cloud visualization)
        _ = [f.readline() for _ in range(n_faces)]
        
    return verts

def plot_point_cloud(verts):
    """Plot the point cloud using matplotlib."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

if __name__ == "__main__":
    # Example path: update as needed
    off_path = 'data/archive/ModelNet10/chair/train/chair_0056.off'
    verts = load_off(off_path)
    plot_point_cloud(verts)
