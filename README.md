# Point Cloud Learning with PointNet

This project implements and trains a basic **PointNet** architecture for 3D shape classification using the **ModelNet10** dataset.  
It is part of an exploration into deep learning methods for point cloud processing.

---

## 🚀 Goal

The objective of this project is to:
- Build a minimal yet functional PointNet model from scratch in PyTorch.
- Train it on the ModelNet10 dataset.
- Gain practical understanding of 3D point cloud classification.

---

## 🧠 Method

**PointNet** is a deep neural network architecture designed to directly process unordered point sets (point clouds).  
Key components:
- An **input T-Net** that learns to align input points into a canonical space (learns a 3×3 transform).
- A shared **MLP (Multi-Layer Perceptron)** applied to each point independently.
- A **global max pooling layer** that aggregates per-point features into a global descriptor.
- A **feature T-Net** (optional) that aligns intermediate features (64×64 transform).
- A final MLP for classification.

Our implementation includes:
- Both input and feature T-Nets.
- Standard MLP head.
- Cross-entropy loss (via `NLLLoss` with `log_softmax` outputs).
- Adam optimizer.

---

## 📈 Results

| Metric            | Value       |
|-------------------|------------|
| Final Train Acc    | ~61.5%     |
| Final Test Acc     | ~11.7%     |

> **Note:**  
The model showed good learning on the training set but poor generalization to the test set. This suggests potential overfitting, data mismatch, or need for better regularization/augmentation.

---

## ⚙️ How to Run

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/point-cloud-learning.git
cd point-cloud-learning
