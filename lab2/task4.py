import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 固定随机种子，保证可复现
np.random.seed(42)

# ================= 1. 数据准备 =================
print("Loading and preprocessing data...")
X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
y = y.reshape(-1, 1)  # 统一形状为 (m, 1)

# 划分训练集 (80%) 和测试集 (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化（标准化后的数据能使梯度下降更稳定）
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ================= 2. 超参数与初始化 =================
print("Initializing parameters with He initialization...")
m = X_train.shape[0]
input_dim = 2
hidden_dim = 10
output_dim = 1
lr = 0.01
epochs = 3000
batch_size = 32

# He 初始化（专门针对 ReLU）
W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2 / input_dim)
b1 = np.zeros((1, hidden_dim))
W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2 / hidden_dim)
b2 = np.zeros((1, output_dim))

# ================= 3. 辅助函数 =================
def sigmoid(z):
    z = np.asarray(z)
    result = np.empty_like(z)
    mask = z >= 0
    result[mask] = 1.0 / (1.0 + np.exp(-z[mask]))
    result[~mask] = np.exp(z[~mask]) / (1.0 + np.exp(z[~mask]))
    return result

def compute_loss(p, y_true):
    eps = 1e-15
    p_clipped = np.clip(p, eps, 1 - eps)
    return -np.mean(y_true * np.log(p_clipped) + (1 - y_true) * np.log(1 - p_clipped))

def compute_accuracy(p, y_true):
    preds = (p > 0.5).astype(float)
    return np.mean(preds == y_true)


# ================= 4. 训练循环 =================
train_losses = []
test_losses = []
train_accs = []
test_accs = []

print("Starting training...")
for epoch in range(epochs):
    # 打乱训练数据
    perm = np.random.permutation(m)
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    for i in range(0, m, batch_size):
        # 获取 Mini-batch
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]
        batch_m = X_batch.shape[0]

        # --- 前向传播 ---
        Z1 = X_batch @ W1 + b1          # (batch_m, 10)
        A1 = np.maximum(0, Z1)          # (batch_m, 10), ReLU激活
        Z2 = A1 @ W2 + b2               # (batch_m, 1)
        A2 = sigmoid(Z2)                # (batch_m, 1)

        # --- 损失计算 (仅用于监控，不参与反传) ---
        loss = compute_loss(A2, y_batch)

        # --- 反向传播 (手写链式法则) ---
        # 1. 输出层梯度
        dZ2 = A2 - y_batch                  # (batch_m, 1)
        dW2 = (A1.T @ dZ2) / batch_m       # (10, 1)
        db2 = np.mean(dZ2, axis=0, keepdims=True)  # (1, 1)

        # 2. 反向传播至隐藏层 (通过 W2)
        dA1 = dZ2 @ W2.T                    # (batch_m, 10)
        
        # 3. 通过 ReLU 激活函数的门控 (关键：导数门)
        dZ1 = dA1 * (Z1 > 0)                # (batch_m, 10)，负数梯度被永久截断

        # 4. 输入层梯度
        dW1 = (X_batch.T @ dZ1) / batch_m  # (2, 10)
        db1 = np.mean(dZ1, axis=0, keepdims=True)  # (1, 10)

        # --- 梯度下降更新 ---
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

    # 每个 Epoch 结束后，计算全量训练集和测试集的表现
    # 注意：此时使用全部数据，不涉及 Batch，用于绘制平滑曲线
    train_pred = sigmoid(X_train @ W1 + b1) @ W2 + b2   # 注意这里相当于先过ReLU再过W2
    # 更正：前向传播必须按顺序: X -> W1+b1 -> ReLU -> W2+b2 -> Sigmoid
    train_pred = sigmoid(np.maximum(0, X_train @ W1 + b1) @ W2 + b2)
    test_pred = sigmoid(np.maximum(0, X_test @ W1 + b1) @ W2 + b2)

    train_loss = compute_loss(train_pred, y_train)
    test_loss = compute_loss(test_pred, y_test)
    train_acc = compute_accuracy(train_pred, y_train)
    test_acc = compute_accuracy(test_pred, y_test)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    if epoch % 300 == 0:
        print(f"Epoch {epoch:4d} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

# ================= 5. 结果可视化 =================
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curves')

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Acc')
plt.plot(test_accs, label='Test Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Curves')
plt.show()

# ================= 6. 决策边界绘制 =================
def plot_decision_boundary(X_set, y_set, title):
    x_min, x_max = X_set[:, 0].min() - 0.5, X_set[:, 0].max() + 0.5
    y_min, y_max = X_set[:, 1].min() - 0.5, X_set[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_scaled = scaler.transform(grid)  # 使用训练集的scaler
    
    # 前向传播预测网格点
    pred_grid = sigmoid(np.maximum(0, grid_scaled @ W1 + b1) @ W2 + b2)
    pred_grid = pred_grid.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, pred_grid, levels=20, cmap='RdBu', alpha=0.6)
    plt.contour(xx, yy, pred_grid, levels=[0.5], colors='black', linewidths=2)
    plt.scatter(X_set[:, 0], X_set[:, 1], c=y_set.ravel(), cmap='RdBu', edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.show()

plot_decision_boundary(X_test, y_test, "Task 4: Single Hidden Layer NN Decision Boundary")


# ================= 7. 梯度检查 =================
import torch
import torch.nn as nn
print("\n===== 开始 PyTorch 梯度检查 =====")

# 1. 固定一个 Batch (使用训练集的前 32 个样本)
batch_X = torch.tensor(X_train[:32], dtype=torch.float32)
batch_y = torch.tensor(y_train[:32], dtype=torch.float32)

# 2. 在 PyTorch 中重建完全相同的网络
class PTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

pt_model = PTModel()

# 3. 关键步骤：将 NumPy 权重同步到 PyTorch (注意转置！)
pt_model.fc1.weight.data = torch.tensor(W1.T, dtype=torch.float32)
pt_model.fc1.bias.data = torch.tensor(b1.flatten(), dtype=torch.float32)
pt_model.fc2.weight.data = torch.tensor(W2.T, dtype=torch.float32)
pt_model.fc2.bias.data = torch.tensor(b2.flatten(), dtype=torch.float32)

# 4. 将 NumPy 模型设为评估模式 (不干扰 BatchNorm/Dropout，这里没有)
# 注意：必须确保 NumPy 反向传播时用的是同样的数据

# 重新在 NumPy 中针对这 32 个样本算一次前向和反向
X_batch_np = X_train[:32]
y_batch_np = y_train[:32]
batch_m = 32

# NumPy 前向
Z1_np = X_batch_np @ W1 + b1
A1_np = np.maximum(0, Z1_np)
Z2_np = A1_np @ W2 + b2
A2_np = sigmoid(Z2_np)

# NumPy 反向
dZ2_np = A2_np - y_batch_np
dW2_np = (A1_np.T @ dZ2_np) / batch_m
db2_np = np.mean(dZ2_np, axis=0, keepdims=True)

dA1_np = dZ2_np @ W2.T
dZ1_np = dA1_np * (Z1_np > 0)
dW1_np = (X_batch_np.T @ dZ1_np) / batch_m
db1_np = np.mean(dZ1_np, axis=0, keepdims=True)

# 5. PyTorch 前向与反向
pt_pred = pt_model(batch_X)
loss_fn = nn.BCELoss()
pt_loss = loss_fn(pt_pred, batch_y)

# 清除之前可能存在的梯度
pt_model.zero_grad()
# 反向传播
pt_loss.backward()

# 6. 提取 PyTorch 梯度 (注意转置回来以匹配 NumPy 形状)
pt_dW1 = pt_model.fc1.weight.grad.numpy().T  # (2, 10)
pt_db1 = pt_model.fc1.bias.grad.numpy().reshape(1, -1)  # (1, 10)
pt_dW2 = pt_model.fc2.weight.grad.numpy().T  # (10, 1)
pt_db2 = pt_model.fc2.bias.grad.numpy().reshape(1, -1)  # (1, 1)

# 7. 比较差值
print("\n----- 梯度差值对比 -----")
diff_W1 = np.abs(dW1_np - pt_dW1).max()
diff_b1 = np.abs(db1_np - pt_db1).max()
diff_W2 = np.abs(dW2_np - pt_dW2).max()
diff_b2 = np.abs(db2_np - pt_db2).max()

print(f"dW1 最大绝对差值: {diff_W1:.2e}")
print(f"db1 最大绝对差值: {diff_b1:.2e}")
print(f"dW2 最大绝对差值: {diff_W2:.2e}")
print(f"db2 最大绝对差值: {diff_b2:.2e}")

if max(diff_W1, diff_b1, diff_W2, diff_b2) < 1e-6:
    print("\n[验证通过]")
else:
    print("\n[验证失败]")