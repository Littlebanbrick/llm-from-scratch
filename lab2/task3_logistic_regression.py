import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

np.random.seed(42)

# 使用 sklearn 的 make_moons 生成“月牙形”数据
# 生成 300 个样本，噪声 0.2（数据会有轻微交错）
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)

# 将标签 y 从 (300,) 重塑为 (300, 1) 以保持一致
y = y.reshape(-1, 1)

# 划分训练集 (80%) 和测试集 (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 计算训练集的均值和标准差
mean = X_train.mean(axis=0, keepdims=True)
std = X_train.std(axis=0, keepdims=True) + 1e-7

# 标准化
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# 初始化参数与超参数
w = np.random.randn(2, 1) * 0.01   # 形状: (2, 1)
b = np.random.randn(1, 1) * 0.01   # 形状: (1, 1)

lr = 0.1
batch_size = 32
epochs = 2000

# 注意：由于现在是分类任务，我们必须使用二分类交叉熵作为损失函数，这里复用task1的内容
from task1_numpy_basics import sigmoid

# 计算损失
def compute_loss(p, y_true):
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

# 训练循环
train_losses, test_losses = [], []
train_accs, test_accs = [], []

for epoch in range(epochs):
    # 打乱数据
    perm = np.random.permutation(len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    for i in range(0, len(X_train), batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]

        # 前向传播
        z = X_batch @ w + b
        p = sigmoid(z)

        # 计算梯度
        dz = p - y_batch
        dw = (X_batch.T @ dz) / batch_size
        db = np.mean(dz, axis=0, keepdims=True)

        # 更新参数
        w = w - lr * dw
        b = b - lr * db

    # 记录本轮结果 (使用全量数据)
    # 记住：参数更新用 Batch，模型监控用全量
    train_p = sigmoid(X_train @ w + b)
    test_p = sigmoid(X_test @ w + b)

    train_loss = compute_loss(train_p, y_train)
    test_loss = compute_loss(test_p, y_test)
    train_acc = np.mean((train_p > 0.5).astype(float) == y_train)
    test_acc = np.mean((test_p > 0.5).astype(float) == y_test)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

# 绘图
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Acc')
plt.plot(test_accs, label='Test Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 决策边界可视化
# 确定网格范围（加上一点边距）
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# 注意：网格点也需要标准化！
grid = np.c_[xx.ravel(), yy.ravel()]
grid_norm = (grid - mean) / std
z_grid = sigmoid(grid_norm @ w + b)
z_grid = z_grid.reshape(xx.shape)

plt.figure(figsize=(8, 6))
# 绘制概率等高线 (0.5 是决策边界)
plt.contourf(xx, yy, z_grid, levels=20, cmap='RdBu', alpha=0.6)
plt.contour(xx, yy, z_grid, levels=[0.5], colors='black', linewidths=2)  # 高亮 0.5 线

# 绘制原始数据点
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train.ravel(), cmap='RdBu', edgecolors='k', label='Train')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test.ravel(), cmap='RdBu', edgecolors='k', marker='s', label='Test')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Decision Boundary')
plt.legend()
plt.show()

'''
关于结果的分析（非常重要）
准确率预期：make_moons 不是线性可分的（两个月牙形状交错）。逻辑回归是一条直线，所以它的准确率上限大约在 85% ~ 90% 之间。这是正常的，不是欠拟合。

决策边界：你画出的 levels=[0.5] 那条黑线，就是逻辑回归找出的最佳直线（或更准确地说，是线性平面在二维空间的投影）。

与 Task 4 的关联：记住这个 85% 的准确率。在 Task 4 中，你会在上面加一个隐藏层（MLP），你会看到准确率飙升到 95% 以上，那条直线会扭曲成一条完美的曲线（月牙形）。这正是深度学习存在的意义。
'''