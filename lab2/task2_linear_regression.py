import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# 生成 200 个样本，特征 x 在 [-3, 3] 上均匀分布，目标 y 服从 y = 2 * x + 1 + noise
m = 200
# uniform(): 生成均匀分布随机数：在给定的区间内，每一个数值被抽中的概率是完全相等的
X = np.random.uniform(-3, 3, (m, 1))          # shape: (200, 1)

true_w = 2.0
true_b = 1.0

# randn(): 与 uniform 相对，生成成态分布随机数
noise = np.random.randn(m, 1) * 0.5           # 标准差 0.5

y = true_w * X + true_b + noise               # shape: (200, 1)

# 由于 noise 的存在，我们希望模型预测的 w, b 分别落在 2.0 和 1.0 的邻域内即可

# 划分训练集与验证集
indices = np.random.permutation(m)
train_idx = indices[:160]
val_idx = indices[160:]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]

# 初始化参数
w = np.random.randn(1, 1) * 0.01   # shape: (1, 1)
b = np.random.randn(1, 1) * 0.01   # shape: (1, 1)

# 超参数定义
lr = 0.01
batch_size = 32
epochs = 1000

# 辅助容器，记录损失历史
train_loss_history = []
val_loss_history = []

# 核心训练循环
# 对于回归任务（输出的是实数，与“分类任务”相对应），损失函数定义为预测值与真实值之差的平方的平均值：L = (1/m) * Σ (y_pred_i - y_i)^2
for epoch in range(epochs):
    # 每轮打乱训练集
    perm = np.random.permutation(len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    # Mini-batch 迭代
    for i in range(0, len(X_train), batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]

        # 前向传播
        y_pred = X_batch @ w + b

        # 计算损失（仅用于记录，训练时可用可不用）
        # loss = np.mean((y_pred - y_batch) ** 2)

        # 反向传播（计算梯度）
        error = y_pred - y_batch

        # L = (1/m) * Σ (error_i)^2
        dw = (2 / batch_size) * (X_batch.T @ error)     # X_batch.T @ error 即 Σ(X_i * error_i), 此处梯度公式推导略
        db = (2 / batch_size) * np.sum(error)

        # 更新参数
        w = w - lr * dw
        b = b - lr * db

    # Epoch 结束后，记录训练集与验证集的平均损失（用于绘图）
    y_train_pred = X_train @ w + b
    train_loss = np.mean((y_train_pred - y_train) ** 2)
    
    y_val_pred = X_val @ w + b
    val_loss = np.mean((y_val_pred - y_val) ** 2)
    
    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

# 最终结果
print("\n=== Final Parameters ===")
print(f"w (learned): {w[0, 0]:.4f} | True w: {true_w}")
print(f"b (learned): {b[0, 0]:.4f} | True b: {true_b}")

# 曲线绘制
plt.figure(figsize=(8, 5))
plt.plot(range(epochs), train_loss_history, label='Train Loss')
plt.plot(range(epochs), val_loss_history, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('Gradient Descent Convergence')
plt.grid(True)
plt.show()

"""
Q:

我们的实验看上去是成功了，最终w=1.9902, b=1.0313，我有两个疑问：1. 我把epochs调到100000，预测的参数没有比1000时更精确，这是为什么？ 2. 为什么最终训练损失和验证损失都稳定在0.23附近？

A:

这两个现象同时出现，说明你的实验不仅成功了，而且**完美地验证了机器学习理论中的两个核心极限**。你的模型并没有“学坏”，而是已经触碰到了数据本身的“物理天花板”。

我们逐一拆解：

### 1. 为什么训练 100000 轮，参数精度反而没有提升？

因为**固定学习率（lr=0.01）下的 Mini-batch SGD 具有“收敛噪声”**。

-   **初期（0~1000 轮）**：参数距离最优解很远，梯度很大。信号强度远大于噪声，模型快速向 `(w=2, b=1)` 冲过去。
-   **后期（1000 轮之后）**：参数已经非常接近谷底，梯度变得**极其微小**（例如 0.001）。此时，你的更新公式是 `w = w - lr * dw`，`dw` 已经小到几乎没有了。而因为你使用的是 **Mini-batch（一次只看 32 个样本）**，每次计算出的 `dw` 并不是全量数据的精确梯度，而是带有**随机采样噪声**的梯度。
-   **“随机游走”现象**：当梯度信号本身已经弱到和采样噪声差不多大小时，SGD 就不再是“下山”，而是在谷底附近做“布朗运动”（随机徘徊）。它今天算出的 `dw` 可能略微偏正，明天略微偏负，导致 `w` 在 `1.990` 附近来回震荡，始终无法稳定在 `2.000000` 上。

**结论**：在固定学习率下，不是轮数越多越精确，而是**当参数进入最优邻域后，多余的轮数只是在“原地踏步”**。要想突破这个瓶颈，你需要引入“学习率衰减”（让后期步长自动缩小），但通常不需要，因为 1.990 已经极其优秀。

---

### 2. 为什么训练损失和验证损失都稳定在 0.23，而不是降到 0？

**因为数据里有“无法消除的噪声”，而 MSE 惩罚的就是这种噪声。**

我们来做个数学计算：

-   你在生成数据时，加了一个标准差为 `0.5` 的高斯噪声（`np.random.randn(m, 1) * 0.5`）。
-   方差（Variance） = 标准差² = `0.5² = 0.25`。

如果你真的找到了完美的 `w=2.0, b=1.0`，代入损失函数：
`L = (1/m) * Σ ( (2x + 1 + noise) - (2x + 1) )²`
`L = (1/m) * Σ (noise)²`
因为 `noise` 是均值为 0、方差为 0.25 的正态分布，所以 `Σ(noise)² / m ≈ 0.25`。

**你得到的 0.23 是什么？**  
这比理论极限 0.25 还要低一点点，这是**采样巧合**。因为你随机生成的这 200 个噪声点，它们的平均平方值恰好略小于 0.25。但无论怎么训练，**损失永远不可能低于这个噪声本身的方差**。

**这对你有什么启发？**  
如果哪天你在真实数据集上训练，发现损失降不下去，不一定是你代码写错了。很可能是**数据本身的噪声太大**，模型已经捕捉到了所有的“规律”（2x+1），剩下的全是“听天由命”的随机波动。强行增加轮数或层数，只会导致过拟合（去拟合这些噪声），而不是降低损失。

你现在可以放心地进入 Task 3 了，因为你已经亲身体验了“参数收敛”和“不可约误差”这两个最核心的工程概念。
"""