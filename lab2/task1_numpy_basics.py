import numpy as np

# Ensure that every time the program runs,
# the pseudo random numbers generated will be the same
np.random.seed(42)

# Create a random array of 100*5, where elements have the mean value of 0
# and the variance of 1
X = np.random.randn(100, 5)

# axis = 0 / 1 means to opearte by row / column, respectively
# keepdims=True 保留了原始数组的维度
# mean: 均值, std: 标准差
col_mean = X.mean(axis=0, keepdims=True)   # shape (1, 5)
col_std = X.std(axis=0, keepdims=True)     # shape (1, 5)
row_mean = X.mean(axis=1, keepdims=True)   # shape (100, 1)
row_std = X.std(axis=1, keepdims=True)     # shape (100, 1)

# 对 X 进行标准化，使得每列的均值为 0，标准差为 1
X_norm = (X - col_mean) / col_std

# 验证
# axis = 0 是因为沿着行操作才能求出列均值
print(X_norm.mean(axis=0))
print(X_norm.std(axis=0))

# Matrices Multiplication
# .T: 转置, @: 矩阵乘法
mat1 = X @ X.T
mat2 = X.T @ X
print(mat1.shape, mat2.shape)

# Sigmoid func: σ(z) = 1 / (1 + exp(-z))
# Notive that when z is too large or too small, calculate directly will cause overflow
def sigmoid(z):
    # 转换成数组类型
    z = np.asarray(z)

    # 生成布尔掩码。结果是一个与 z 形状完全相同的布尔数组，其中非负元素的位置是 True，负元素的位置是 False
    positive_mask = z >= 0

    # 创建一个与 z 形状相同、但内容未初始化的空数组
    # 显式指定为浮点型防止截断
    result = np.empty_like(z, dtype=np.float64)
    
    # 对于 z >= 0，直接计算
    result[positive_mask] = 1.0 / (1.0 + np.exp(-z[positive_mask]))
    
    # 对于 z < 0，使用等价形式
    neg_mask = ~positive_mask
    result[neg_mask] = np.exp(z[neg_mask]) / (1.0 + np.exp(z[neg_mask]))
    
    return result

print(sigmoid(np.array([-1000, 0, 1000])))

# 计算二分类交叉熵损失: L = - (y * log(p) + (1 - y) * log(1 - p))
# Avoid log(0)
eps = 1e-15

y = np.random.randint(0, 2, size=100)          # shape (100,)
p = sigmoid(X_norm[:, 0])                      # shape (100,) 随意取一列作为预测
p_clipped = np.clip(p, eps, 1 - eps)
loss = -np.mean(y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped))
print(loss)