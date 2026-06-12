import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1. 生成数据（确保两类线性可分且无重叠）
torch.manual_seed(42)

car_volume = torch.rand(100, 1) * 1.0 + 0.5      # 0.5~1.5
car_weight = torch.rand(100, 1) * 0.7 + 0.3      # 0.3~1.0
car_data = torch.cat([car_volume, car_weight], dim=1)
car_label = torch.zeros(100, 1)

truck_volume = torch.rand(100, 1) * 2.0 + 3.0    # 3.0~5.0
truck_weight = torch.rand(100, 1) * 2.0 + 2.0    # 2.0~4.0
truck_data = torch.cat([truck_volume, truck_weight], dim=1)
truck_label = torch.ones(100, 1)

X = torch.cat([car_data, truck_data], dim=0)
y = torch.cat([car_label, truck_label], dim=0)
shuffle = torch.randperm(200)
X, y = X[shuffle], y[shuffle]

# 2. 特征归一化（关键步骤）
mean = X.mean(dim=0, keepdim=True)
std = X.std(dim=0, keepdim=True) + 1e-7
X_norm = (X - mean) / std

# 划分数据
X_norm_np = X_norm.numpy()
y_np = y.numpy()
X_train, X_test, y_train, y_test = train_test_split(
    X_norm_np, y_np, test_size=0.2, random_state=42
)

X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train)
X_test_t = torch.tensor(X_test)
y_test_t = torch.tensor(y_test)

# 3. 数据加载器
dataset = TensorDataset(X_norm, y)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# 4. 模型（隐藏层神经元增加到 8 个，学习率调低）
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

model = Classifier()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)   # Adam + 较小学习率

# 5. 训练
epochs = 500
for epoch in range(epochs):
    total_loss = 0
    for bx, by in loader:
        pred = model(bx)
        loss = criterion(pred, by)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 100 == 0:
        print(f"Epoch {epoch:3d}, Loss: {total_loss/len(loader):.4f}")

# 6. 评估
with torch.no_grad():
    pred_all = model(X_norm)
    pred_class = (pred_all > 0.5).float()
    acc = (pred_class == y).float().mean()
    print(f"\nAccuracy: {acc:.2%}")
    print(f"Min probability: {pred_all.min():.4f}")
    print(f"Max probability: {pred_all.max():.4f}")

# 7. 预测新样本（记得要归一化）
def predict(volume, weight):
    model.eval()
    raw = torch.tensor([[volume, weight]], dtype=torch.float32)
    raw_norm = (raw - mean) / std
    with torch.no_grad():
        prob = model(raw_norm).item()
    return "Truck" if prob > 0.5 else "Car", prob

# 测试
test_cases = [(1.0, 0.6), (4.0, 3.2)]
for v, w in test_cases:
    name, prob = predict(v, w)
    print(f"Volume={v:.1f}, Weight={w:.1f} -> {name} (prob={prob:.3f})")

boundary_samples = [
    (2.0, 1.5),   # 中间地带，按你的数据分布应该还是 Car（因为 Truck 体积>3.0）
    (2.8, 2.0),   # 接近 Truck 边界
    (3.0, 2.0),   # Truck 边界下限
    (1.8, 1.2),   # Car 上限附近
]
for v, w in boundary_samples:
    name, prob = predict(v, w)
    print(f"Volume={v:.1f}, Weight={w:.1f} -> {name} (prob={prob:.3f})")

with torch.no_grad():
    test_preds = model(X_test_t)
    test_acc = ((test_preds > 0.5).float() == y_test_t).float().mean()
    print(f"Test accuracy: {test_acc:.2%}")


