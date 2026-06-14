import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# 1. 超参数（基准配置）
# ---------------------------
config = {
    'batch_size': 64,
    'epochs': 20,
    'lr': 0.001,
    'hidden_layers': [128, 128],   # 每层神经元个数，[] 表示无隐藏层，这里的列表表示两个隐藏层，每个隐藏层有 128 神经元
    'activation': 'relu',           # 'relu', 'tanh', 'sigmoid'
    'dropout': 0.2,                 # 实验 D 表明 0.2 的 Dropout 较佳，因此这里采用这个值
    'use_batch_norm': True,
}

# ---------------------------
# 2. 数据加载与预处理
# ---------------------------
transform = transforms.Compose([
    transforms.ToTensor(),          # 将 PIL 图像转为 Tensor，并除以 255 归一化到 [0,1]
    transforms.Lambda(lambda x: x.view(-1))  # 展平为 784 维向量
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 从训练集中划分 10% 作为验证集
train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

# ---------------------------
# 3. 可配置的 MLP 模型
# ---------------------------
def get_activation(name):
    if name == 'relu':
        return nn.ReLU()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise ValueError(f'Unknown activation: {name}')

class FlexibleMLP(nn.Module):
    def __init__(self, input_dim=784, output_dim=10, hidden_layers=[128,128],
                 activation='relu', dropout=0.0, use_batch_norm=False):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ---------------------------
# 4. 训练函数
# ---------------------------
def train_model(model, train_loader, val_loader, epochs, lr, device='cpu'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        train_loss = total_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 验证阶段
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item() * data.size(0)
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        val_loss = val_loss / total
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f'Epoch {epoch+1:2d}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')

    return train_losses, val_losses, train_accs, val_accs

# ---------------------------
# 5. 绘制损失与准确率曲线
# ---------------------------
def plot_curves(train_losses, val_losses, train_accs, val_accs, title_suffix=''):
    epochs = range(1, len(train_losses)+1)
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Loss Curves {title_suffix}')

    plt.subplot(1,2,2)
    plt.plot(epochs, train_accs, 'b-', label='Train Acc')
    plt.plot(epochs, val_accs, 'r-', label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'Accuracy Curves {title_suffix}')
    plt.show()

# ---------------------------
# 6. 运行基准模型
# ---------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

model = FlexibleMLP(input_dim=784, output_dim=10,
                    hidden_layers=config['hidden_layers'],
                    activation=config['activation'],
                    dropout=config['dropout'],
                    use_batch_norm=config['use_batch_norm'])
print(model)

train_losses, val_losses, train_accs, val_accs = train_model(
    model, train_loader, val_loader, epochs=config['epochs'],
    lr=config['lr'], device=device
)
plot_curves(train_losses, val_losses, train_accs, val_accs, title_suffix='(Baseline)')

# 最终测试集评估
model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = torch.max(output, 1)
        test_total += target.size(0)
        test_correct += (predicted == target).sum().item()
print(f'Test accuracy: {test_correct / test_total:.4f}')

'''
Using device: cpu
FlexibleMLP(
  (net): Sequential(
    (0): Linear(in_features=784, out_features=128, bias=True)
    (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
    (2): ReLU()
    (3): Dropout(p=0.2, inplace=False)
    (4): Linear(in_features=128, out_features=128, bias=True)
    (5): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
    (6): ReLU()
    (7): Dropout(p=0.2, inplace=False)
    (8): Linear(in_features=128, out_features=10, bias=True)
  )
)
Epoch  1/20 | Train Loss: 0.3125 | Train Acc: 0.9136 | Val Loss: 0.1295 | Val Acc: 0.9595
Epoch  2/20 | Train Loss: 0.1533 | Train Acc: 0.9524 | Val Loss: 0.0989 | Val Acc: 0.9693
Epoch  3/20 | Train Loss: 0.1265 | Train Acc: 0.9604 | Val Loss: 0.0913 | Val Acc: 0.9705
Epoch  4/20 | Train Loss: 0.1054 | Train Acc: 0.9666 | Val Loss: 0.0881 | Val Acc: 0.9725
Epoch  5/20 | Train Loss: 0.0948 | Train Acc: 0.9698 | Val Loss: 0.0773 | Val Acc: 0.9773
Epoch  6/20 | Train Loss: 0.0853 | Train Acc: 0.9723 | Val Loss: 0.0705 | Val Acc: 0.9788
Epoch  7/20 | Train Loss: 0.0796 | Train Acc: 0.9741 | Val Loss: 0.0749 | Val Acc: 0.9772
Epoch  8/20 | Train Loss: 0.0735 | Train Acc: 0.9761 | Val Loss: 0.0690 | Val Acc: 0.9808
Epoch  9/20 | Train Loss: 0.0681 | Train Acc: 0.9778 | Val Loss: 0.0716 | Val Acc: 0.9790
Epoch 10/20 | Train Loss: 0.0633 | Train Acc: 0.9791 | Val Loss: 0.0747 | Val Acc: 0.9778
Epoch 11/20 | Train Loss: 0.0611 | Train Acc: 0.9801 | Val Loss: 0.0702 | Val Acc: 0.9792
Epoch 12/20 | Train Loss: 0.0588 | Train Acc: 0.9808 | Val Loss: 0.0696 | Val Acc: 0.9797
Epoch 13/20 | Train Loss: 0.0562 | Train Acc: 0.9819 | Val Loss: 0.0694 | Val Acc: 0.9798
Epoch 14/20 | Train Loss: 0.0525 | Train Acc: 0.9830 | Val Loss: 0.0695 | Val Acc: 0.9803
Epoch 15/20 | Train Loss: 0.0523 | Train Acc: 0.9821 | Val Loss: 0.0732 | Val Acc: 0.9782
Epoch 16/20 | Train Loss: 0.0499 | Train Acc: 0.9836 | Val Loss: 0.0736 | Val Acc: 0.9788
Epoch 17/20 | Train Loss: 0.0501 | Train Acc: 0.9831 | Val Loss: 0.0676 | Val Acc: 0.9797
Epoch 18/20 | Train Loss: 0.0437 | Train Acc: 0.9848 | Val Loss: 0.0666 | Val Acc: 0.9808
Epoch 19/20 | Train Loss: 0.0439 | Train Acc: 0.9853 | Val Loss: 0.0684 | Val Acc: 0.9808
Epoch 20/20 | Train Loss: 0.0423 | Train Acc: 0.9860 | Val Loss: 0.0695 | Val Acc: 0.9788
Test accuracy: 0.9810
'''