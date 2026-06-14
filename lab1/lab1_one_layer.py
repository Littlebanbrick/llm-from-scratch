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
    'hidden_layers': [128],   # 1 个隐藏层
    'activation': 'relu',           # 'relu', 'tanh', 'sigmoid'
    'dropout': 0.0,                 # dropout 概率，0 表示不使用
    'use_batch_norm': False,
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
    (1): ReLU()
    (2): Linear(in_features=128, out_features=10, bias=True)
  )
)
Epoch  1/20 | Train Loss: 0.3637 | Train Acc: 0.9011 | Val Loss: 0.2006 | Val Acc: 0.9448
Epoch  2/20 | Train Loss: 0.1689 | Train Acc: 0.9505 | Val Loss: 0.1392 | Val Acc: 0.9627
Epoch  3/20 | Train Loss: 0.1174 | Train Acc: 0.9662 | Val Loss: 0.1154 | Val Acc: 0.9678
Epoch  4/20 | Train Loss: 0.0883 | Train Acc: 0.9734 | Val Loss: 0.0975 | Val Acc: 0.9702
Epoch  5/20 | Train Loss: 0.0695 | Train Acc: 0.9789 | Val Loss: 0.0988 | Val Acc: 0.9705
Epoch  6/20 | Train Loss: 0.0558 | Train Acc: 0.9839 | Val Loss: 0.0838 | Val Acc: 0.9747
Epoch  7/20 | Train Loss: 0.0454 | Train Acc: 0.9866 | Val Loss: 0.0859 | Val Acc: 0.9738
Epoch  8/20 | Train Loss: 0.0376 | Train Acc: 0.9887 | Val Loss: 0.0791 | Val Acc: 0.9770
Epoch  9/20 | Train Loss: 0.0303 | Train Acc: 0.9916 | Val Loss: 0.0777 | Val Acc: 0.9760
Epoch 10/20 | Train Loss: 0.0242 | Train Acc: 0.9935 | Val Loss: 0.0789 | Val Acc: 0.9755
Epoch 11/20 | Train Loss: 0.0198 | Train Acc: 0.9946 | Val Loss: 0.0790 | Val Acc: 0.9760
Epoch 12/20 | Train Loss: 0.0171 | Train Acc: 0.9951 | Val Loss: 0.0736 | Val Acc: 0.9768
Epoch 13/20 | Train Loss: 0.0142 | Train Acc: 0.9964 | Val Loss: 0.0792 | Val Acc: 0.9762
Epoch 14/20 | Train Loss: 0.0111 | Train Acc: 0.9974 | Val Loss: 0.0853 | Val Acc: 0.9773
Epoch 15/20 | Train Loss: 0.0099 | Train Acc: 0.9977 | Val Loss: 0.0840 | Val Acc: 0.9783
Epoch 16/20 | Train Loss: 0.0077 | Train Acc: 0.9984 | Val Loss: 0.0774 | Val Acc: 0.9783
Epoch 17/20 | Train Loss: 0.0081 | Train Acc: 0.9982 | Val Loss: 0.0888 | Val Acc: 0.9777
Epoch 18/20 | Train Loss: 0.0061 | Train Acc: 0.9986 | Val Loss: 0.0908 | Val Acc: 0.9767
Epoch 19/20 | Train Loss: 0.0061 | Train Acc: 0.9987 | Val Loss: 0.0812 | Val Acc: 0.9787
Epoch 20/20 | Train Loss: 0.0037 | Train Acc: 0.9994 | Val Loss: 0.0941 | Val Acc: 0.9773
Test accuracy: 0.9764
'''