import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
import numpy as np

# ---------------------------
# 1. Generate overlapping data (500 each class -> 1000 total)
# ---------------------------
torch.manual_seed(42)

n_samples = 500

# Car: volume 0.5~2.3, weight 0.5~2.0
car_volume = torch.rand(n_samples, 1) * 1.8 + 0.5
car_weight = torch.rand(n_samples, 1) * 1.5 + 0.5
car_data = torch.cat([car_volume, car_weight], dim=1)
car_label = torch.zeros(n_samples, 1)

# Truck: volume 2.0~5.0, weight 1.5~4.0
truck_volume = torch.rand(n_samples, 1) * 3.0 + 2.0
truck_weight = torch.rand(n_samples, 1) * 2.5 + 1.5
truck_data = torch.cat([truck_volume, truck_weight], dim=1)
truck_label = torch.ones(n_samples, 1)

X = torch.cat([car_data, truck_data], dim=0)
y = torch.cat([car_label, truck_label], dim=0)
shuffle = torch.randperm(2 * n_samples)
X, y = X[shuffle], y[shuffle]

# ---------------------------
# 2. Normalization (using full dataset)
# ---------------------------
mean = X.mean(dim=0, keepdim=True)
std = X.std(dim=0, keepdim=True) + 1e-7
X_norm = (X - mean) / std

# ---------------------------
# 3. Split into train+val and test (80% train+val, 20% test)
# ---------------------------
X_norm_np = X_norm.numpy()
y_np = y.numpy()
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_norm_np, y_np, test_size=0.2, random_state=42
)

X_trainval_t = torch.tensor(X_trainval, dtype=torch.float32)
y_trainval_t = torch.tensor(y_trainval, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

# ---------------------------
# 4. Model definition with Dropout and weight decay
# ---------------------------
class Classifier(nn.Module):
    def __init__(self, dropout_prob=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),          # increased hidden size
            nn.ReLU(),
            nn.Dropout(dropout_prob),  # dropout for regularization
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# ---------------------------
# 5. Training with validation set and early stopping
# ---------------------------
def train_with_validation(X_train, y_train, X_val, y_val, epochs=500, patience=20, weight_decay=1e-4):
    model = Classifier(dropout_prob=0.2)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=weight_decay)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        for bx, by in train_loader:
            pred = model(bx)
            loss = criterion(pred, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()
            val_acc = ((val_pred > 0.5).float() == y_val).float().mean().item()

        if epoch % 100 == 0:
            print(f"Epoch {epoch:3d}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_model_state)
    return model

# Split trainval into train and validation (80% of trainval -> train, 20% -> val)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval_t, y_trainval_t, test_size=0.2, random_state=42
)

print("=== Training with validation, dropout, and weight decay ===")
model = train_with_validation(X_train, y_train, X_val, y_val, epochs=500, patience=20, weight_decay=1e-4)

# ---------------------------
# 6. Evaluation on test set
# ---------------------------
with torch.no_grad():
    test_preds = model(X_test_t)
    test_acc = ((test_preds > 0.5).float() == y_test_t).float().mean()
    print(f"\nFinal test accuracy: {test_acc:.2%}")

# ---------------------------
# 7. K-Fold Cross Validation (5 folds) with the same model
# ---------------------------
print("\n=== 5-Fold Cross Validation ===")
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_trainval_t)):
    X_train_fold = X_trainval_t[train_idx]
    y_train_fold = y_trainval_t[train_idx]
    X_val_fold = X_trainval_t[val_idx]
    y_val_fold = y_trainval_t[val_idx]

    model_fold = Classifier(dropout_prob=0.2)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model_fold.parameters(), lr=0.01, weight_decay=1e-4)

    train_dataset_fold = TensorDataset(X_train_fold, y_train_fold)
    train_loader_fold = DataLoader(train_dataset_fold, batch_size=32, shuffle=True)

    # Train for fixed epochs (200 for speed)
    for epoch in range(200):
        model_fold.train()
        for bx, by in train_loader_fold:
            pred = model_fold(bx)
            loss = criterion(pred, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model_fold.eval()
    with torch.no_grad():
        val_pred_fold = model_fold(X_val_fold)
        val_acc_fold = ((val_pred_fold > 0.5).float() == y_val_fold).float().mean().item()
    fold_accuracies.append(val_acc_fold)
    print(f"Fold {fold+1}, Validation Accuracy: {val_acc_fold:.2%}")

print(f"Average CV Accuracy: {np.mean(fold_accuracies):.2%} (+/- {np.std(fold_accuracies):.2%})")

# ---------------------------
# 8. Probability heatmap (decision boundary)
# ---------------------------
def plot_decision_boundary(model, mean, std):
    model.eval()
    x_min, x_max = 0, 6
    y_min, y_max = 0, 5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_t = torch.tensor(grid, dtype=torch.float32)
    grid_norm = (grid_t - mean) / std
    with torch.no_grad():
        Z = model(grid_norm).numpy().reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.6)
    plt.colorbar(label='P(Truck)')
    plt.scatter(X[:,0], X[:,1], c=y.flatten(), cmap='RdBu', edgecolor='k', alpha=0.8)
    plt.xlabel('Volume')
    plt.ylabel('Weight')
    plt.title('Decision Boundary with Dropout & Weight Decay')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()

plot_decision_boundary(model, mean, std)

# ---------------------------
# 9. Predict on sample inputs
# ---------------------------
def predict(volume, weight):
    model.eval()
    raw = torch.tensor([[volume, weight]], dtype=torch.float32)
    raw_norm = (raw - mean) / std
    with torch.no_grad():
        prob = model(raw_norm).item()
    return "Truck" if prob > 0.5 else "Car", prob

sample_input = [
    (1.0, 0.6),
    (4.0, 3.2),
    (2.1, 1.6),
    (2.0, 1.5),
    (2.3, 2.0),
    (2.5, 1.8),
    (1.8, 2.2),
    (2.2, 1.4),
]

print("\n=== Sample Predictions ===")
for v, w in sample_input:
    name, prob = predict(v, w)
    print(f"Volume={v:.1f}, Weight={w:.1f} -> {name} (prob={prob:.3f})")
