import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier


def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CNN1D(nn.Module):
    def __init__(self, n_features, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.BatchNorm1d(32), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.BatchNorm1d(64), nn.AdaptiveAvgPool1d(8),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(64*8, hidden), nn.ReLU(), nn.Dropout(0.25), nn.Linear(hidden, 2)
        )

    def forward(self, x):
        return self.head(self.net(x))


def train_rf(X, y, n_estimators=250, seed=42):
    model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=seed, n_jobs=-1,
        class_weight='balanced_subsample', min_samples_leaf=1
    )
    model.fit(X, y)
    return model


def train_cnn(X, y, n_features, epochs=8, batch_size=512, lr=1e-3,
              weight_decay=1e-5, hidden=128, device=None, seed=42):
    seed_everything(seed)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN1D(n_features, hidden).to(device)
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    model.train()
    for ep in range(epochs):
        total = 0.0
        for xb, yb in loader:
            xb = xb.to(device).unsqueeze(1); yb = yb.to(device)
            opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
            total += loss.item() * len(yb)
        history.append(total / len(ds))
        print(f'Epoch {ep+1}/{epochs} loss={history[-1]:.4f}')
    return model, history, device


def cnn_proba(model, X, device=None, batch_size=2048):
    device = device or next(model.parameters()).device
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i+batch_size]).float().to(device).unsqueeze(1)
            out.append(torch.softmax(model(xb), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.array([], dtype='float32')


def cnn_predict(model, X, device=None, threshold=0.5):
    return (cnn_proba(model, X, device) >= threshold).astype(int)
