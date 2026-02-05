import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt

# ===============================
# 1. 全局参数
# ===============================
DATA_PATH = r"D:\machine_learning\data\KX01\HRSN01_clean.csv"

L_IN = 32
H = 6
EPOCHS = 30
BATCH_SIZE = 64
LR = 0.0009113878901512495
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RAMP_THR = 1.0
ENERGY_THR = 0.5

LAMBDA_GATE = 0.14530436942798092
LAMBDA_CONS = 0.0006287740407869141
SMOOTH = 0.16668586091239587
ALPHA_START = 0.8068823326132036
ALPHA_END = 0.10200457484455444

LAMBDA_FLUCT = 0.7
LAMBDA_PEAK = 0.3
PEAK_Q = 0.9

# ===============================
# 2. 数据读取与标准化
# ===============================
df = pd.read_csv(DATA_PATH)
df["power"] = pd.to_numeric(df["power"], errors="coerce")
df = df.dropna(subset=["power"]).reset_index(drop=True)

power = df["power"].values.astype(np.float32)
mean_p = power.mean()
std_p = power.std() if power.std() > 1e-6 else 1.0
power = (power - mean_p) / std_p

# ===============================
# 3. 多尺度分解
# ===============================
def multi_scale_decompose(x):
    try:
        import pywt
        coeffs = pywt.wavedec(x, 'db4', level=2)
        low = pywt.waverec([coeffs[0]] + [None]*(len(coeffs)-1), 'db4')
        high = x[:len(low)] - low[:len(x)]
        return x[:len(low)], low[:len(x)], high[:len(x)]
    except:
        win = 5
        low = np.convolve(x, np.ones(win)/win, mode='same')
        high = x - low
        return x, low, high

# ===============================
# 4. 伪标签判别
# ===============================
def judge_state(x, ramp_thr=RAMP_THR, energy_thr=ENERGY_THR):
    ramp = np.max(x) - np.min(x)
    energy = np.mean((x[1:] - x[:-1]) ** 2)
    return 0 if (ramp < ramp_thr and energy < energy_thr) else 1

# ===============================
# 5. 数据集
# ===============================
class PowerDataset(Dataset):
    def __init__(self, series, L_in, H):
        self.samples = []
        T = len(series)
        for t in range(L_in, T - H):
            raw = series[t - L_in : t]
            raw, low, high = multi_scale_decompose(raw)
            y = series[t : t + H]
            state = judge_state(raw)
            x = np.stack([raw, low, high], axis=1)
            self.samples.append((x, y, state))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, state = self.samples[idx]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(state, dtype=torch.float32)
        )

# ===============================
# 6. 模型定义
# ===============================
class SharedEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return h[-1]

class GatingNet(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, h):
        p_stable = self.fc(h)
        p_fluct = 1.0 - p_stable
        return p_stable, p_fluct

class ExpertPredictor(nn.Module):
    def __init__(self, hidden_size, H):
        super().__init__()
        self.fc = nn.Linear(hidden_size, H)

    def forward(self, h):
        return self.fc(h)

class AdaptiveForecastModel(nn.Module):
    def __init__(self, input_dim=3, hidden_size=64, H=6, use_mix=True, use_feature_flip=True):
        super().__init__()
        self.mix = nn.Linear(input_dim, input_dim) if use_mix else nn.Identity()
        self.use_feature_flip = use_feature_flip

        self.encoder = SharedEncoder(input_dim, hidden_size)
        self.gate = GatingNet(hidden_size)
        self.expert_stable = ExpertPredictor(hidden_size, H)
        self.expert_fluct = ExpertPredictor(hidden_size, H)

    def forward(self, x):
        x_mix = self.mix(x)

        h = self.encoder(x_mix)
        p_stable, p_fluct = self.gate(h)

        x_t = torch.flip(x_mix, dims=[1])
        h_t = self.encoder(x_t)

        y_stable_fwd = self.expert_stable(h)
        y_stable_rev = self.expert_stable(h_t)
        y_stable = 0.5 * y_stable_fwd + 0.5 * torch.flip(y_stable_rev, dims=[1])

        y_fluct_fwd = self.expert_fluct(h)

        if self.use_feature_flip:
            x_f = torch.flip(x_mix, dims=[2])
            h_f = self.encoder(x_f)
            y_fluct_rev = self.expert_fluct(h_f)
            y_fluct = 0.5 * y_fluct_fwd + 0.5 * y_fluct_rev
        else:
            y_fluct = y_fluct_fwd

        y_pred = p_stable * y_stable + p_fluct * y_fluct
        return y_pred, p_stable, y_stable, y_fluct

# ===============================
# 7. 训练 + 评估封装
# ===============================
def run_experiment(name, use_mix, use_feature_flip):
    print(f"\n===== Running {name} =====")

    dataset = PowerDataset(power, L_IN, H)
    split = int(len(dataset) * 0.8)
    train_idx = range(0, split)
    test_idx = range(split, len(dataset))

    train_set = Subset(dataset, train_idx)
    test_set = Subset(dataset, test_idx)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    model = AdaptiveForecastModel(input_dim=3, hidden_size=64, H=H,
                                  use_mix=use_mix, use_feature_flip=use_feature_flip).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    bce_loss = nn.BCELoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        alpha = ALPHA_START - (ALPHA_START - ALPHA_END) * (epoch / max(1, EPOCHS - 1))

        for x, y, state in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            state = state.to(DEVICE).unsqueeze(1)

            optimizer.zero_grad()
            y_pred, p_stable, y_stable, y_fluct = model(x)

            mse_each = torch.mean((y_pred - y) ** 2, dim=1, keepdim=True)
            w_fluct = 1.0 + LAMBDA_FLUCT * state
            loss_fluct = (w_fluct * mse_each).mean()

            y_flat = y.view(y.size(0), -1)
            threshold = torch.quantile(y_flat, PEAK_Q, dim=1, keepdim=True)
            mask_peak = (y_flat > threshold).float()
            peak_err = ((y_flat - y_pred.view(y.size(0), -1)) ** 2) * mask_peak
            loss_peak = peak_err.mean()

            loss_pred = loss_fluct + LAMBDA_PEAK * loss_peak

            state_smooth = state * (1 - SMOOTH) + 0.5 * SMOOTH
            rule_target = 1.0 - state_smooth
            dyn_target = alpha * rule_target + (1 - alpha) * p_stable.detach()
            dyn_target = torch.clamp(dyn_target, 0.05, 0.95)
            loss_gate = bce_loss(p_stable, dyn_target)

            err_stable = torch.mean((y_stable - y) ** 2, dim=1, keepdim=True)
            err_fluct = torch.mean((y_fluct - y) ** 2, dim=1, keepdim=True)
            w_stable = torch.exp(-err_stable)
            w_fluct = torch.exp(-err_fluct)
            gate_target = (w_stable / (w_stable + w_fluct + 1e-8)).detach()
            loss_cons = bce_loss(p_stable, gate_target)

            loss = loss_pred + LAMBDA_GATE * loss_gate + LAMBDA_CONS * loss_cons

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}]  Loss: {avg_loss:.6f}  alpha={alpha:.3f}")

    print("Training finished.")

    model.eval()
    y_true_list = []
    y_pred_list = []
    state_labels = []

    with torch.no_grad():
        for x, y, state in test_loader:
            x = x.to(DEVICE)
            y_pred, _, _, _ = model(x)
            y_true_list.append(y.numpy().flatten()[0])
            y_pred_list.append(y_pred.cpu().numpy().flatten()[0])
            state_labels.append(int(state.item()))

    y_true = np.array(y_true_list) * std_p + mean_p
    y_pred = np.array(y_pred_list) * std_p + mean_p
    state_labels = np.array(state_labels)

    eps = 1e-6
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / (ss_tot + eps)

    mask_fluct = (state_labels == 1)
    mae_fluct = np.mean(np.abs(y_true[mask_fluct] - y_pred[mask_fluct]))

    q = 0.9
    thr = np.quantile(y_true, q)
    mask_peak = y_true >= thr
    mae_peak = np.mean(np.abs(y_true[mask_peak] - y_pred[mask_peak]))

    print("===== Quantitative Metrics =====")
    print(f"MAE   : {mae:.4f}")
    print(f"RMSE  : {rmse:.4f}")
    print(f"MAPE  : {mape:.2f}%")
    print(f"R2    : {r2:.4f}")
    print(f"MAE(Fluct) : {mae_fluct:.4f}")
    print(f"MAE(Peak>q{q}) : {mae_peak:.4f}")

# ===============================
# 8. 三组对比
# ===============================
# A：只保留时间反转（去掉特征反转）
run_experiment("A: time flip only (mix on)", use_mix=True, use_feature_flip=False)

# B：关闭 mix（回退到原输入）
run_experiment("B: no mix (feature flip on)", use_mix=False, use_feature_flip=True)

# C：当前版本（mix + 特征反转）
run_experiment("C: mix + feature flip", use_mix=True, use_feature_flip=True)
