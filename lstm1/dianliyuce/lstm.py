# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib import rcParams


rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

df = pd.read_excel("dianli.xlsx", index_col=0)
df.index = pd.to_datetime(df.index.astype(str) + "-12-31")

# 缺失值处理
print("缺失值情况:\n", df.isna().sum())
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)

# 异常值检测与修复函数
def detect_outliers_zscore(series: pd.Series, z_thresh: float = 3.0) -> pd.Series:
    """返回布尔 Series，True 表示该点为 z-score 异常"""
    mean = series.mean()
    std = series.std()
    if std == 0 or np.isnan(std):
        return pd.Series(False, index=series.index)
    z_scores = (series - mean) / std
    return z_scores.abs() > z_thresh

def detect_outliers_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
    """返回布尔 Series，True 表示该点为 IQR 异常"""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return pd.Series(False, index=series.index)
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return (series < lower) | (series > upper)

def detect_and_fix_outliers(series: pd.Series,
                            method: str = 'both',   # 'z', 'iqr', or 'both'
                            z_thresh: float = 3.0,
                            iqr_k: float = 1.5,
                            plot: bool = True,
                            name: str = None) -> pd.Series:
    """
    检测并修复异常值：
      - method: 'z' / 'iqr' / 'both'
      - 检测后将异常点设为 NaN，再用线性插值修复，最后 ffill / bfill 以防首尾 NaN
    返回修复后的 Series
    """
    s = series.astype(float).copy()
    if method == 'z':
        mask_z = detect_outliers_zscore(s, z_thresh)
        mask = mask_z
    elif method == 'iqr':
        mask_iqr = detect_outliers_iqr(s, iqr_k)
        mask = mask_iqr
    else:
        mask_z = detect_outliers_zscore(s, z_thresh)
        mask_iqr = detect_outliers_iqr(s, iqr_k)
        mask = mask_z | mask_iqr

    n_outliers = int(mask.sum())
    print(f"[异常检测] '{name or series.name}': 检出异常点 {n_outliers} 个。")

    # 若无异常，直接返回原序列（但保持 float 类型）
    if n_outliers == 0:
        return s

    # 可视化异常点（原始序列 + 标红异常）
    if plot:
        plt.figure(figsize=(10,4))
        plt.plot(s.index.year, s.values, marker='o', label=f"{name or series.name} 原始值")
        plt.scatter(s.index.year[mask], s.values[mask], color='red', label='检测为异常的点', zorder=5)
        plt.title(f"{name or series.name} - 异常值检测 (method={method})")
        plt.xlabel("年份")
        plt.ylabel(series.name)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

    # 将异常点设为 NaN，然后插值修复
    s.loc[mask] = np.nan
    # 使用线性插值（datetime 索引也适用）；若仍有 NaN 则 ffill, bfill
    s = s.interpolate(method='linear', limit_direction='both')
    s = s.fillna(method='ffill').fillna(method='bfill')
    # 如果插值后仍然存在 NaN（极端小数据），用原始值回退（保险）
    s = s.fillna(series.astype(float))

    return s

# 对关键序列进行异常检测与修复
cols_to_check = [
    "用电总量（万千瓦小时）",
    "城乡居民生活用电（万千瓦小时）"
]

for col in cols_to_check:
    if col in df.columns:
        fixed = detect_and_fix_outliers(df[col], method='both', z_thresh=3.0, iqr_k=1.5, plot=True, name=col)
        df[col] = fixed
    else:
        print(f"警告：DataFrame 中没有列 '{col}'，跳过异常检测。")


# 回测指标函数（保持不变）
def calculate_metrics(y_true, y_pred):
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true==0,1,y_true))) * 100
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE(%)": mape}

# LSTM 模型定义（保持不变）
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


# LSTM 训练函数（保持不变）

def train_lstm(series, seq_len=5, epochs=500, lr=0.01, pred_steps=3, mc_samples=100):
    scaler = MinMaxScaler()
    series_scaled = scaler.fit_transform(series.values.reshape(-1,1))
    
    # 构造训练数据
    X, y = [], []
    for i in range(len(series_scaled)-seq_len):
        X.append(series_scaled[i:i+seq_len])
        y.append(series_scaled[i+seq_len])
    X = torch.tensor(np.array(X)).float()
    y = torch.tensor(np.array(y)).float()
    
    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 训练
    for epoch in range(1, epochs+1):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.6f}")
    
    # 历史回测预测
    with torch.no_grad():
        preds_scaled = model(X).numpy()
        preds = scaler.inverse_transform(preds_scaled)
    
    metrics = calculate_metrics(series[seq_len:], preds.flatten())
    
    # 未来预测及置信区间
    last_seq = series_scaled[-seq_len:].reshape(1, seq_len, 1)
    last_seq_tensor = torch.tensor(last_seq).float()
    
    mc_preds = []
    for _ in range(mc_samples):
        seq_mc = last_seq_tensor.clone()
        seq_mc += torch.normal(0, 0.01, size=seq_mc.shape)
        preds_mc = []
        for _ in range(pred_steps):
            with torch.no_grad():
                next_val = model(seq_mc).item()
            preds_mc.append(next_val)
            seq_mc = torch.cat([seq_mc[:,1:,:], torch.tensor([[[next_val]]], dtype=torch.float32)], dim=1)
        mc_preds.append(preds_mc)
    
    mc_preds = scaler.inverse_transform(np.array(mc_preds).reshape(-1,pred_steps))
    mean_preds = mc_preds.mean(axis=0)
    std_preds = mc_preds.std(axis=0)
    lower_preds = mean_preds - 1.96*std_preds
    upper_preds = mean_preds + 1.96*std_preds
    
    return model, scaler, preds.flatten(), metrics, mean_preds, lower_preds, upper_preds

# 基线预测函数（移动平均 + 未来预测）
def baseline_forecast(series, window=3, pred_steps=3):
    rolling_pred = series.rolling(window).mean().shift(1).fillna(method='bfill')
    
    # 未来预测
    future_pred = []
    temp_series = series.copy()
    last_index = temp_series.index[-1]
    for _ in range(pred_steps):
        val = temp_series[-window:].mean()
        future_pred.append(val)
        last_index = last_index + pd.offsets.YearEnd(1)
        temp_series = pd.concat([temp_series, pd.Series([val], index=[last_index])])
    
    return rolling_pred, np.array(future_pred)


# 设置预测年份
future_years = np.array([df.index[-1].year + i for i in range(1,4)])
years_hist = df.index.year


# 总用电量预测
print("\n训练总用电量 LSTM...")
model_total, scaler_total, lstm_total_hist, metrics_total, preds_total, lower_total, upper_total = train_lstm(df["用电总量（万千瓦小时）"])
baseline_total_hist, baseline_total_future = baseline_forecast(df["用电总量（万千瓦小时）"] if "用电量（万千瓦小时）" in df.columns else df["用电总量（万千瓦小时）"])

metrics_baseline_total = calculate_metrics(df["用电总量（万千瓦小时）"].iloc[5:], baseline_total_hist.iloc[5:])
print("回测指标 (总用电量 LSTM):", metrics_total)
print("回测指标 (总用电量 基线):", metrics_baseline_total)

# 居民用电量预测
print("\n训练居民用电量 LSTM...")
model_res, scaler_res, lstm_res_hist, metrics_res, preds_res, lower_res, upper_res = train_lstm(df["城乡居民生活用电（万千瓦小时）"])
baseline_res_hist, baseline_res_future = baseline_forecast(df["城乡居民生活用电（万千瓦小时）"])

metrics_baseline_res = calculate_metrics(df["城乡居民生活用电（万千瓦小时）"].iloc[5:], baseline_res_hist.iloc[5:])
print("回测指标 (居民用电量 LSTM):", metrics_res)
print("回测指标 (居民用电量 基线):", metrics_baseline_res)

# 绘图：总用电量（历史 + LSTM拟合 + 未来预测 + 基线）
plt.figure(figsize=(12,6))
plt.plot(years_hist, df["用电总量（万千瓦小时）"], label="总用电量-历史", marker='o')
plt.plot(years_hist[5:], lstm_total_hist, label="总用电量-LSTM拟合", linestyle='-', color='blue')
plt.plot(future_years, preds_total, label="总用电量-未来预测(LSTM)", marker='x', linestyle='--', color='blue')
plt.fill_between(future_years, lower_total, upper_total, color='blue', alpha=0.2, label="总用电量-95%置信区间")
plt.plot(years_hist, baseline_total_hist, label="总用电量-基线(MA)", linestyle='--', color='green')
plt.plot(future_years, baseline_total_future, linestyle='--', color='green', marker='^', label="总用电量-基线预测")
plt.xlabel("年份")
plt.ylabel("用电量 (万千瓦小时)")
plt.title("总用电量预测与基线对比（含LSTM历史拟合）")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12,6))
plt.plot(years_hist, df["城乡居民生活用电（万千瓦小时）"], label="居民用电量-历史", marker='o')
plt.plot(years_hist[5:], lstm_res_hist, label="居民用电量-LSTM拟合", linestyle='-', color='orange')
plt.plot(future_years, preds_res, label="居民用电量-未来预测(LSTM)", marker='x', linestyle='--', color='orange')
plt.fill_between(future_years, lower_res, upper_res, color='orange', alpha=0.3, label="居民用电量-95%置信区间")
plt.plot(years_hist, baseline_res_hist, label="居民用电量-基线(MA)", linestyle='--', color='purple')
plt.plot(future_years, baseline_res_future, linestyle='--', color='purple', marker='^', label="居民用电量-基线预测")
plt.xlabel("年份")
plt.ylabel("用电量 (万千瓦小时)")
plt.title("居民用电量预测与基线对比（含LSTM历史拟合）")
plt.legend()
plt.grid(True)
plt.show()
