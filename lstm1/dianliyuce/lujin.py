import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# 假设数据已经加载到DataFrame中，并包含列'年份'和'用电总量（万千瓦小时）'
data = pd.read_excel("dianli.xlsx")  # 请替换为您的数据文件路径

# 1. 数据预处理
data.fillna(0, inplace=True)  # 填充缺失值为0
print(data.head())  # 显示数据的前几行，检查数据格式

# 2. 基线方法：简单移动平均（5年）
window_size = 5  # 选择一个窗口大小（比如5年）
data['移动平均（5年）'] = data['用电总量（万千瓦小时）'].rolling(window=window_size).mean()

# 3. 基线方法：简单指数平滑
model = SimpleExpSmoothing(data['用电总量（万千瓦小时）'])
model_fit = model.fit(smoothing_level=0.2, optimized=False)  # 选择一个合适的平滑因子
data['指数平滑预测'] = model_fit.fittedvalues

# 4. 绘制实际用电量、移动平均和指数平滑结果
plt.figure(figsize=(10, 6))

# 绘制实际用电量
plt.plot(data['年份'], data['用电总量（万千瓦小时）'], label='实际用电量', color='blue')

# 绘制5年移动平均
plt.plot(data['年份'], data['移动平均（5年）'], label=f'移动平均（{window_size}年）', color='red')

# 绘制指数平滑预测
plt.plot(data['年份'], data['指数平滑预测'], label='简单指数平滑', color='green')

# 设置图表标题与标签
plt.title('用电量与移动平均和简单指数平滑对比')
plt.xlabel('年份')
plt.ylabel('用电量（万千瓦小时）')

# 显示图例
plt.legend()
plt.show()

# 5. 评估：计算均方误差（MSE）和均绝对误差（MAE）
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 假设我们用最后3年的实际数据作为测试集
y_true = data['用电总量（万千瓦小时）'][-3:]  # 实际值（最后3年）
y_pred_ma = data['移动平均（5年）'][-3:]  # 移动平均预测值
y_pred_es = data['指数平滑预测'][-3:]  # 指数平滑预测值

# 计算误差
mse_ma = mean_squared_error(y_true, y_pred_ma)  # 移动平均的MSE
mae_ma = mean_absolute_error(y_true, y_pred_ma)  # 移动平均的MAE

mse_es = mean_squared_error(y_true, y_pred_es)  # 指数平滑的MSE
mae_es = mean_absolute_error(y_true, y_pred_es)  # 指数平滑的MAE

print(f"移动平均法 MSE: {mse_ma}, MAE: {mae_ma}")
print(f"指数平滑法 MSE: {mse_es}, MAE: {mae_es}")
