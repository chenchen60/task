import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# 读取数据
df = pd.read_excel("D:\\lstm\\dianli.xlsx", parse_dates=['年份'])

# 设置时间序列索引
ts = df.set_index('年份')['城乡居民生活用电（万千瓦小时）']

# 绘制图形
fig, ax = plt.subplots(3, 1, figsize=(6, 8))

# 原始序列
ts.plot(ax=ax[0], title='原始序列')

# 一阶差分序列
ts.diff().plot(ax=ax[1], title='一阶差分序列')

# 自相关图
plot_acf(ts, lags=10, ax=ax[2])
ax[2].set_title('自相关图')

plt.tight_layout()
plt.show()
