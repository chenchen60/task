# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

df = pd.read_excel("dianli.xlsx")

plt.figure(figsize=(8, 5))
plt.plot(df["年份"], df['城乡居民生活用电（万千瓦小时）'], marker="o", linestyle="-", linewidth=2)#可切换用电总量

plt.title("城乡居民生活用电变化趋势", fontsize=14)
plt.xlabel("年份", fontsize=12)
plt.ylabel("城乡居民生活用电（万千瓦小时）", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)


plt.tight_layout()
plt.show()
