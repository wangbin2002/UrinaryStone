import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ================== 参数设置 ==================
csv_file = "dca_data_20250916_170844.csv"  # CSV文件路径
save_dir = "./dca_curve_results"  # 保存文件夹
save_name = "dca_curve.pdf"               # 保存文件名
dpi_val = 1200                             # 清晰度
positive_class = "UA"                     # 定义哪个类别算阳性

# ================== 创建保存文件夹 ==================
os.makedirs(save_dir, exist_ok=True)

# ================== 读取数据 ==================
df = pd.read_csv(csv_file)
classes = df['Class'].unique()

# ================== 绘制DCA曲线 ==================
plt.figure(figsize=(8, 6))

# 自定义颜色，保持和ROC曲线一致 (依次给UA, NUA, MIX)
colors = {
    "UA": "aqua",
    "NUA": "darkorange",
    "MIX": "cornflowerblue"
}

for cls in classes:
    cls_data = df[df['Class'] == cls]
    plt.plot(cls_data['Threshold'], cls_data['Net_Benefit'],
             lw=2, label=f"DCA curve of class {cls}", color=colors.get(cls, None))

# 添加 Treat None 曲线
thresholds = np.linspace(0, 1, 200)
plt.plot(thresholds, np.zeros(len(thresholds)), 'k--', lw=2, label='Treat None')

# 自动计算 Prevalence（阳性比例）
prevalence = (df['Class'] == positive_class).sum() / len(df)

# Treat All 曲线
treat_all = prevalence - (1 - prevalence) * thresholds / (1 - thresholds)
plt.plot(thresholds, treat_all, 'k:', lw=2, label='Treat All')

# 图形修饰
plt.xlim([0.0, 1.0])
plt.ylim([min(df['Net_Benefit']) - 0.05, max(df['Net_Benefit']) + 0.05])
plt.xlabel('Threshold Probability', fontsize=16)
plt.ylabel('Net Benefit', fontsize=16)
plt.title('Decision Curve Analysis', fontsize=18)
plt.legend(loc="upper right", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)

# ================== 保存PDF ==================
save_path = os.path.join(save_dir, save_name)
plt.savefig(save_path, dpi=dpi_val, format="pdf", bbox_inches="tight")
plt.close()

print(f"DCA曲线已保存到: {save_path}")
