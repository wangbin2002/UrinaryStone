import pandas as pd
import matplotlib.pyplot as plt
import os

# ================== 参数设置 ==================
csv_file = "roc_data_20250916_170844.csv"  # CSV文件路径
save_dir = "./roc_curve_results"  # 保存文件夹
save_name = "roc_curve.pdf"               # 保存文件名
dpi_val = 1200                             # 清晰度

# ================== 创建保存文件夹 ==================
os.makedirs(save_dir, exist_ok=True)

# ================== 读取数据 ==================
df = pd.read_csv(csv_file)

# ================== 绘制ROC曲线 ==================
plt.figure(figsize=(8, 6))
classes = df['Class'].unique()

for cls in classes:
    cls_data = df[df['Class'] == cls]
    plt.plot(cls_data['FPR'], cls_data['TPR'], 
             label=f"{cls} (AUC={cls_data['AUC'].iloc[0]:.3f})")

# 对角线（随机分类器参考线）
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")

plt.xlabel("False Positive Rate", fontsize=16)
plt.ylabel("True Positive Rate", fontsize=16)
plt.title("ROC Curve", fontsize=18)
plt.legend(loc="lower right", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.6)

# ================== 保存PDF ==================
save_path = os.path.join(save_dir, save_name)
plt.savefig(save_path, format="pdf", dpi=dpi_val, bbox_inches="tight")
plt.close()

print(f"ROC曲线已保存到: {save_path}")
