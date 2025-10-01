import pandas as pd
import matplotlib.pyplot as plt
import os

# ================== 参数设置 ==================
csv_file = "pr_data_20250916_170844.csv"  # 你的CSV文件
save_dir = "./pr_curve_results"  # 保存文件夹
save_name = "pr_curve.pdf"               # 保存的PDF文件名
dpi_val = 1200                            # 输出清晰度

# ================== 创建保存文件夹 ==================
os.makedirs(save_dir, exist_ok=True)

# ================== 读取数据 ==================
df = pd.read_csv(csv_file)

# ================== 绘制PR曲线 ==================
plt.figure(figsize=(8, 6))
classes = df['Class'].unique()

for cls in classes:
    cls_data = df[df['Class'] == cls]
    plt.plot(cls_data['Recall'], cls_data['Precision'], label=f"{cls} (AP={cls_data['AP'].iloc[0]:.3f})")

plt.xlabel("Recall", fontsize=16)
plt.ylabel("Precision", fontsize=16)
plt.title("Precision-Recall Curve", fontsize=18)
plt.legend(loc="lower left", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)

# ================== 保存PDF ==================
save_path = os.path.join(save_dir, save_name)
plt.savefig(save_path, format="pdf", dpi=dpi_val, bbox_inches="tight")
plt.close()

print(f"PR曲线已保存到: {save_path}")
