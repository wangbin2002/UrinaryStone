import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================== 参数设置 ==================
csv_file = "confusion_matrix_count_20250916_170844.csv"  # CSV文件路径
save_dir = "./confusion_matrix_results"  # 保存文件夹
save_name = "confusion_matrix.pdf"               # 保存文件名
dpi_val = 1200                                    # 清晰度
font_size=16

# ================== 创建保存文件夹 ==================
os.makedirs(save_dir, exist_ok=True)

# ================== 读取数据 ==================
df = pd.read_csv(csv_file, index_col=0)

# ================== 绘制混淆矩阵 ==================
plt.figure(figsize=(8, 6))
sns.heatmap(df, annot=True, fmt="d", cmap="Blues", cbar=True, 
            annot_kws={"size": font_size},  # 设置数字字体大小
            xticklabels=df.columns, yticklabels=df.index)

plt.xlabel("Predicted Label", fontsize=16)
plt.ylabel("True Label", fontsize=16)
plt.title("Confusion Matrix", fontsize=18)

# ================== 保存PDF ==================
save_path = os.path.join(save_dir, save_name)
plt.savefig(save_path, format="pdf", dpi=dpi_val, bbox_inches="tight")
plt.close()

print(f"混淆矩阵已保存到: {save_path}")
