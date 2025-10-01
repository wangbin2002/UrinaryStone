import pandas as pd
import matplotlib.pyplot as plt
import os

# ================== 参数设置 ==================
csv_file = "cic_data_20250916_170844.csv"  # CSV文件路径
save_dir = "./cic_curve_results"  # 保存文件夹
save_name = "cic_curve.pdf"               # 保存文件名
dpi_val = 1200                             # 清晰度
total_patients = 208                      # 总患者数 (需与你的数据一致)

# ================== 创建保存文件夹 ==================
os.makedirs(save_dir, exist_ok=True)

# ================== 读取数据 ==================
df = pd.read_csv(csv_file)
classes = df['Class'].unique()

# ================== 绘制CIC曲线 ==================
plt.figure(figsize=(8, 6))

# 与DCA一致的颜色
colors = {
    "UA": "aqua",
    "NUA": "darkorange",
    "MIX": "cornflowerblue"
}

for cls in classes:
    cls_data = df[df['Class'] == cls]
    
    # 转换为百分比
    high_risk_pct = cls_data['High_Risk_Count'] / total_patients * 100
    tp_pct = cls_data['True_Positive_Count'] / total_patients * 100
    
    # 高风险患者曲线 (实线)
    plt.plot(cls_data['Threshold'], high_risk_pct, lw=2, 
             label=f"High risk count - {cls}", color=colors.get(cls, None))
    
    # 真阳性曲线 (虚线)
    plt.plot(cls_data['Threshold'], tp_pct, lw=2, linestyle="--", 
             label=f"True positive count - {cls}", color=colors.get(cls, None))

# 图形修饰
plt.xlim([0.0, 1.0])
plt.ylim([0, 100])
plt.xlabel("Threshold Probability", fontsize=16)
plt.ylabel("Percentage of Patients (%)", fontsize=16)
plt.title("Clinical Impact Curve", fontsize=18)
plt.legend(loc="best", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)

# ================== 保存PDF ==================
save_path = os.path.join(save_dir, save_name)
plt.savefig(save_path, dpi=dpi_val, format="pdf", bbox_inches="tight")
plt.close()

print(f"CIC曲线已保存到: {save_path}")
