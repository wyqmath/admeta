import pandas as pd
import numpy as np
import os
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
from scipy.stats import zscore
import patsy

# ===================================================================
# Part 0: 环境设置 (Environment Settings)
# ===================================================================

warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=FutureWarning) 
pd.options.mode.chained_assignment = None

# Nature 期刊风格绘图设置
plt.rcParams.update({
    'font.family': 'Arial',
    'axes.unicode_minus': False,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'axes.linewidth': 1,
    'xtick.major.width': 1,
    'ytick.major.width': 1
})

# 定义颜色方案
PALETTE = {
    "Non-carrier (0)": "#3B5488",  
    "Heterozygote (1)": "#E3A04D", 
    "Homozygote (2)": "#C03539"    
}

# 路径设置
script_dir = os.path.dirname(os.path.abspath(__file__))
data_file_path = os.path.join(script_dir, 'NACC_HV.csv') # 请确保文件名正确
output_directory = os.path.join(script_dir, "Results__Perfect")

# 如果文件夹不存在则创建
if not os.path.exists(output_directory): 
    os.makedirs(output_directory)

# ===================================================================
# Part 1: 数据预处理 (Data Preprocessing & QC)
# ===================================================================
print("--- Step 1: Loading and Cleaning Data ---")

try:
    long_data_raw = pd.read_csv(data_file_path)
except FileNotFoundError:
    print(f"Error: File not found at {data_file_path}"); exit()

# 1.1: 基础清洗 (移除缺失值)
df = long_data_raw.dropna(subset=['Hippocampus', 'Age', 'Sex', 'eTIV']).copy()

# 1.2: 时间计算 (Time Calculation)
df['Scan_Date'] = pd.to_datetime(df['Scan_Date'], format='%Y%m%d', errors='coerce')
df = df.dropna(subset=['Scan_Date']).sort_values(by=['NACCID', 'Scan_Date'])
# 计算每个人的基线日期
df['Baseline_Date'] = df.groupby('NACCID')['Scan_Date'].transform('min')
# 计算距离基线的时间（年）
df['Time'] = (df['Scan_Date'] - df['Baseline_Date']).dt.days / 365.25

# 1.3: 锁定基线诊断 (Fix Endogeneity)
# 提取基线时的诊断，并将其广播到该人的所有时间点，防止因随访期间诊断改变导致的内生性问题
baseline_info = df.loc[df['Time'] == 0, ['NACCID', 'diagnosis']].drop_duplicates(subset=['NACCID'])
baseline_info.rename(columns={'diagnosis': 'Baseline_Diagnosis'}, inplace=True)
df = df.merge(baseline_info, on='NACCID', how='left')

# 1.4: 变量转换 (APOE Categorization)
df['APOE4_Dosage'] = pd.Categorical(
    df['e4_count'].map({0: 'Non-carrier (0)', 1: 'Heterozygote (1)', 2: 'Homozygote (2)'}), 
    categories=['Non-carrier (0)', 'Heterozygote (1)', 'Homozygote (2)'], ordered=True
)

# -----------------------------------------------------------
# QC 1: Cross-sectional QC (截面质控)
# 修正点：加入 Baseline_Diagnosis，避免误删病情严重的 AD 患者
# -----------------------------------------------------------
print("Running Cross-sectional QC...")
qc_formula = "Hippocampus ~ Age + C(Sex) + eTIV + C(Baseline_Diagnosis)"
qc_model = smf.ols(qc_formula, data=df).fit()

# 计算残差 Z 分数
df['resid_z'] = zscore(qc_model.resid)
# 仅剔除残差绝对值大于 4 的极端异常值
df = df[df['resid_z'].abs() <= 4.0] 

# -----------------------------------------------------------
# QC 2: Longitudinal QC (纵向质控/斜率检查)
# 修正点：解决短时间间隔导致的速率计算错误 (Fix A)
# -----------------------------------------------------------
print("Running Longitudinal QC...")
df = df.sort_values(by=['NACCID', 'Time'])

# 计算体积变化百分比
df['vol_pct_change'] = df.groupby('NACCID')['Hippocampus'].pct_change()
# 计算时间间隔
df['time_diff'] = df.groupby('NACCID')['Time'].diff()

# 【关键修正】: 如果两次扫描间隔小于 6 个月 (0.5年)，将 time_diff 设为 NaN
# 这样计算出的 annual_pct_change 也会变成 NaN，从而不会被判定为异常速率而被剔除
# 这些数据点会被保留在模型中（因为 LMM 不需要计算斜率），但不会干扰 QC
mask_short_interval = df['time_diff'] < 0.5
df.loc[mask_short_interval, 'time_diff'] = np.nan 

# 计算年化变化率
df['annual_pct_change'] = df['vol_pct_change'] / df['time_diff']

# 设定生物学合理阈值：年萎缩 > 30% 或 年增加 > 20% 视为测量错误
mask_biological_impossible = (df['annual_pct_change'] > 0.20) | (df['annual_pct_change'] < -0.30)
n_dropped = mask_biological_impossible.sum()
df = df[~mask_biological_impossible].copy()
print(f"  -> Dropped {n_dropped} observations due to biologically impossible rates.")

# 1.5: 中心化与标准化 (Standardization)
df['Age_Centered'] = df['Age'] - df['Age'].mean()
df['eTIV_Scaled'] = df['eTIV'] / 1000.0

# 1.6: 最终筛选与导出
cols = ['NACCID', 'Time', 'Hippocampus', 'APOE4_Dosage', 'Age_Centered', 'Sex', 'eTIV_Scaled', 'Baseline_Diagnosis']
df_final = df[cols].dropna()

# 导出清洗后的数据
clean_data_path = os.path.join(output_directory, "NACC_Final_Processed_Data.csv")
df_final.to_csv(clean_data_path, index=False)
print(f"✅ Cleaned data saved to: {clean_data_path}")
print(f"   Total Subjects: {df_final['NACCID'].nunique()}, Total Scans: {len(df_final)}")

# 设定参考组
ref_diag = 'Normal' if 'Normal' in df_final['Baseline_Diagnosis'].unique() else df_final['Baseline_Diagnosis'].mode()[0]

# ===================================================================
# Part 2: 统计建模与表格导出 (Statistical Modeling)
# ===================================================================
print("\n--- Step 2: Fitting Linear Mixed Model ---")

formula = (
    f"Hippocampus ~ Time * C(APOE4_Dosage, Treatment('Non-carrier (0)')) "
    f"+ Time * Age_Centered "
    f"+ Time * C(Sex) "
    f"+ eTIV_Scaled "
    f"+ C(Baseline_Diagnosis, Treatment('{ref_diag}'))"
)

# 建立混合效应模型 (LMM)
model = smf.mixedlm(formula, df_final, groups=df_final['NACCID'], re_formula="1 + Time")
lmm_result = model.fit(method='lbfgs', maxiter=50000, full_output=True)

if lmm_result.converged:
    print("✅ Model Converged Successfully.")
else:
    print("⚠️ WARNING: Model did not strictly converge. Consider checking data.")

# 2.1: 保存详细文本报告
with open(os.path.join(output_directory, "Model_Summary.txt"), "w") as f:
    f.write(lmm_result.summary().as_text())

# 2.2: 导出统计表格 (Excel/CSV ready)
# 【关键修正】：P-value 保留原始浮点数，不转换为字符串
results_table = pd.DataFrame({
    'Coefficient': lmm_result.params,
    'Std.Err': lmm_result.bse,
    'z-value': lmm_result.tvalues,
    'P-value': lmm_result.pvalues, # 这里是具体的数值 (如 1.23e-10)
    'CI_Lower (2.5%)': lmm_result.conf_int()[0],
    'CI_Upper (97.5%)': lmm_result.conf_int()[1]
})

table_path = os.path.join(output_directory, "Model_Results_Table.csv")
results_table.to_csv(table_path)
print(f"✅ Statistical table saved to: {table_path} (Contains raw P-values)")

# ===================================================================
# Part 3: 绘图 (Visualization)
# ===================================================================
print("\n--- Step 3: Generating Figures ---")

# 提取模型参数供绘图使用
params = lmm_result.params
cov_matrix = lmm_result.cov_params()
colors = [PALETTE["Non-carrier (0)"], PALETTE["Heterozygote (1)"], PALETTE["Homozygote (2)"]]

# -----------------------------------------------------------
# Figure 1: Longitudinal Trajectories (含 Jitter 修正)
# -----------------------------------------------------------
print("Plotting Figure 1...")
fig1, ax1 = plt.subplots(figsize=(7, 6), constrained_layout=True)

# 随机抽样部分数据绘制背景散点，避免渲染过慢
sample_ids = np.random.choice(df_final['NACCID'].unique(), size=min(400, len(df_final['NACCID'].unique())), replace=False)
subset_data = df_final[df_final['NACCID'].isin(sample_ids)].copy()

# 【关键修正】：Jittering (抖动)
# 给绘图用的 Time 加一点随机噪声，让 X=0 处的点散开，不再是一堵墙
subset_data['Time_Jitter'] = subset_data['Time'] + np.random.uniform(-0.15, 0.15, size=len(subset_data))

# 绘制散点 (使用 Time_Jitter)
sns.scatterplot(
    data=subset_data, 
    x='Time_Jitter', y='Hippocampus', hue='APOE4_Dosage', 
    palette=PALETTE, alpha=0.2, legend=False, ax=ax1, 
    s=15, linewidth=0 
)

# 准备绘制拟合线 (使用原始 Time)
time_points = np.linspace(0, 8, 100)
mean_start_vol = df_final['Hippocampus'].mean() # 仅用于展示斜率的起始高度

# 提取斜率 (Slopes)
slope_0 = params['Time'] # Non-carrier
# Heterozygote Slope = Base Slope + Interaction Term
term_1_name = [x for x in params.index if 'Heterozygote' in x and 'Time' in x][0]
slope_1 = slope_0 + params[term_1_name]
# Homozygote Slope = Base Slope + Interaction Term
term_2_name = [x for x in params.index if 'Homozygote' in x and 'Time' in x][0]
slope_2 = slope_0 + params[term_2_name]

slopes = [slope_0, slope_1, slope_2]

# 循环绘制三条线
for i, (group, color, slope) in enumerate(zip(PALETTE.keys(), colors, slopes)):
    y_values = mean_start_vol + slope * time_points
    ax1.plot(time_points, y_values, color=color, linewidth=3, label=group)
    
    # 计算置信区间 (Confidence Intervals)
    if i == 0: 
        var_slope = cov_matrix.loc['Time', 'Time']
    elif i == 1: 
        # Var(A+B) = Var(A) + Var(B) + 2Cov(A,B)
        var_slope = (cov_matrix.loc['Time', 'Time'] + 
                     cov_matrix.loc[term_1_name, term_1_name] + 
                     2 * cov_matrix.loc['Time', term_1_name])
    else: 
        var_slope = (cov_matrix.loc['Time', 'Time'] + 
                     cov_matrix.loc[term_2_name, term_2_name] + 
                     2 * cov_matrix.loc['Time', term_2_name])
    
    se_slope = np.sqrt(var_slope)
    
    # 绘制阴影区域
    # 随着时间推移，误差累积，所以是 se * time_points
    ci_upper = y_values + 1.96 * se_slope * time_points
    ci_lower = y_values - 1.96 * se_slope * time_points
    ax1.fill_between(time_points, ci_lower, ci_upper, color=color, alpha=0.15)

ax1.set_xlabel("Time from Baseline (Years)", fontweight='bold')
ax1.set_ylabel("Hippocampal Volume ($mm^3$)", fontweight='bold')
ax1.set_title("Longitudinal Atrophy Trajectories", fontweight='bold', pad=15)
ax1.legend(title="APOE4 Genotype", loc='lower left')
ax1.grid(True, linestyle='--', alpha=0.3)
sns.despine()

fig1_path = os.path.join(output_directory, "Figure1_Trajectories.png")
plt.savefig(fig1_path, bbox_inches='tight')
plt.close()
print("✅ Figure 1 generated.")

# -----------------------------------------------------------
# Figure 2: Rate Comparison (含重叠修正与动态 Y 轴)
# -----------------------------------------------------------
print("Plotting Figure 2...")
fig2, ax2 = plt.subplots(figsize=(7, 6), constrained_layout=True)

# 准备绘图数据 (再次计算以确保清晰)
rates = [slope_0, slope_1, slope_2]

# 计算每个 Rate 的标准误 (SE) 用于误差棒
se_0 = np.sqrt(cov_matrix.loc['Time', 'Time'])
term_1_name = [x for x in params.index if 'Heterozygote' in x and 'Time' in x][0]
se_1 = np.sqrt(cov_matrix.loc['Time', 'Time'] + cov_matrix.loc[term_1_name, term_1_name] + 2 * cov_matrix.loc['Time', term_1_name])
term_2_name = [x for x in params.index if 'Homozygote' in x and 'Time' in x][0]
se_2 = np.sqrt(cov_matrix.loc['Time', 'Time'] + cov_matrix.loc[term_2_name, term_2_name] + 2 * cov_matrix.loc['Time', term_2_name])
errors = [se_0, se_1, se_2]

groups = ["Non-carrier\n(0)", "Heterozygote\n(1)", "Homozygote\n(2)"]

# 绘制柱状图
bars = ax2.bar(groups, rates, yerr=errors, color=colors, 
        capsize=6, width=0.6, edgecolor='black', linewidth=1.2, alpha=0.9, zorder=3,
        error_kw={'elinewidth': 1.5, 'markeredgewidth': 1.5})

ax2.axhline(0, color='black', linewidth=1.5, zorder=4)

# 【关键修正】：动态计算 Y 轴下限，防止文字跑出画布
lowest_point = min([r - e for r, e in zip(rates, errors)])
# 向下扩展 35% 的空间给文字
y_lower_limit = lowest_point * 1.35
ax2.set_ylim(y_lower_limit, 20) 

# 统计标注
p_val_homo = lmm_result.pvalues[term_2_name]
fold_change = slope_2 / slope_0

if p_val_homo < 0.05:
    # 画横线位置
    line_y = lowest_point - 5 
    ax2.plot([0, 0, 2, 2], [line_y, line_y - 5, line_y - 5, line_y], lw=1.2, c='k')
    
    # 这里的 P 值仅用于绘图展示，保留 < 0.001 格式比较美观
    # 具体的数值请看导出的 Excel 表格
    p_text = "P < 0.001" if p_val_homo < 0.001 else f"P = {p_val_homo:.3f}"
    ax2.text(1, line_y - 8, p_text, ha='center', va='top', color='black', fontweight='bold', fontsize=11)

# 倍数标注
text_y = (slope_2 - se_2) - 15 
ax2.text(2, text_y, f"{fold_change:.1f}x Faster", 
         ha='center', va='top', color=PALETTE["Homozygote (2)"], fontweight='bold', fontsize=14)

ax2.set_ylabel("Annual Change Rate ($mm^3$/year)", fontweight='bold', fontsize=12)
ax2.set_title("Accelerated Atrophy by Genotype", fontweight='bold', fontsize=14, pad=20)
ax2.grid(True, axis='y', linestyle='--', alpha=0.5, zorder=0)
sns.despine(ax=ax2, bottom=False, left=False)

fig2_path = os.path.join(output_directory, "Figure2_Rates_Final.png")
plt.savefig(fig2_path, bbox_inches='tight')
plt.close()
print("✅ Figure 2 generated.")

print("\nAll tasks completed successfully!")
print(f"Results are saved in: {output_directory}")