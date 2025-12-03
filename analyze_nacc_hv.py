import pandas as pd
import numpy as np
import os
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
from scipy.stats import zscore

# ===================================================================
# Part 0: 环境设置
# ===================================================================
warnings.filterwarnings('ignore', category=ConvergenceWarning)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False 
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600

# Nature 风格配色
PALETTE = {
    "Non-carrier (0)": "#3B5488",  # Navy Blue
    "Heterozygote (1)": "#E3A04D", # Muted Gold
    "Homozygote (2)": "#C03539"    # Deep Red
}

script_dir = os.path.dirname(os.path.abspath(__file__))
data_file_path = os.path.join(script_dir, 'NACC_HV.csv')
# 输出文件夹改名，方便区分
output_directory = os.path.join(script_dir, "Results_NACC_HV_Clean_Look")
if not os.path.exists(output_directory): os.makedirs(output_directory)

log_path = os.path.join(output_directory, "Analysis_Log.txt")
log_file = open(log_path, 'w', encoding='utf-8')
print(f"脚本开始运行... 结果将保存至: {output_directory}")

# 读取数据
try:
    long_data_raw = pd.read_csv(data_file_path, sep=',', encoding='utf-8')
except FileNotFoundError:
    print("Error: File not found."); exit()

# ===================================================================
# Part 1: 数据预处理与 QC (逻辑不变)
# ===================================================================
long_data_processed_temp = long_data_raw.copy().dropna(subset=['Hippocampus', 'Age', 'Sex', 'eTIV'])

# QC Stage 1
qc_model = smf.ols("Hippocampus ~ Age + C(Sex) + eTIV", data=long_data_processed_temp).fit()
long_data_processed_temp['residuals_zscore'] = zscore(qc_model.resid)
long_data_processed_temp = long_data_processed_temp[long_data_processed_temp['residuals_zscore'].abs() <= 3.5]
long_data_processed_temp = long_data_processed_temp.drop(columns=['residuals_zscore'])

# QC Stage 2
long_data_processed_temp['Scan_Date'] = pd.to_datetime(long_data_processed_temp['Scan_Date'], format='%Y%m%d')
long_data_processed_temp = long_data_processed_temp.sort_values(by=['NACCID', 'Scan_Date'])
long_data_processed_temp['time_diff'] = long_data_processed_temp.groupby('NACCID')['Scan_Date'].diff().dt.days / 365.25
long_data_processed_temp['volume_diff'] = long_data_processed_temp.groupby('NACCID')['Hippocampus'].diff()
long_data_processed_temp['annual_change'] = np.where(long_data_processed_temp['time_diff'] > 0.25, 
                                                 long_data_processed_temp['volume_diff'] / long_data_processed_temp['time_diff'], np.nan)
Q1, Q3 = long_data_processed_temp['annual_change'].quantile([0.25, 0.75])
IQR = Q3 - Q1
mask = (long_data_processed_temp['annual_change'] < (Q1 - 3*IQR)) | (long_data_processed_temp['annual_change'] > (Q3 + 3*IQR))
long_data_processed_temp = long_data_processed_temp[~mask].drop(columns=['time_diff', 'volume_diff', 'annual_change'])

# Final Setup
long_data_processed_temp['Baseline_Date'] = long_data_processed_temp.groupby('NACCID')['Scan_Date'].transform('min')
long_data_processed_temp['Time'] = (long_data_processed_temp['Scan_Date'] - long_data_processed_temp['Baseline_Date']).dt.days / 365.25
long_data_processed_temp['APOE4_Dosage'] = pd.Categorical(long_data_processed_temp['e4_count'].map({0: 'Non-carrier (0)', 1: 'Heterozygote (1)', 2: 'Homozygote (2)'}), categories=['Non-carrier (0)', 'Heterozygote (1)', 'Homozygote (2)'], ordered=True)
cols = ['NACCID', 'Time', 'Hippocampus', 'APOE4_Dosage', 'Age', 'Sex', 'eTIV', 'diagnosis']
long_data_processed = long_data_processed_temp[cols].dropna()
diagnosis_ref = long_data_processed['diagnosis'].mode()[0]
long_data_processed['diagnosis'] = pd.Categorical(long_data_processed['diagnosis'])
print(f"数据处理完成，N: {long_data_processed.shape[0]}")

# ===================================================================
# Part 2: 统计建模 (Strict LMM)
# ===================================================================
print("正在拟合模型...")
formula_nacc = f"Hippocampus ~ Time * C(APOE4_Dosage, Treatment('Non-carrier (0)')) + Age + C(Sex) + eTIV + C(diagnosis, Treatment('{diagnosis_ref}'))"
model = smf.mixedlm(formula_nacc, long_data_processed, groups=long_data_processed['NACCID'], re_formula="1 + Time")
lmm_result = model.fit(method='lbfgs', maxiter=2000)

if lmm_result.converged:
    print("模型收敛成功。")
else:
    print("警告：模型未收敛。"); exit()

# ===================================================================
# Part 3: 最终极简版可视化 (Clean & Impactful)
# ===================================================================
def produce_clean_figures(model_result, data, output_dir):
    print("\n--- 正在绘制极简版图表 ---")
    params = model_result.params
    cov = model_result.cov_params()
    
    # -----------------------------------------------------------
    # Figure 1: Trajectories (保持之前的完美状态)
    # -----------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(8, 6.5))
    sns.scatterplot(data=data, x="Time", y="Hippocampus", hue="APOE4_Dosage", 
                    palette=PALETTE, alpha=0.12, s=12, linewidth=0, ax=ax1, legend=False)
    
    time_seq = np.linspace(0, 10, 100)
    age_mean = data['Age'].mean()
    etiv_mean = data['eTIV'].mean()
    
    for i, group in enumerate(["Non-carrier (0)", "Heterozygote (1)", "Homozygote (2)"]):
        pred_df = pd.DataFrame({'Time': time_seq, 'APOE4_Dosage': group, 'Age': age_mean, 'eTIV': etiv_mean,
                                'Sex': data['Sex'].mode()[0], 'diagnosis': data['diagnosis'].mode()[0]})
        y_pred = model_result.predict(exog=pred_df)
        ax1.plot(time_seq, y_pred, color=PALETTE[group], linewidth=3, label=group)
        
        term_time = 'Time'
        if i == 0: slope_se = np.sqrt(cov.loc[term_time, term_time])
        elif i == 1: 
            term_inter = [x for x in params.index if 'Heterozygote' in x and 'Time' in x][0]
            slope_se = np.sqrt(cov.loc[term_time, term_time] + cov.loc[term_inter, term_inter] + 2*cov.loc[term_time, term_inter])
        else:
            term_inter = [x for x in params.index if 'Homozygote' in x and 'Time' in x][0]
            slope_se = np.sqrt(cov.loc[term_time, term_time] + cov.loc[term_inter, term_inter] + 2*cov.loc[term_time, term_inter])
            
        ci_upper = y_pred + (slope_se * 1.96 * time_seq)
        ci_lower = y_pred - (slope_se * 1.96 * time_seq)
        ax1.fill_between(time_seq, ci_lower, ci_upper, color=PALETTE[group], alpha=0.15, linewidth=0)

    ax1.set_title("Longitudinal Hippocampal Atrophy Trajectories", fontweight='bold', fontsize=16, pad=20, loc='left')
    ax1.set_xlabel("Time from Baseline (Years)", fontweight='bold', fontsize=12)
    ax1.set_ylabel("Hippocampal Volume ($mm^3$)", fontweight='bold', fontsize=12)
    ax1.set_xlim(0, 10)
    y_min, y_max = data['Hippocampus'].quantile(0.005), data['Hippocampus'].quantile(0.995)
    ax1.set_ylim(y_min, y_max)
    ax1.legend(title="APOE-$\epsilon$4 Dosage", title_fontsize=10, fontsize=9, loc='upper right', frameon=True, facecolor='white', framealpha=0.9, edgecolor='none')
    ax1.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5, color='#dddddd')
    sns.despine(trim=True, offset=10)
    plt.savefig(os.path.join(output_dir, "Figure1_Trajectories.png"), bbox_inches='tight')
    plt.close()

    # -----------------------------------------------------------
    # Figure 2: Rates (无数字版 + 倍数标注)
    # -----------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    
    # Calculate Data
    term_time = 'Time'
    inter_het = [x for x in params.index if 'Heterozygote' in x and 'Time' in x][0]
    inter_hom = [x for x in params.index if 'Homozygote' in x and 'Time' in x][0]
    
    rate_0 = params[term_time]
    se_0 = np.sqrt(cov.loc[term_time, term_time])
    rate_1 = rate_0 + params[inter_het]
    se_1 = np.sqrt(cov.loc[term_time, term_time] + cov.loc[inter_het, inter_het] + 2*cov.loc[term_time, inter_het])
    rate_2 = rate_0 + params[inter_hom]
    se_2 = np.sqrt(cov.loc[term_time, term_time] + cov.loc[inter_hom, inter_hom] + 2*cov.loc[term_time, inter_hom])
    
    # Calculate Fold Change (倍数)
    fold_change = rate_2 / rate_0 # e.g., -116 / -44 = 2.6
    
    p_val_hom = model_result.pvalues[inter_hom]
    significance_text = "P < 0.05 *" if p_val_hom < 0.05 else "n.s."
    if p_val_hom < 0.001: significance_text = "P < 0.001 ***"
    
    groups = ["Non-carrier\n(0)", "Heterozygote\n(1)", "Homozygote\n(2)"]
    rates = [rate_0, rate_1, rate_2]
    errors = [se_0, se_1, se_2]
    bar_colors = [PALETTE["Non-carrier (0)"], PALETTE["Heterozygote (1)"], PALETTE["Homozygote (2)"]]
    
    # Draw Bars (无数字)
    bars = ax2.bar(groups, rates, yerr=errors, color=bar_colors, 
                   capsize=5, width=0.55, edgecolor='black', linewidth=1, alpha=0.9, zorder=3)
    
    # --- 1. 顶部显著性标记 ---
    if p_val_hom < 0.05:
        x1, x2 = 0, 2
        y_line = 10 
        h = 5 
        ax2.plot([x1, x1, x2, x2], [y_line-h, y_line, y_line, y_line-h], lw=1.2, c='k')
        ax2.text((x1+x2)*.5, y_line + 2, significance_text, ha='center', va='bottom', 
                 color='black', fontweight='bold', fontsize=11)

    # --- 2. 核心倍数标注 (Visualizing the ~2.6x difference) ---
    # 我们在红色柱子旁边画一个注释，强调它是蓝色的2.6倍
    # 位置：在红色柱子右侧，或者直接在柱子上
    # 这里我们采用在红色柱子下方添加文本的方式，干净有力
    
    ax2.text(2, rate_2 - se_2 - 15, f"{fold_change:.1f}x Faster", 
             ha='center', va='top', color=PALETTE["Homozygote (2)"], 
             fontweight='bold', fontsize=12)

    # --- 3. 移除具体的数值标注 (Less is More) ---
    # (此前的 ax2.text 循环已被删除)

    ax2.set_title("Accelerated Atrophy Rates by Genotype", fontweight='bold', fontsize=16, pad=20)
    ax2.set_ylabel("Annual Change Rate ($mm^3$/year)", fontweight='bold', fontsize=12)
    ax2.set_xlabel("APOE-$\epsilon$4 Dosage", fontweight='bold', fontsize=12)
    
    ax2.axhline(0, color='black', linewidth=1)
    
    # Y轴范围调整
    y_bottom = min(rates) - max(errors) - 40 # 留出更多空间给下方的 "2.6x Faster"
    ax2.set_ylim(y_bottom, 30) 
    
    # 确保网格线清晰，便于读者目测
    ax2.grid(True, axis='y', linestyle='--', alpha=0.6, zorder=0)
    sns.despine()
    
    plt.savefig(os.path.join(output_dir, "Figure2_Rates_Clean.png"), bbox_inches='tight')
    plt.close()
    print("极简版图表生成完成！")

# 运行
produce_clean_figures(lmm_result, long_data_processed, output_directory)
log_file.close()
print(f"\n--- 运行结束！请查看: {output_directory} ---")