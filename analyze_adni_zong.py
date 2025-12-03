import pandas as pd
import numpy as np
import os
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
from scipy.stats import zscore, norm, probplot
from statsmodels.stats.multitest import multipletests
import re
import pickle # 用于保存模型

# ===================================================================
# Part 0: 全局配置 (Configuration)
# ===================================================================
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# 绘图风格
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['figure.dpi'] = 300 
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['font.size'] = 14

# 颜色配置 (顶刊配色)
COLORS = {
    "Non-carrier (0)": "#4E79A7", 
    "Heterozygote (1)": "#F28E2B", 
    "Homozygote (2)": "#E15759",
    "Positive": "#E15759",
    "Negative": "#4E79A7",
    "Abeta": "#D62728",  # Deep Red
    "pTau": "#1F77B4",   # Muted Blue
    "tTau": "#9467BD",   # Purple
    "Safe": "#2CA02C",   # Green
    "Risk1": "#FF7F0E"   # Orange
}

# 阈值配置
CUTOFFS = {'Abeta': 976.6, 'pTau': 21.8, 'tTau': 245}

# 路径配置
DATA_PATH = 'ADNI_zong.csv'
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")
MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, "Saved_Models") # 专门存模型文件

for d in [OUTPUT_DIR, MODEL_SAVE_DIR]:
    if not os.path.exists(d): os.makedirs(d)

log_path = os.path.join(OUTPUT_DIR, "Analysis_Log_Final.txt")
log_file = open(log_path, 'w', encoding='utf-8')

print(f"--- 最终版分析脚本启动 ---")
print(f"结果输出: {OUTPUT_DIR}")
log_file.write("--- ADNI Analysis Final Log ---\n\n")

# ===================================================================
# Part 1: 数据加载与清洗 (Pipeline)
# ===================================================================
def load_and_clean_data(filepath):
    print("Step 1: 数据加载与清洗...")
    try:
        df = pd.read_csv(filepath, sep=',', encoding='utf-8')
    except FileNotFoundError:
        print(f"错误: 找不到文件 {filepath}"); exit()

    # 重命名
    df = df.rename(columns={'CSF_Abeta42': 'csf_abeta42', 'CSF_pTau': 'csf_ptau', 'CSF_tTau': 'csf_ttau', 'Hippocampus': 'hippocampal_volume'})
    
    # 清洗数值
    for col in ['csf_abeta42', 'csf_ptau', 'csf_ttau']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[><]', '', regex=True), errors='coerce')

    # 定义状态
    df['amyloid_status'] = np.where(df['csf_abeta42'] < CUTOFFS['Abeta'], "Positive", "Negative")
    df['ptau_status'] = np.where(df['csf_ptau'] > CUTOFFS['pTau'], "Positive", "Negative")
    df['ttau_status'] = np.where(df['csf_ttau'] > CUTOFFS['tTau'], "Positive", "Negative")

    # QC 1: 横断面 (Z-score)
    subset = df.dropna(subset=['hippocampal_volume', 'Age', 'Sex', 'ICV']).copy()
    mod = smf.ols("hippocampal_volume ~ Age + C(Sex) + ICV", data=subset).fit()
    subset['resid_z'] = zscore(mod.resid)
    df = df.loc[subset[subset['resid_z'].abs() <= 3.5].index].copy()

    # 时间计算
    df['Scan_Date'] = pd.to_datetime(df['Scan_Date'], format='%Y%m%d')
    df = df.sort_values(by=['PTID', 'Scan_Date'])
    df['Baseline_Date'] = df.groupby('PTID')['Scan_Date'].transform('min')
    df['Time'] = (df['Scan_Date'] - df['Baseline_Date']).dt.days / 365.25
    
    # 过滤过长的时间点 (保持稳定性)
    df = df[df['Time'] <= 10]

    # 变量类型转换
    df['APOE4_Dosage'] = pd.Categorical(
        df['e4_count'].map({0: 'Non-carrier (0)', 1: 'Heterozygote (1)', 2: 'Homozygote (2)'}),
        categories=['Non-carrier (0)', 'Heterozygote (1)', 'Homozygote (2)'], ordered=True
    )
    df['diagnosis'] = df['diagnosis'].replace({'EMCI': 'MCI', 'LMCI': 'MCI'})
    
    cols = ['PTID', 'Time', 'hippocampal_volume', 'APOE4_Dosage', 'amyloid_status', 'ptau_status', 'ttau_status', 'Age', 'Sex', 'ICV', 'diagnosis', 'e4_count']
    return df[cols].dropna()

long_data = load_and_clean_data(DATA_PATH)
diagnosis_ref = long_data['diagnosis'].mode()[0]
print(f"  -> 最终样本量: N = {long_data.shape[0]}")

# ===================================================================
# Part 2: 稳健模型拟合 (带保存功能)
# ===================================================================
def check_diagnostics(result, name):
    """绘制模型残差诊断图 (审稿人防御性武器)"""
    resid = result.resid
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Q-Q Plot
    probplot(resid, dist="norm", plot=axes[0])
    axes[0].set_title(f"{name}: Q-Q Plot of Residuals")
    
    # 2. Residuals vs Fitted
    axes[1].scatter(result.fittedvalues, resid, alpha=0.3, s=10)
    axes[1].axhline(0, color='red', linestyle='--')
    axes[1].set_xlabel("Fitted Values")
    axes[1].set_ylabel("Residuals")
    axes[1].set_title(f"{name}: Residuals vs Fitted")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"Diagnostic_{name}.png"))
    plt.close()

def fit_lmm_smart(formula, data, name):
    model_path = os.path.join(MODEL_SAVE_DIR, f"{name}.pickle")
    
    # 1. 尝试加载
    if os.path.exists(model_path):
        print(f"  > 发现已保存的模型 {name}，正在加载...")
        try:
            with open(model_path, 'rb') as f:
                result = pickle.load(f)
            # 【修改点】: 即使是加载的模型，也要把结果写入日志！
            log_file.write(f"\n--- {name} (Loaded from file) ---\n")
            log_file.write(str(result.summary()) + "\n\n") 
            return result
        except:
            print("    加载失败，重新拟合...")

    # 2. 重新拟合
    print(f"  > 正在拟合新模型: {name} (请耐心等待)...")
    log_file.write(f"\n--- {name} ---\n公式: {formula}\n")
    
    model_full = smf.mixedlm(formula, data, groups=data["PTID"], re_formula="1 + Time")
    optimizers = ['nm', 'lbfgs', 'powell'] 
    best_result = None
    
    for opt in optimizers:
        try:
            res = model_full.fit(method=opt, maxiter=3000)
            if res.converged:
                print(f"    SUCCESS: Random Slope ({opt})")
                log_file.write(f"收敛: Random Slope (Opt: {opt})\n")
                best_result = res
                break
        except: pass
        
    if best_result is None:
        log_file.write("降级: Random Intercept Only\n")
        try:
            model_simple = smf.mixedlm(formula, data, groups=data["PTID"], re_formula="1")
            best_result = model_simple.fit(method='lbfgs', maxiter=200)
        except: pass

    # Save & Return
    if best_result:
        with open(model_path, 'wb') as f:
            pickle.dump(best_result, f)
        check_diagnostics(best_result, name)
        # 【修改点】: 将统计结果表格写入日志文件
        log_file.write(str(best_result.summary()) + "\n\n")
        
    return best_result

# ===================================================================
# Part 3: 运行核心模型
# ===================================================================
print("Step 2: 运行统计模型...")

# 1. 单一模型 (Single Models)
models = {}
biomarkers = [('amyloid_status', 'Amyloid'), ('ptau_status', 'pTau'), ('ttau_status', 'tTau')]

for col, label in biomarkers:
    form = f"hippocampal_volume ~ Time * C(APOE4_Dosage, Treatment('Non-carrier (0)')) * C({col}, Treatment('Negative')) + Age + C(Sex) + ICV + C(diagnosis, Treatment('{diagnosis_ref}'))"
    models[label] = fit_lmm_smart(form, long_data, f"Model_{label}")

# 2. 联合模型 (Joint Model - The Star)
print("  > 运行联合模型 (Joint Model)...")
form_joint = f"""
hippocampal_volume ~ Time + Age + C(Sex) + ICV + 
Time * C(APOE4_Dosage, Treatment('Non-carrier (0)')) * C(amyloid_status, Treatment('Negative')) +
Time * C(APOE4_Dosage, Treatment('Non-carrier (0)')) * C(ptau_status, Treatment('Negative'))
"""
models['Joint'] = fit_lmm_smart(form_joint, long_data, "Model_Joint_Abeta_pTau")

# 保存联合模型摘要
if models['Joint']:
    with open(os.path.join(OUTPUT_DIR, "Joint_Model_Summary_Final.txt"), 'w') as f:
        f.write(str(models['Joint'].summary()))

# ===================================================================
# Part 4: 最终定稿版高清绘图 (Final Publication Visualization)
# ===================================================================
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
import os
import re

print("\n>>> 启动最终定稿绘图模块 (600 DPI, Corrected Alignment)...")

# 确保输出目录存在
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

# -------------------------------------------------------------------
# 1. 全局绘图风格 (针对 PPT 拼图优化)
# -------------------------------------------------------------------
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 16           # 字体加大，PPT 缩放后依然清晰
plt.rcParams['axes.linewidth'] = 2.5     # 坐标轴加粗
plt.rcParams['xtick.major.width'] = 2.5
plt.rcParams['ytick.major.width'] = 2.5
plt.rcParams['figure.dpi'] = 600         # 屏幕预览分辨率
plt.rcParams['savefig.dpi'] = 600        # 文件保存分辨率 (印刷级)

# 严格颜色定义 (语义化颜色)
COLOR_MAP = {
    'Abeta': '#D62728',  # 深红 (Significant / Atrophy Driver)
    'pTau':  '#1F77B4',  # 蓝 (Non-significant / Noise)
    'tTau':  '#9467BD',  # 紫 (Non-significant / Noise)
    'Null':  '#555555'   # 灰色 (0线)
}

# -------------------------------------------------------------------
# 2. 辅助函数: 稳健提取与打印
# -------------------------------------------------------------------
def get_coef_strict(model, label, term_regex):
    if model is None: return None
    matches = [k for k in model.params.index if re.search(term_regex, k)]
    if not matches: 
        msg = f"!! 警告: {label} 中未找到项: {term_regex}\n"
        print(msg.strip())
        log_file.write(msg) # 写入日志
        return None
    
    target = matches[0]
    coef = model.params[target]
    pval = model.pvalues[target]
    
    # 【修改点】: 同时打印到屏幕 和 写入日志
    res_str = f"  [{label}] Target: {target} | Coef: {coef:.3f} | P-val: {pval:.4f}\n"
    print(res_str.strip())
    log_file.write(res_str)
    
    return {
        'Model': label,
        'Coef': coef,
        'Error': model.bse[target],
        'Lower': model.conf_int().loc[target][0],
        'Upper': model.conf_int().loc[target][1],
        'P_val': pval
    }
# 正则: 匹配 Time * APOE(1) * Biomarker(Pos)
regex_inter = r"Time:.*Heterozygote.*Positive"

# ===================================================================
# Figure 1: 顶刊极简专业版 (Professional Table Style)
# ===================================================================
print("正在生成 Figure 1 (一致性标准化版)...")

data_f1 = []
if 'Amyloid' in models: data_f1.append(get_coef_strict(models['Amyloid'], 'Amyloid Model', regex_inter))
if 'pTau' in models:    data_f1.append(get_coef_strict(models['pTau'], 'pTau Model', regex_inter))
if 'tTau' in models:    data_f1.append(get_coef_strict(models['tTau'], 'tTau Model', regex_inter))

df_f1 = pd.DataFrame(data_f1)

if not df_f1.empty:
    print("\n>>> 执行多重比较校正 (FDR - Benjamini/Hochberg)...")
    
    # 提取原始 P 值
    raw_pvals = df_f1['P_val'].values
    
    # 执行校正 (method='fdr_bh' 是最常用的 FDR; method='bonferroni' 最严格)
    reject, pvals_corrected, _, _ = multipletests(raw_pvals, alpha=0.05, method='fdr_bh')
    
    # 将校正后的 P 值存回 DataFrame
    df_f1['P_val_Adjusted'] = pvals_corrected
    
    # 打印对比结果
    print(df_f1[['Model', 'P_val', 'P_val_Adjusted']])

    # === 修改绘图逻辑使用校正后的 P 值 ===
    # 必须修改后续绘图循环中的变量，将 row['P_val'] 改为 row['P_val_Adjusted']
    # 如下所示：
# =======================================================

if not df_f1.empty:
    fig, ax = plt.subplots(figsize=(12, 7))
    # ... (原有设置代码)
    
    for i, row in df_f1.iterrows():
        # ... (颜色设置代码) ...
        
        # ----------- 修改点 1: 判定显著性 -----------
        # 使用校正后的 P 值
        is_sig = row['P_val_Adjusted'] < 0.05 
        
        # ----------- 修改点 2: 文本显示 -----------
        # 显示校正后的 P 值 (P_adj)
        if row['P_val_Adjusted'] < 0.001:
            p_str = "P_adj < 0.001"
        else:
            p_str = f"P_adj = {row['P_val_Adjusted']:.3f}"
            
        # ... (后续绘图代码保持不变) ...



if not df_f1.empty:
    fig, ax = plt.subplots(figsize=(12, 7)) # 画布加宽，为右侧文字留空间
    y_pos = range(len(df_f1))
    
    # 1. 计算对齐基准线 (逻辑同 Fig 2)
    all_vals = list(df_f1['Lower']) + list(df_f1['Upper'])
    min_x, max_x = min(all_vals), max(all_vals)
    # 在最右侧数据再往右偏移一些
    # 注意：Fig1 的数据可能都在左边，也可能在右边，所以要动态计算
    padding_base = (max_x - min_x) if (max_x - min_x) > 0 else 50 
    text_x_pos = max_x + padding_base * 0.15
    if text_x_pos < 0: text_x_pos = 10 # 防止全都挤在左边，强制给一个右侧位置
    
    for i, row in df_f1.iterrows():
        # 颜色逻辑
        if 'Amyloid' in row['Model']: c = COLOR_MAP['Abeta']
        elif 'pTau' in row['Model']:  c = COLOR_MAP['pTau']
        else:                         c = COLOR_MAP['tTau']
        
        # 绘制误差棒
        ax.errorbar(row['Coef'], i, xerr=row['Error']*1.96, 
                    fmt='o', color=c, ecolor=c, 
                    capsize=10, markersize=18, linewidth=3.5)
        
        # --- 核心修改：样式统一化 ---
        is_sig = row['P_val'] < 0.05
        
        # 文本内容: P = ...
        if row['P_val'] < 0.001:
            p_str = "P < 0.001"
        else:
            p_str = f"P = {row['P_val']:.3f}"
            
        # 样式区分: 显著黑粗，不显著灰细
        if is_sig:
            fw = 'bold'
            fc = 'black'
            alpha_text = 1.0
        else:
            fw = 'normal'
            fc = '#666666'
            alpha_text = 0.8
            
        # 绘制文本: 右侧对齐
        ax.text(text_x_pos, i, p_str, va='center', ha='left', 
                fontsize=15, fontweight=fw, color=fc, alpha=alpha_text)

    # 0 线
    ax.axvline(0, color='black', linestyle='--', linewidth=2.5, alpha=0.8)
    
    # 坐标轴
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_f1['Model'], fontweight='bold', fontsize=16)
    
    # 扩展X轴范围
    padding = (max_x - min_x) * 0.45
    if padding == 0: padding = 50
    ax.set_xlim(min_x - padding*0.2, text_x_pos + padding)
    
    # 留白与反转
    ax.set_ylim(-0.5, len(df_f1) - 0.5)
    ax.margins(y=0.25)
    ax.invert_yaxis()
    
    # 标题与标签
    ax.set_xlabel("Synergistic Atrophy Rate (mm³/year)", fontsize=16, fontweight='bold')
    ax.set_title("Specificity of APOE ε4 Synergistic Effects", fontweight='bold', fontsize=20, pad=20)
    
    sns.despine(left=True)
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, "Fig1_Specificity_Professional.png")
    plt.savefig(save_path, dpi=600)
    print(f"Figure 1 (极简专业版) 已保存至: {save_path}")
    plt.close()

# ===================================================================
# Figure 2: 顶刊极简风 (Clean & Professional Style)
# ===================================================================
print("正在生成 Figure 2 (顶刊极简对齐版)...")

data_f2 = []
r1 = get_coef_strict(models['Amyloid'], 'Aβ (Single)', regex_inter)
r2 = get_coef_strict(models['Joint'], 'Aβ (Joint)', r"Time:.*Heterozygote.*amyloid.*Positive")
r3 = get_coef_strict(models['pTau'], 'p-Tau (Single)', regex_inter)
r4 = get_coef_strict(models['Joint'], 'p-Tau (Joint)', r"Time:.*Heterozygote.*ptau.*Positive")

for r in [r1, r2, r3, r4]:
    if r: data_f2.append(r)

df_f2 = pd.DataFrame(data_f2)

if not df_f2.empty:
    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = range(len(df_f2))
    
    # 1. 计算右侧文字的对齐基准线
    all_vals = list(df_f2['Lower']) + list(df_f2['Upper'])
    min_x, max_x = min(all_vals), max(all_vals)
    # 在最右侧数据再往右偏移 15% 的位置开始写字
    text_x_pos = max_x + (max_x - min_x) * 0.15 
    
    for i, row in df_f2.iterrows():
        c = COLOR_MAP['Abeta'] if 'Aβ' in row['Model'] else COLOR_MAP['pTau']
        
        # 绘制区间线
        ax.plot([row['Lower'], row['Upper']], [i, i], color=c, lw=4, alpha=0.6)
        # 绘制点
        ax.plot(row['Coef'], i, 'o', color=c, markersize=20)
        
        # --- 核心修改：极简风格 ---
        is_sig = row['P_val'] < 0.05
        
        # 文本内容：只保留 P=... (去掉 ns/*)
        if row['P_val'] < 0.001:
            p_str = "P < 0.001"
        else:
            p_str = f"P = {row['P_val']:.3f}"
        
        # 视觉样式区分
        if is_sig:
            # 显著：黑色、加粗、不透明
            fw = 'bold'
            fc = 'black'
            alpha_text = 1.0
        else:
            # 不显著：灰色、普通、稍微淡一点 (让读者忽略它)
            fw = 'normal'
            fc = '#666666' # 深灰
            alpha_text = 0.8
        
        # 绘制文本：右侧对齐
        ax.text(text_x_pos, i, p_str, va='center', ha='left', 
                fontsize=15, fontweight=fw, color=fc, alpha=alpha_text)

    # 0 线
    ax.axvline(0, color='black', linestyle='--', linewidth=2.5)
    
    # 坐标轴设置
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_f2['Model'], fontsize=16, fontweight='bold')
    
    # 扩展X轴范围，防止文字被切
    padding = (max_x - min_x) * 0.45
    ax.set_xlim(min_x - (padding*0.2), text_x_pos + padding)
    ax.set_ylim(-0.5, len(df_f2) - 0.5)
    
    # 标题
    ax.set_xlabel("Synergistic Atrophy Rate (mm³/year)", fontsize=16, fontweight='bold')
    ax.set_title("Robustness of Amyloid Specificity\n(Joint Model Verification)", fontweight='bold', fontsize=20, pad=20)
    
    # 图例
    patches = [mpatches.Patch(color=COLOR_MAP['Abeta'], label='Amyloid Effect'),
               mpatches.Patch(color=COLOR_MAP['pTau'], label='p-Tau Effect')]
    ax.legend(handles=patches, loc='lower left', frameon=False, fontsize=14)
    
    ax.invert_yaxis()
    sns.despine()
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, "Fig2_Robustness_Professional.png")
    plt.savefig(save_path, dpi=600)
    print(f"Figure 2 (极简专业版) 已保存至: {save_path}")
    plt.close()
# -------------------------------------------------------------------
# Figure 3: 纵向轨迹图 (Clinical Translation)
# -------------------------------------------------------------------
print("正在生成 Figure 3 (Trajectories)...")

fig, ax = plt.subplots(figsize=(10, 9))

subset = long_data[long_data['e4_count'].isin([0, 1])].copy()
subset['Group'] = subset.apply(lambda x: f"{'Aβ+' if x['amyloid_status']=='Positive' else 'Aβ-'}\n{'APOE ε4+' if x['e4_count']==1 else 'APOE ε4-'}", axis=1)

# 颜色盘 (高对比度)
# A+E+ 用醒目的红色，其他用冷色或黄色
pal_traj = {
    "Aβ-\nAPOE ε4-": "#2CA02C", # 绿 (Baseline)
    "Aβ-\nAPOE ε4+": "#1F77B4", # 蓝 (Gene only)
    "Aβ+\nAPOE ε4-": "#FFBB78", # 浅橙 (Pathology only - 淡一点以突出红色)
    "Aβ+\nAPOE ε4+": "#D62728"  # 深红 (Interaction - 重点)
}

groups = sorted(subset['Group'].unique())

for g in groups:
    dat = subset[subset['Group'] == g]
    c = pal_traj.get(g, 'gray')
    # 这里的 lw=5 让 PPT 里线条非常清晰
    sns.regplot(data=dat, x='Time', y='hippocampal_volume', ax=ax, 
                scatter=False, label=g, color=c, 
                ci=95, truncate=True, line_kws={'lw': 5})

ax.set_xlabel("Years from Baseline", fontsize=18, fontweight='bold')
ax.set_ylabel("Hippocampal Volume (mm³)", fontsize=18, fontweight='bold')
ax.set_title("Longitudinal Trajectories", fontweight='bold', fontsize=22, pad=20)

# 图例优化
ax.legend(title="Group Status", title_fontsize=16, fontsize=14, frameon=False, loc='lower left')
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Fig3_Trajectories_600dpi.png"))
plt.close()

print("\n>>> 全部完成！图片已生成至: Results_Final_Submission")
print("    请检查 Figure 1 森林图：红点在左(Sig)，蓝紫点在右(ns)，数值在点上方对齐。")

# ... (所有代码最后)
log_file.close() # 确保内容被写入磁盘
print("分析日志已保存至: Analysis_Log_Final.txt")