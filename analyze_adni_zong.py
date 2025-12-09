import pandas as pd
import numpy as np
import os
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import warnings

# ===================================================================
# Part 0: 全局配置
# ===================================================================
warnings.filterwarnings('ignore')
OUTPUT_DIR = "Results_Final_QC_Version"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

log_path = os.path.join(OUTPUT_DIR, "Analysis_Log_QC.txt")
log_file = open(log_path, 'w', encoding='utf-8')

def log(msg):
    print(msg)
    log_file.write(msg + "\n")

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 300
sns.set_theme(style="ticks", font_scale=1.1)

COLORS = {
    "Abeta": "#D62728", 
    "pTau":  "#1F77B4", 
    "tTau":  "#9467BD", 
    "Safe":  "#2CA02C", 
    "Risk":  "#D62728",
}

CUTOFFS = {'Abeta': 976.6, 'pTau': 21.8, 'tTau': 245}

# ===================================================================
# Part 1: 数据加载
# ===================================================================
def load_and_prep_data(filepath):
    log("Step 1: Loading Data...")
    try:
        df = pd.read_csv(filepath)
    except:
        log("Error: ADNI_zong.csv not found."); exit()
    
    rename_map = {'CSF_Abeta42': 'csf_abeta', 'CSF_pTau': 'csf_ptau', 'CSF_tTau': 'csf_ttau', 'Hippocampus': 'hippo_vol'}
    df = df.rename(columns=rename_map)
    for col in ['csf_abeta', 'csf_ptau', 'csf_ttau']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[><]', '', regex=True), errors='coerce')

    df['amyloid_status'] = np.where(df['csf_abeta'] < CUTOFFS['Abeta'], "Positive", "Negative")
    df['ptau_status'] = np.where(df['csf_ptau'] > CUTOFFS['pTau'], "Positive", "Negative")
    df['ttau_status'] = np.where(df['csf_ttau'] > CUTOFFS['tTau'], "Positive", "Negative")

    df['Scan_Date'] = pd.to_datetime(df['Scan_Date'], format='%Y%m%d', errors='coerce')
    df = df.sort_values(by=['PTID', 'Scan_Date'])
    df['Baseline_Date'] = df.groupby('PTID')['Scan_Date'].transform('min')
    df['Time'] = (df['Scan_Date'] - df['Baseline_Date']).dt.days / 365.25
    df['Age_bl'] = df.groupby('PTID')['Age'].transform('first')
    df['APOE4_Group'] = df['e4_count'].map({0: 'Non-carrier', 1: 'Heterozygote', 2: 'Homozygote'})
    
    # 临时 Z-score 用于后续 QC
    for col in ['Age_bl', 'ICV', 'hippo_vol']:
        df[f'{col}_z'] = (df[col] - df[col].mean()) / df[col].std()
    
    cols = ['PTID', 'Time', 'hippo_vol', 'hippo_vol_z', 'APOE4_Group', 'e4_count',
            'amyloid_status', 'ptau_status', 'ttau_status', 
            'csf_abeta', 'csf_ptau', 'Age_bl_z', 'Sex', 'ICV_z']
    return df[cols] # 这里先不 dropna，留给 QC 处理

# ===================================================================
# Part 1.5: 质量控制 (QC) - 新增模块
# ===================================================================
def perform_qc(df):
    log("\nStep 1.5: Performing Quality Control (QC)...")
    n_init = len(df)
    n_sub_init = df['PTID'].nunique()
    log(f"  -> Initial: {n_init} observations from {n_sub_init} subjects.")

    # 1. 删除关键变量缺失的行
    df = df.dropna(subset=['hippo_vol', 'Age_bl_z', 'ICV_z', 'csf_abeta', 'csf_ptau'])
    log(f"  -> After Drop Missing: {len(df)} observations.")

    # 2. 纵向数据要求：至少有 2 个时间点 (才能算斜率)
    counts = df['PTID'].value_counts()
    valid_ptids = counts[counts >= 2].index
    df = df[df['PTID'].isin(valid_ptids)]
    log(f"  -> After Longitudinal Filter (>=2 visits): {len(df)} observations from {df['PTID'].nunique()} subjects.")

    # 3. 异常值剔除：Z-score > 4 (极值剔除，避免分割错误干扰)
    # 我们认为 Z > 4 或 Z < -4 是测量误差，而非生物学变异
    df = df[df['hippo_vol_z'].abs() <= 4]
    log(f"  -> After Outlier Removal (|Z|>4): {len(df)} observations.")
    
    # 重新计算 Z 分数 （样本变了)
    for col in ['hippo_vol']:
        df[f'{col}_z'] = (df[col] - df[col].mean()) / df[col].std()
        
    return df

# ===================================================================
# Part 2: 统计模型
# ===================================================================
def fit_best_lmm(data, formula):
    model = smf.mixedlm(formula, data, groups=data["PTID"], re_formula="1 + Time")
    for opt in ['nm', 'lbfgs', 'powell']:
        try:
            res = model.fit(method=opt, maxiter=5000, remix=False)
            if res.converged: return res, "Random Slope"
        except: pass
    try:
        model = smf.mixedlm(formula, data, groups=data["PTID"], re_formula="1")
        res = model.fit(method='lbfgs')
        return res, "Random Intercept"
    except: return None, "Failed"

# ===================================================================
# Part 3: 中介分析 (精确 P 值版)
# ===================================================================
def run_mediation_final(df):
    log("\nStep 4: Running Mediation Analysis (Exact P-values)...")
    carriers = df[df['e4_count'] == 1].copy()
    slopes = []
    
    for ptid, group in carriers.groupby('PTID'):
        if len(group) < 2: continue
        try:
            r = smf.ols("hippo_vol_z ~ Time", data=group).fit()
            base = group.iloc[0]
            slopes.append({
                'PTID': ptid, 'Atrophy_Rate': r.params['Time'],
                'Abeta_Pos': 1 if base['amyloid_status'] == 'Positive' else 0,
                'pTau_Level': base['csf_ptau'],
                'Age_bl_z': base['Age_bl_z'], 'Sex': base['Sex'], 'ICV_z': base['ICV_z']
            })
        except: pass
    
    slope_df = pd.DataFrame(slopes)
    if slope_df.empty: return None
    
    slope_df['pTau_z'] = (slope_df['pTau_Level'] - slope_df['pTau_Level'].mean()) / slope_df['pTau_Level'].std()
    
    # Models
    m_a = smf.ols("pTau_z ~ Abeta_Pos + Age_bl_z + C(Sex)", data=slope_df).fit()
    m_b = smf.ols("Atrophy_Rate ~ Abeta_Pos + pTau_z + Age_bl_z + C(Sex) + ICV_z", data=slope_df).fit()
    
    # Coefs
    a = m_a.params['Abeta_Pos']
    b = m_b.params['pTau_z']
    c_prime = m_b.params['Abeta_Pos']
    indirect_beta = a * b
    
    # P-values
    p_a = m_a.pvalues['Abeta_Pos']
    p_b = m_b.pvalues['pTau_z']
    p_c = m_b.pvalues['Abeta_Pos']
    
    # Sobel
    sa, sb = m_a.bse['Abeta_Pos'], m_b.bse['pTau_z']
    z_score = indirect_beta / np.sqrt(b**2 * sa**2 + a**2 * sb**2)
    p_indirect = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    # === Drawing Figure 3 (Exact P-values) ===
    log("  -> Drawing Figure 3...")
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.2, 1.0); ax.axis('off')
    
    pos = {'X': (0, 0), 'Y': (1, 0), 'M': (0.5, 0.75)}
    
    def draw_box(x, y, text, color):
        ax.text(x, y, text, ha='center', va='center', fontsize=13, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=color, lw=2), zorder=10)
    
    def draw_arrow(start, end, color):
        ax.annotate("", xy=end, xytext=start, 
                    arrowprops=dict(arrowstyle="->", color=color, lw=2, 
                                    shrinkA=25, shrinkB=25, mutation_scale=20), zorder=1)

    draw_box(*pos['X'], "Amyloid\n(Pathology)", COLORS['Abeta'])
    draw_box(*pos['Y'], "Atrophy\n(Outcome)", "black")
    draw_box(*pos['M'], "p-Tau\n(Mediator)", COLORS['pTau'])
    
    draw_arrow(pos['X'], pos['M'], COLORS['Abeta'])
    draw_arrow(pos['M'], pos['Y'], COLORS['pTau'])
    draw_arrow(pos['X'], pos['Y'], "gray")
    
    # 格式化 P 值的辅助函数：如果极小，显示科学计数法；否则显示3位小数
    def fmt_stats(beta, p):
        if p < 0.001:
            # 科学计数法，例如 2.3e-05
            p_str = f"{p:.1e}".replace("e-0", "e-")
        else:
            p_str = f"{p:.3f}"
        return f"$\\beta$={beta:.2f}\n$P$={p_str}"
    
    # Path Labels
    ax.text(0.18, 0.45, f"Path a (A$\\rightarrow$pT)\n{fmt_stats(a, p_a)}", 
            color=COLORS['Abeta'], ha='center', fontweight='bold', fontsize=11, bbox=dict(fc='white', ec='none', alpha=0.9))
    ax.text(0.82, 0.45, f"Path b (pT$\\rightarrow$Y)\n{fmt_stats(b, p_b)}", 
            color=COLORS['pTau'], ha='center', fontweight='bold', fontsize=11, bbox=dict(fc='white', ec='none', alpha=0.9))
    ax.text(0.5, -0.12, f"Direct Effect (c')\n{fmt_stats(c_prime, p_c)}", 
            ha='center', color='gray', fontsize=11, bbox=dict(fc='white', ec='none', alpha=0.9))
    
    # Title
    if p_indirect < 0.001:
        p_val_title = f"{p_indirect:.1e}".replace("e-0", "e-")
    else:
        p_val_title = f"{p_indirect:.3f}"
        
    ax.set_title(f"Mediation Analysis (Carriers Only)\nIndirect Effect: $\\beta$={indirect_beta:.3f} ($P$={p_val_title})", 
                 fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig3_Mediation.png"))

# ===================================================================
# Part 4: 主流程
# ===================================================================
# 1. 加载
df_raw = load_and_prep_data('ADNI_zong.csv')
# 2. 质量控制 (QC)
df = perform_qc(df_raw)

covars = "Age_bl_z + C(Sex) + ICV_z"
models_config = [
    {'id': 'S_Abeta', 'form': f"hippo_vol_z ~ Time * C(APOE4_Group, Treatment('Non-carrier')) * C(amyloid_status, Treatment('Negative')) + {covars}", 'kw': 'amyloid', 'name': 'Single Factor Model', 'marker': 'Abeta'},
    {'id': 'S_pTau',  'form': f"hippo_vol_z ~ Time * C(APOE4_Group, Treatment('Non-carrier')) * C(ptau_status, Treatment('Negative')) + {covars}", 'kw': 'ptau',    'name': 'Single Factor Model', 'marker': 'pTau'},
    {'id': 'S_tTau',  'form': f"hippo_vol_z ~ Time * C(APOE4_Group, Treatment('Non-carrier')) * C(ttau_status, Treatment('Negative')) + {covars}", 'kw': 'ttau',    'name': 'Single Factor Model', 'marker': 'tTau'},
    {'id': 'J_pTau',  'form': f"hippo_vol_z ~ Time + {covars} + Time * C(APOE4_Group, Treatment('Non-carrier')) * C(amyloid_status, Treatment('Negative')) + Time * C(APOE4_Group, Treatment('Non-carrier')) * C(ptau_status, Treatment('Negative'))", 
     'extract': [('amyloid', 'Abeta', 'Joint Model\n(adj. for p-Tau)'), ('ptau', 'pTau', 'Joint Model\n(adj. for Amyloid)')]},
    {'id': 'J_tTau',  'form': f"hippo_vol_z ~ Time + {covars} + Time * C(APOE4_Group, Treatment('Non-carrier')) * C(amyloid_status, Treatment('Negative')) + Time * C(APOE4_Group, Treatment('Non-carrier')) * C(ttau_status, Treatment('Negative'))", 
     'extract': [('amyloid', 'Abeta', 'Joint Model\n(adj. for t-Tau)'), ('ttau', 'tTau', 'Joint Model\n(adj. for Amyloid)')]}
]

results = []
log("\nStep 2: Fitting Models...")
for cfg in models_config:
    res, tag = fit_best_lmm(df, cfg['form'])
    if not res: continue
    params = res.params.index
    if 'extract' in cfg:
        for kw, marker, disp_name in cfg['extract']:
            term = [p for p in params if 'Time' in p and 'Heterozygote' in p and 'Positive' in p and kw in p][0]
            results.append({'Model': disp_name, 'Marker': marker, 'Coef': res.params[term], 'SE': res.bse[term], 'P_Raw': res.pvalues[term]})
    else:
        term = [p for p in params if 'Time' in p and 'Heterozygote' in p and 'Positive' in p and cfg['kw'] in p][0]
        results.append({'Model': cfg['name'], 'Marker': cfg['marker'], 'Coef': res.params[term], 'SE': res.bse[term], 'P_Raw': res.pvalues[term]})

# 导出表
res_df = pd.DataFrame(results)
_, res_df['P_FDR'], _, _ = multipletests(res_df['P_Raw'], method='fdr_bh')
res_df.to_csv(os.path.join(OUTPUT_DIR, "Table_Statistics.csv"), index=False)

# 运行中介
run_mediation_final(df)

# === Figure 1: Forest (Stars Only) ===
log("Drawing Figure 1...")
fig, ax = plt.subplots(figsize=(10, 7))
res_df['Sort1'] = res_df['Marker'].apply(lambda x: 0 if x == 'Abeta' else (1 if x == 'pTau' else 2))
res_df['Sort2'] = res_df['Model'].apply(lambda x: 0 if 'Single' in x else 1)
plot_df = res_df.sort_values(by=['Sort1', 'Sort2'], ascending=[False, False]).reset_index(drop=True)

y_pos = range(len(plot_df))
for i, row in plot_df.iterrows():
    c = COLORS[row['Marker']]
    ax.errorbar(row['Coef'], i, xerr=row['SE']*1.96, fmt='o', color=c, ecolor=c, markersize=14, capsize=4, elinewidth=2.5)
    
    # 星号系统
    if row['P_Raw'] < 0.001: sig = "***"
    elif row['P_Raw'] < 0.01: sig = "**"
    elif row['P_Raw'] < 0.05: sig = "*"
    else: sig = ""
    if sig: ax.text(row['Coef'], i-0.25, sig, ha='center', va='center', fontsize=18, fontweight='bold', color='black')

ax.set_yticks(y_pos); ax.set_yticklabels(plot_df['Model'], fontweight='bold', fontsize=12)
ax.axvline(0, color='black', linestyle='--', alpha=0.4)
ax.set_xlabel("Synergistic Atrophy Rate (Standardized $\\beta$)", fontweight='bold', fontsize=14)
ax.set_title("Specificity of APOE $\\epsilon$4 Synergistic Effects", fontweight='bold', fontsize=15, pad=20)
ax.set_xlim(-0.16, 0.16)
patches = [mpatches.Patch(color=COLORS['Abeta'], label='Amyloid Effect'),
           mpatches.Patch(color=COLORS['pTau'], label='p-Tau Effect'),
           mpatches.Patch(color=COLORS['tTau'], label='t-Tau Effect')]
ax.legend(handles=patches, loc='lower left', frameon=True, fontsize=10)
ax.text(0.98, 0.02, "* P<0.05, ** P<0.01, *** P<0.001", transform=ax.transAxes, ha='right', fontsize=10, style='italic', color='gray')
sns.despine(); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Fig1_Forest_Clean.png"))

# === Figure 2: Trajectories (含 QC 后的数据) ===
log("Drawing Figure 2...")
subset = df[df['e4_count'].isin([0, 1])].copy()
subset['Group'] = subset.apply(lambda x: 
    "A+E+" if (x['amyloid_status']=='Positive' and x['e4_count']==1) else
    "A-E-" if (x['amyloid_status']=='Negative' and x['e4_count']==0) else "Other", axis=1)
subset = subset[subset['Group'] != "Other"]

fig, ax = plt.subplots(figsize=(9, 7))
sns.regplot(data=subset[subset['Group']=="A-E-"], x='Time', y='hippo_vol', ax=ax, scatter=False, 
            color=COLORS['Safe'], label="Aβ- / ε4- (Control)", line_kws={'lw': 4})
sns.regplot(data=subset[subset['Group']=="A+E+"], x='Time', y='hippo_vol', ax=ax, scatter=False, 
            color=COLORS['Risk'], label="Aβ+ / ε4+ (Risk)", line_kws={'lw': 4})
ax.set_xlabel("Years from Baseline", fontweight='bold', fontsize=14)
ax.set_ylabel("Hippocampal Volume (mm³)", fontweight='bold', fontsize=14)
ax.set_title("Longitudinal Atrophy Trajectories", fontweight='bold', fontsize=16, pad=15)
ax.legend(fontsize=12, frameon=False); sns.despine(); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Fig2_Trajectories.png"))

# === Figure 4: Distribution ===
log("Drawing Figure 4...")
full_slopes = []
for ptid, group in df.groupby('PTID'):
    if len(group) < 2: continue
    try:
        r = smf.ols("hippo_vol_z ~ Time", data=group).fit()
        base = group.iloc[0]
        if base['e4_count'] == 1:
            a = "A+" if base['amyloid_status'] == 'Positive' else "A-"
            t = "pT+" if base['ptau_status'] == 'Positive' else "pT-"
            full_slopes.append({'Group': f"{a}{t}", 'Rate': r.params['Time']})
    except: pass

plot_data = pd.DataFrame(full_slopes)
if not plot_data.empty:
    fig, ax = plt.subplots(figsize=(9, 6))
    order = ['A-pT-', 'A-pT+', 'A+pT-', 'A+pT+']
    pal = {'A-pT-': COLORS['Safe'], 'A-pT+': COLORS['pTau'], 'A+pT-': COLORS['Risk'], 'A+pT+': "#8B0000"}
    sns.violinplot(data=plot_data, x='Group', y='Rate', order=order, palette=pal, alpha=0.3, inner=None, ax=ax)
    sns.swarmplot(data=plot_data, x='Group', y='Rate', order=order, color='k', size=3, alpha=0.6, ax=ax)
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_ylabel("Annual Atrophy Rate (Z-score/year)", fontweight='bold', fontsize=12)
    ax.set_xlabel("Amyloid / p-Tau Status (in APOE $\\epsilon$4 Carriers)", fontweight='bold', fontsize=12)
    ax.set_title("Distribution of Atrophy Rates by AT Status", fontweight='bold', fontsize=14)
    sns.despine(); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig4_Distribution.png"))

log("All Done. Check 'Results_Final_QC_Version'.")
log_file.close()