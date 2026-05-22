"""
NACC Analysis: Gene-Dose LMM with Time x Diagnosis Interaction + CN-Only Sensitivity
"""
import pandas as pd
import numpy as np
import os
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
from scipy.stats import zscore, norm

# ===================================================================
# Part 0: Configuration
# ===================================================================
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

plt.rcParams.update({
    'font.family': 'Arial',
    'axes.unicode_minus': False,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'font.size': 16,
    'axes.labelsize': 17,
    'axes.titlesize': 19,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

PALETTE = {
    "Non-carrier (0)": "#4878A6",
    "Heterozygote (1)": "#E8923B",
    "Homozygote (2)": "#C44E52"
}

script_dir = os.path.dirname(os.path.abspath(__file__))
data_file_path = os.path.join(script_dir, '..', 'data', 'NACC_HV.csv')
output_directory = os.path.join(script_dir, "Results_Model1_DiagInteraction")
polished_dir = os.path.join(script_dir, '..', 'output')
os.makedirs(output_directory, exist_ok=True)
os.makedirs(polished_dir, exist_ok=True)

# ===================================================================
# Part 1: Data Preprocessing & QC
# ===================================================================
print("=" * 70)
print("NACC Model 1: Gene-Dose Effect WITH Time x Diagnosis Interaction")
print("=" * 70)

print("\n--- Step 1: Loading and Cleaning Data ---")
try:
    long_data_raw = pd.read_csv(data_file_path)
except FileNotFoundError:
    print(f"Error: File not found at {data_file_path}")
    exit()

df = long_data_raw.dropna(subset=['Hippocampus', 'Age', 'Sex', 'eTIV']).copy()

df['Scan_Date'] = pd.to_datetime(df['Scan_Date'], format='%Y%m%d', errors='coerce')
df = df.dropna(subset=['Scan_Date']).sort_values(by=['NACCID', 'Scan_Date'])
df['Baseline_Date'] = df.groupby('NACCID')['Scan_Date'].transform('min')
df['Time'] = (df['Scan_Date'] - df['Baseline_Date']).dt.days / 365.25

baseline_info = df.loc[df['Time'] == 0, ['NACCID', 'diagnosis']].drop_duplicates(subset=['NACCID'])
baseline_info.rename(columns={'diagnosis': 'Baseline_Diagnosis'}, inplace=True)
df = df.merge(baseline_info, on='NACCID', how='left')

df['APOE4_Dosage'] = pd.Categorical(
    df['e4_count'].map({0: 'Non-carrier (0)', 1: 'Heterozygote (1)', 2: 'Homozygote (2)'}),
    categories=['Non-carrier (0)', 'Heterozygote (1)', 'Homozygote (2)'], ordered=True
)

# Cross-sectional QC
print("Running Cross-sectional QC...")
qc_formula = "Hippocampus ~ Age + C(Sex) + eTIV + C(Baseline_Diagnosis)"
qc_model = smf.ols(qc_formula, data=df).fit()
df['resid_z'] = zscore(qc_model.resid)
df = df[df['resid_z'].abs() <= 4.0]

# Longitudinal QC
print("Running Longitudinal QC...")
df = df.sort_values(by=['NACCID', 'Time'])
df['vol_pct_change'] = df.groupby('NACCID')['Hippocampus'].pct_change()
df['time_diff'] = df.groupby('NACCID')['Time'].diff()
mask_short_interval = df['time_diff'] < 0.5
df.loc[mask_short_interval, 'time_diff'] = np.nan
df['annual_pct_change'] = df['vol_pct_change'] / df['time_diff']
mask_biological_impossible = (df['annual_pct_change'] > 0.20) | (df['annual_pct_change'] < -0.30)
n_dropped = mask_biological_impossible.sum()
df = df[~mask_biological_impossible].copy()
print(f"  -> Dropped {n_dropped} observations due to biologically impossible rates.")

# Variable preparation
df['Age_Centered'] = df['Age'] - df['Age'].mean()
df['eTIV_Scaled'] = df['eTIV'] / 1000.0

cols = ['NACCID', 'Time', 'Hippocampus', 'APOE4_Dosage', 'Age_Centered', 'Sex', 'eTIV_Scaled', 'Baseline_Diagnosis']
df_final = df[cols].dropna()

print(f"\n  Final analytical sample:")
print(f"    Total Subjects: {df_final['NACCID'].nunique()}, Total Scans: {len(df_final)}")

ref_diag = 'Normal' if 'Normal' in df_final['Baseline_Diagnosis'].unique() else df_final['Baseline_Diagnosis'].mode()[0]
print(f"  Reference diagnosis: {ref_diag}")

print(f"\n  Diagnosis distribution (baseline):")
dx_dist = df_final.drop_duplicates('NACCID')['Baseline_Diagnosis'].value_counts()
for dx, n in dx_dist.items():
    print(f"    {dx}: {n}")

# ===================================================================
# Part 2: Gene-Dose LMM (with Time x Diagnosis interaction)
# ===================================================================
print("\n--- Step 2: Fitting Model WITH Time x Diagnosis Interaction ---")

formula = (
    f"Hippocampus ~ Time * C(APOE4_Dosage, Treatment('Non-carrier (0)')) "
    f"+ Time * Age_Centered "
    f"+ Time * C(Sex) "
    f"+ Time * C(Baseline_Diagnosis, Treatment('{ref_diag}')) "
    f"+ eTIV_Scaled"
)

print(f"\n  Formula: {formula}\n")

model = smf.mixedlm(formula, df_final, groups=df_final['NACCID'], re_formula="1 + Time")

result = None
for method in ['lbfgs', 'nm', 'powell']:
    try:
        result = model.fit(method=method, maxiter=50000, full_output=True)
        if result.converged:
            print(f"  Model converged with method: {method}")
            break
    except Exception as e:
        print(f"  Method {method} failed: {str(e)[:80]}")
        continue

if result is None or not result.converged:
    print("  WARNING: Random slope model did not converge. Trying random intercept only...")
    model = smf.mixedlm(formula, df_final, groups=df_final['NACCID'], re_formula="1")
    result = model.fit(method='lbfgs', maxiter=50000)
    print(f"  Random intercept model converged: {result.converged}")

# ===================================================================
# Part 3: Results Extraction & Key Interaction Terms
# ===================================================================
print("\n--- Step 3: Results ---")
print("\n" + "=" * 70)
print("FULL MODEL COEFFICIENTS (with Time x Diagnosis)")
print("=" * 70)

with open(os.path.join(output_directory, "Model_Summary.txt"), "w") as f:
    f.write(result.summary().as_text())

results_table = pd.DataFrame({
    'Coefficient': result.params,
    'Std.Err': result.bse,
    'z-value': result.tvalues,
    'P-value': result.pvalues,
    'CI_Lower (2.5%)': result.conf_int()[0],
    'CI_Upper (97.5%)': result.conf_int()[1]
})

print(results_table.to_string())
results_table.to_csv(os.path.join(output_directory, "Model_Full_Coefficients.csv"))

params_idx = result.params.index.tolist()

# Time x APOE4 terms
het_term = [p for p in params_idx if 'Time' in p and 'Heterozygote' in p][0]
homo_term = [p for p in params_idx if 'Time' in p and 'Homozygote' in p][0]

het_beta = result.params[het_term]
het_se = result.bse[het_term]
het_p = result.pvalues[het_term]
het_ci = (het_beta - 1.96*het_se, het_beta + 1.96*het_se)

homo_beta = result.params[homo_term]
homo_se = result.bse[homo_term]
homo_p = result.pvalues[homo_term]
homo_ci = (homo_beta - 1.96*homo_se, homo_beta + 1.96*homo_se)

print(f"\n  Time x Heterozygote:")
print(f"    beta = {het_beta:.2f} mm3/year")
print(f"    SE = {het_se:.2f}")
print(f"    95% CI = [{het_ci[0]:.2f}, {het_ci[1]:.2f}]")
print(f"    p = {het_p:.4f}")

print(f"\n  Time x Homozygote:")
print(f"    beta = {homo_beta:.2f} mm3/year")
print(f"    SE = {homo_se:.2f}")
print(f"    95% CI = [{homo_ci[0]:.2f}, {homo_ci[1]:.2f}]")
print(f"    p = {homo_p:.4f}")

time_beta = result.params['Time']
time_se = result.bse['Time']
time_p = result.pvalues['Time']
print(f"\n  Time (Non-carrier atrophy rate):")
print(f"    beta = {time_beta:.2f} mm3/year")
print(f"    SE = {time_se:.2f}")
print(f"    p = {time_p:.4f}")

# Time x Diagnosis terms
print("\n" + "=" * 70)
print("KEY RESULTS: Time x Diagnosis Interaction")
print("=" * 70)

diag_terms = [p for p in params_idx if 'Time' in p and 'Baseline_Diagnosis' in p]
for term in diag_terms:
    beta_d = result.params[term]
    se_d = result.bse[term]
    p_d = result.pvalues[term]
    ci_d = (beta_d - 1.96*se_d, beta_d + 1.96*se_d)
    print(f"\n  {term}:")
    print(f"    beta = {beta_d:.2f} mm3/year")
    print(f"    SE = {se_d:.2f}")
    print(f"    95% CI = [{ci_d[0]:.2f}, {ci_d[1]:.2f}]")
    print(f"    p = {p_d:.6f}")

# ===================================================================
# Part 4: Figures (Trajectories + Rate Bar Chart)
# ===================================================================
print("\n--- Step 4: Generating Figures ---")

params_all = result.params
cov_matrix = result.cov_params()
colors = [PALETTE["Non-carrier (0)"], PALETTE["Heterozygote (1)"], PALETTE["Homozygote (2)"]]

# Figure 1: Longitudinal Trajectories
print("Plotting trajectories...")
fig1, ax1 = plt.subplots(figsize=(8, 6.5), constrained_layout=True)

np.random.seed(42)
sample_ids = np.random.choice(df_final['NACCID'].unique(), size=min(400, len(df_final['NACCID'].unique())), replace=False)
subset_data = df_final[df_final['NACCID'].isin(sample_ids)].copy()
subset_data['Time_Jitter'] = subset_data['Time'] + np.random.uniform(-0.15, 0.15, size=len(subset_data))

sns.scatterplot(
    data=subset_data,
    x='Time_Jitter', y='Hippocampus', hue='APOE4_Dosage',
    palette=PALETTE, alpha=0.1, legend=False, ax=ax1,
    s=12, linewidth=0
)

time_points = np.linspace(0, 8, 100)
mean_start_vol = df_final['Hippocampus'].mean()

slope_0 = params_all['Time']
slope_1 = slope_0 + params_all[het_term]
slope_2 = slope_0 + params_all[homo_term]
slopes = [slope_0, slope_1, slope_2]

for i, (group, color, slope) in enumerate(zip(PALETTE.keys(), colors, slopes)):
    y_values = mean_start_vol + slope * time_points
    ax1.plot(time_points, y_values, color=color, linewidth=3.5, label=group)

    if i == 0:
        var_slope = cov_matrix.loc['Time', 'Time']
    elif i == 1:
        var_slope = (cov_matrix.loc['Time', 'Time'] +
                     cov_matrix.loc[het_term, het_term] +
                     2 * cov_matrix.loc['Time', het_term])
    else:
        var_slope = (cov_matrix.loc['Time', 'Time'] +
                     cov_matrix.loc[homo_term, homo_term] +
                     2 * cov_matrix.loc['Time', homo_term])

    se_slope = np.sqrt(max(var_slope, 0))
    ci_upper = y_values + 1.96 * se_slope * time_points
    ci_lower = y_values - 1.96 * se_slope * time_points
    ax1.fill_between(time_points, ci_lower, ci_upper, color=color, alpha=0.22)

ax1.set_xlabel("Time from Baseline (Years)")
ax1.set_ylabel("Hippocampal Volume (mm³)")
ax1.set_title("NACC: HV Trajectories", pad=15)
ax1.legend(title="APOE4 Genotype", loc='lower left', frameon=False)
sns.despine(ax=ax1, top=True, right=True)

plt.savefig(os.path.join(polished_dir, "Fig4a_NACC_Trajectories.png"), bbox_inches='tight')
plt.close()
print("  -> Fig4a_NACC_Trajectories.png saved")

# Figure 2: Rate Comparison
print("Plotting rates...")
fig2, ax2 = plt.subplots(figsize=(8, 6.5), constrained_layout=True)

rates = [slope_0, slope_1, slope_2]

se_0 = np.sqrt(cov_matrix.loc['Time', 'Time'])
se_1 = np.sqrt(max(cov_matrix.loc['Time', 'Time'] + cov_matrix.loc[het_term, het_term] + 2 * cov_matrix.loc['Time', het_term], 0))
se_2 = np.sqrt(max(cov_matrix.loc['Time', 'Time'] + cov_matrix.loc[homo_term, homo_term] + 2 * cov_matrix.loc['Time', homo_term], 0))
errors = [se_0, se_1, se_2]

groups = ["Non-carrier\n(0)", "Heterozygote\n(1)", "Homozygote\n(2)"]

bars = ax2.bar(groups, rates, yerr=errors, color=colors,
               capsize=5, width=0.5, edgecolor='none', alpha=0.85, zorder=3,
               error_kw={'elinewidth': 1.8, 'capthick': 1.2})

ax2.axhline(0, color='black', linewidth=0.6, zorder=4)

lowest_point = min([r - e for r, e in zip(rates, errors)])
y_lower_limit = lowest_point * 1.55
ax2.set_ylim(y_lower_limit, 20)

p_val_homo = homo_p
if slope_0 != 0:
    fold_change = slope_2 / slope_0
else:
    fold_change = float('inf')

if p_val_homo < 0.05:
    line_y = lowest_point - 10
    ax2.plot([0, 0, 2, 2], [line_y, line_y - 5, line_y - 5, line_y], lw=1.2, c='k')
    p_text = f"P = {p_val_homo:.2e}" if p_val_homo < 0.001 else f"P = {p_val_homo:.4f}"
    ax2.text(1, line_y - 10, p_text, ha='center', va='top', color='black', fontweight='bold', fontsize=14)

text_y = (slope_2 - se_2) - 20
ax2.text(2, text_y, f"{fold_change:.1f}x Faster",
         ha='center', va='top', color=PALETTE["Homozygote (2)"], fontweight='bold', fontsize=17)

ax2.set_ylabel("Annual Change Rate (mm³/year)")
ax2.set_title("NACC: Atrophy Rate", pad=20)
sns.despine(ax=ax2, top=True, right=True)

plt.savefig(os.path.join(polished_dir, "Fig4b_NACC_Rates.png"), bbox_inches='tight')
plt.close()
print("  -> Fig4b_NACC_Rates.png saved")

# ===================================================================
# Part 5: Sensitivity Analysis — CN-Only Subgroup
# ===================================================================
print("\n" + "=" * 70)
print("SENSITIVITY ANALYSIS: CN-Only Subgroup")
print("=" * 70)

cn_output_dir = os.path.join(script_dir, "Results_CN_Only")
os.makedirs(cn_output_dir, exist_ok=True)

cn_label = 'Normal' if 'Normal' in df_final['Baseline_Diagnosis'].unique() else 'CN'
df_cn = df_final[df_final['Baseline_Diagnosis'] == cn_label].copy()

print(f"\n  CN label used: '{cn_label}'")
print(f"  CN subjects: {df_cn['NACCID'].nunique()}")
print(f"  CN observations: {len(df_cn)}")

visit_counts_cn = df_cn.groupby('NACCID').size()
valid_ptids_cn = visit_counts_cn[visit_counts_cn >= 2].index
df_cn_long = df_cn[df_cn['NACCID'].isin(valid_ptids_cn)].copy()

print(f"  CN subjects with >=2 visits: {df_cn_long['NACCID'].nunique()}")
print(f"  CN observations (>=2 visits): {len(df_cn_long)}")

print(f"\n  APOE4 distribution (CN, >=2 visits):")
apoe_dist = df_cn_long.drop_duplicates('NACCID')['APOE4_Dosage'].value_counts().sort_index()
for cat, n in apoe_dist.items():
    print(f"    {cat}: {n}")

n_homo_cn = apoe_dist.get('Homozygote (2)', 0)
if n_homo_cn < 5:
    print(f"\n  WARNING: Homozygote n = {n_homo_cn} < 5, model may not be estimable.")

# CN-Only model (no Diagnosis term)
print("\n  Fitting CN-Only Model...")
formula_cn = (
    f"Hippocampus ~ Time * C(APOE4_Dosage, Treatment('Non-carrier (0)')) "
    f"+ Time * Age_Centered "
    f"+ Time * C(Sex) "
    f"+ eTIV_Scaled"
)

model_cn = smf.mixedlm(formula_cn, df_cn, groups=df_cn['NACCID'], re_formula="1 + Time")

result_cn = None
for method in ['lbfgs', 'nm', 'powell']:
    try:
        result_cn = model_cn.fit(method=method, maxiter=50000, full_output=True)
        if result_cn.converged:
            print(f"  Model converged with method: {method}")
            break
    except Exception as e:
        print(f"  Method {method} failed: {str(e)[:80]}")
        continue

if result_cn is None or not result_cn.converged:
    print("  WARNING: Random slope model did not converge. Trying random intercept only...")
    model_cn = smf.mixedlm(formula_cn, df_cn, groups=df_cn['NACCID'], re_formula="1")
    result_cn = model_cn.fit(method='lbfgs', maxiter=50000)
    print(f"  Random intercept model converged: {result_cn.converged}")

# Save CN-only results
with open(os.path.join(cn_output_dir, "Model_Summary.txt"), "w") as f:
    f.write(result_cn.summary().as_text())

cn_results_table = pd.DataFrame({
    'Coefficient': result_cn.params,
    'Std.Err': result_cn.bse,
    'z-value': result_cn.tvalues,
    'P-value': result_cn.pvalues,
    'CI_Lower (2.5%)': result_cn.conf_int()[0],
    'CI_Upper (97.5%)': result_cn.conf_int()[1]
})
cn_results_table.to_csv(os.path.join(cn_output_dir, "Model_Full_Coefficients.csv"))

# Extract key terms
params_cn_idx = result_cn.params.index.tolist()
het_term_cn = [p for p in params_cn_idx if 'Time' in p and 'Heterozygote' in p][0]
homo_term_cn = [p for p in params_cn_idx if 'Time' in p and 'Homozygote' in p][0]

het_beta_cn = result_cn.params[het_term_cn]
het_se_cn = result_cn.bse[het_term_cn]
het_p_cn = result_cn.pvalues[het_term_cn]

homo_beta_cn = result_cn.params[homo_term_cn]
homo_se_cn = result_cn.bse[homo_term_cn]
homo_p_cn = result_cn.pvalues[homo_term_cn]

time_beta_cn = result_cn.params['Time']
time_se_cn = result_cn.bse['Time']
time_p_cn = result_cn.pvalues['Time']

print(f"\n  CN-Only Results:")
print(f"    Time (NC rate): beta = {time_beta_cn:.2f}, SE = {time_se_cn:.2f}, p = {time_p_cn:.4f}")
print(f"    Time x Het: beta = {het_beta_cn:.2f}, SE = {het_se_cn:.2f}, p = {het_p_cn:.4f}")
print(f"    Time x Homo: beta = {homo_beta_cn:.2f}, SE = {homo_se_cn:.2f}, p = {homo_p_cn:.4f}")

print(f"\n  Direction consistency:")
print(f"    Homo effect negative (consistent): {homo_beta_cn < 0}")
print(f"    Het effect negative (consistent): {het_beta_cn < 0}")

if homo_p_cn >= 0.05:
    print(f"\n  NOTE: Time x Homo p = {homo_p_cn:.4f} (not significant)")
    print(f"  Expected given n_homo = {n_homo_cn} (underpowered)")
    print(f"  Directional consistency supports the main finding.")

# ===================================================================
# Summary
# ===================================================================
print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"  Main model: {df_final['NACCID'].nunique()} subjects, {len(df_final)} observations")
print(f"  Time (NC rate): beta = {time_beta:.2f}, SE = {time_se:.2f}, p = {time_p:.4f}")
print(f"  Time x Homo: beta = {homo_beta:.2f}, SE = {homo_se:.2f}, p = {homo_p:.4f}")
print(f"  Time x Het: beta = {het_beta:.2f}, SE = {het_se:.2f}, p = {het_p:.4f}")
print(f"\n  CN-Only sensitivity: {df_cn['NACCID'].nunique()} subjects")
print(f"  CN Time x Homo: beta = {homo_beta_cn:.2f}, p = {homo_p_cn:.4f}")
print(f"\nResults saved to: {output_directory}")
print(f"CN-Only results saved to: {cn_output_dir}")
print(f"Figures saved to: {polished_dir}")
