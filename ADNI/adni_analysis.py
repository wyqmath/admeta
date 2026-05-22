"""
ADNI Analysis: Gene-Dose LMM, Two-Cohort Meta-Analysis, and CSF Biomarker Interactions
"""
import pandas as pd
import numpy as np
import os
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.stats import zscore, norm, chi2
from scipy import stats
import warnings

# ===================================================================
# Part 0: Configuration
# ===================================================================
warnings.filterwarnings('ignore')
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
    "Non-carrier": "#4878A6",
    "Heterozygote": "#E8923B",
    "Homozygote": "#C44E52"
}

COLORS_BIO = {
    "Abeta": "#D62728",
    "pTau":  "#1F77B4",
    "tTau":  "#9467BD",
    "Safe":  "#2CA02C",
    "Risk":  "#D62728",
}

CUTOFFS = {'Abeta': 976.6, 'pTau': 21.8, 'tTau': 245}

script_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(script_dir, '..', 'data', 'ADNI_z.csv')
output_dir = os.path.join(script_dir, 'Results_Model1_DiagInteraction')
polished_dir = os.path.join(script_dir, '..', 'output')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(polished_dir, exist_ok=True)

# ===================================================================
# Part 1: Data Loading & Preprocessing
# ===================================================================
print("=" * 70)
print("ADNI Model 1: Gene-Dose Effect WITH Time x Diagnosis Interaction")
print("=" * 70)

print("\n--- Step 1: Loading and Preprocessing ---")
df = pd.read_csv(data_file)
print(f"  Raw data: {len(df)} observations, {df['PTID'].nunique()} subjects")

df = df.dropna(subset=['Hippocampus', 'Age', 'Sex', 'ICV', 'e4_count']).copy()
print(f"  After dropping missing: {len(df)} obs, {df['PTID'].nunique()} subjects")

df['Scan_Date'] = pd.to_datetime(df['Scan_Date'], format='%Y%m%d', errors='coerce')
df = df.dropna(subset=['Scan_Date']).sort_values(by=['PTID', 'Scan_Date'])
df['Baseline_Date'] = df.groupby('PTID')['Scan_Date'].transform('min')
df['Time'] = (df['Scan_Date'] - df['Baseline_Date']).dt.days / 365.25

baseline_info = df.loc[df['Time'] == 0, ['PTID', 'diagnosis']].drop_duplicates(subset=['PTID'])
baseline_info.rename(columns={'diagnosis': 'Baseline_Diagnosis'}, inplace=True)
df = df.merge(baseline_info, on='PTID', how='left')

df['APOE4_Dosage'] = pd.Categorical(
    df['e4_count'].map({0: 'Non-carrier', 1: 'Heterozygote', 2: 'Homozygote'}),
    categories=['Non-carrier', 'Heterozygote', 'Homozygote'], ordered=True
)

df['Field_Strength'] = df['Field_Strength'].astype(str)

# ===================================================================
# Part 2: Quality Control
# ===================================================================
print("\n--- Step 2: Quality Control ---")

visit_counts = df.groupby('PTID').size()
valid_ptids = visit_counts[visit_counts >= 2].index
df = df[df['PTID'].isin(valid_ptids)].copy()
print(f"  After >=2 visits filter: {len(df)} obs, {df['PTID'].nunique()} subjects")

df['hippo_z'] = zscore(df['Hippocampus'])
n_outliers = (df['hippo_z'].abs() > 4).sum()
df = df[df['hippo_z'].abs() <= 4].copy()
print(f"  After outlier removal (|Z|>4): removed {n_outliers} obs -> {len(df)} obs, {df['PTID'].nunique()} subjects")

# ===================================================================
# Part 3: Variable Preparation
# ===================================================================
print("\n--- Step 3: Preparing Variables ---")

df['Age_bl'] = df.groupby('PTID')['Age'].transform('first')
df['Age_Centered'] = df['Age_bl'] - df['Age_bl'].mean()
df['ICV_Scaled'] = df['ICV'] / 1000.0

print(f"\n  Final analytical sample:")
print(f"    Observations: {len(df)}")
print(f"    Subjects: {df['PTID'].nunique()}")
final_apoe = df.drop_duplicates('PTID')['APOE4_Dosage'].value_counts().sort_index()
for cat, n in final_apoe.items():
    print(f"    {cat}: {n}")
print(f"\n  Diagnosis distribution (baseline):")
dx_dist = df.drop_duplicates('PTID')['Baseline_Diagnosis'].value_counts()
for dx, n in dx_dist.items():
    print(f"    {dx}: {n}")
print(f"\n  Mean follow-up time: {df.groupby('PTID')['Time'].max().mean():.2f} years")

# ===================================================================
# Part 4: Gene-Dose LMM (with Time x Diagnosis interaction)
# ===================================================================
print("\n--- Step 4: Fitting Model WITH Time x Diagnosis Interaction ---")

ref_diag = 'CN' if 'CN' in df['Baseline_Diagnosis'].unique() else df['Baseline_Diagnosis'].mode()[0]
print(f"  Reference diagnosis: {ref_diag}")

formula = (
    f"Hippocampus ~ Time * C(APOE4_Dosage, Treatment('Non-carrier')) "
    f"+ Time * Age_Centered "
    f"+ Time * C(Sex) "
    f"+ Time * C(Baseline_Diagnosis, Treatment('{ref_diag}')) "
    f"+ ICV_Scaled "
    f"+ C(Field_Strength, Treatment('1.5')) "
    f"+ Time:C(Field_Strength, Treatment('1.5'))"
)

print(f"\n  Formula: {formula}\n")

model = smf.mixedlm(formula, df, groups=df['PTID'], re_formula="1 + Time")

result = None
for method in ['lbfgs', 'nm', 'powell']:
    try:
        result = model.fit(method=method, maxiter=5000)
        if result.converged:
            print(f"  Model converged with method: {method}")
            break
    except Exception as e:
        print(f"  Method {method} failed: {str(e)[:80]}")
        continue

if result is None or not result.converged:
    print("  WARNING: Random slope model did not converge. Trying random intercept only...")
    model = smf.mixedlm(formula, df, groups=df['PTID'], re_formula="1")
    result = model.fit(method='lbfgs', maxiter=5000)
    print(f"  Random intercept model converged: {result.converged}")

# ===================================================================
# Part 5: Results Extraction & Key Interaction Terms
# ===================================================================
print("\n--- Step 5: Results ---")
print("\n" + "=" * 70)
print("FULL MODEL COEFFICIENTS (with Time x Diagnosis)")
print("=" * 70)

coef_df = pd.DataFrame({
    'Coefficient': result.params,
    'Std.Error': result.bse,
    'z-value': result.tvalues,
    'P-value': result.pvalues,
    'CI_lower': result.params - 1.96 * result.bse,
    'CI_upper': result.params + 1.96 * result.bse
})
print(coef_df.to_string())
coef_df.to_csv(os.path.join(output_dir, 'Model1_Full_Coefficients.csv'))

with open(os.path.join(output_dir, 'Model_Summary.txt'), 'w') as f:
    f.write(result.summary().as_text())

params_idx = result.params.index.tolist()
het_term = [p for p in params_idx if 'Time' in p and 'Heterozygote' in p and 'Field' not in p][0]
homo_term = [p for p in params_idx if 'Time' in p and 'Homozygote' in p and 'Field' not in p][0]

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
print(f"    p = {het_p:.6e}")

print(f"\n  Time x Homozygote:")
print(f"    beta = {homo_beta:.2f} mm3/year")
print(f"    SE = {homo_se:.2f}")
print(f"    95% CI = [{homo_ci[0]:.2f}, {homo_ci[1]:.2f}]")
print(f"    p = {homo_p:.6e}")

time_beta = result.params['Time']
time_se = result.bse['Time']
time_p = result.pvalues['Time']
print(f"\n  Time (Non-carrier atrophy rate):")
print(f"    beta = {time_beta:.2f} mm3/year")
print(f"    SE = {time_se:.2f}")
print(f"    p = {time_p:.6e}")

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
    print(f"    p = {p_d:.6e}")

# ===================================================================
# Part 6: Two-Cohort Meta-Analysis (NACC + ADNI pooled)
# ===================================================================
print("\n" + "=" * 70)
print("TWO-COHORT FIXED-EFFECT META-ANALYSIS")
print("=" * 70)

NACC_RESULTS_FILE = os.path.join(script_dir, '..', 'NACC', 'Results_Model1_DiagInteraction', 'Model_Full_Coefficients.csv')

nacc_homo_beta = None
nacc_homo_se = None
nacc_het_beta = None
nacc_het_se = None

if os.path.exists(NACC_RESULTS_FILE):
    print("\n  Loading NACC updated results...")
    nacc_results = pd.read_csv(NACC_RESULTS_FILE, index_col=0)
    for idx in nacc_results.index:
        if 'Time' in idx and 'Homozygote' in idx:
            nacc_homo_beta = nacc_results.loc[idx, 'Coefficient']
            nacc_homo_se = nacc_results.loc[idx, 'Std.Err']
        if 'Time' in idx and 'Heterozygote' in idx:
            nacc_het_beta = nacc_results.loc[idx, 'Coefficient']
            nacc_het_se = nacc_results.loc[idx, 'Std.Err']
    print(f"    NACC Time x Homo: beta = {nacc_homo_beta:.2f}, SE = {nacc_homo_se:.2f}")
    print(f"    NACC Time x Het: beta = {nacc_het_beta:.2f}, SE = {nacc_het_se:.2f}")
else:
    print("\n  NACC results not yet available. Using original values as placeholder.")
    nacc_homo_beta = -75.11
    nacc_homo_se = 36.49
    nacc_het_beta = -26.37
    nacc_het_se = 21.62

adni_homo_beta = homo_beta
adni_homo_se = homo_se
adni_het_beta = het_beta
adni_het_se = het_se

print(f"\n  NACC:")
print(f"    Time x Homozygote: beta = {nacc_homo_beta:.2f}, SE = {nacc_homo_se:.2f}")
print(f"    Time x Heterozygote: beta = {nacc_het_beta:.2f}, SE = {nacc_het_se:.2f}")
print(f"\n  ADNI:")
print(f"    Time x Homozygote: beta = {adni_homo_beta:.2f}, SE = {adni_homo_se:.2f}")
print(f"    Time x Heterozygote: beta = {adni_het_beta:.2f}, SE = {adni_het_se:.2f}")


def fixed_effect_meta(betas, ses):
    weights = 1.0 / np.array(ses)**2
    beta_pooled = np.sum(np.array(betas) * weights) / np.sum(weights)
    se_pooled = np.sqrt(1.0 / np.sum(weights))
    z = beta_pooled / se_pooled
    p = 2 * (1 - norm.cdf(abs(z)))
    ci_lower = beta_pooled - 1.96 * se_pooled
    ci_upper = beta_pooled + 1.96 * se_pooled
    Q = np.sum(weights * (np.array(betas) - beta_pooled)**2)
    df_q = len(betas) - 1
    p_Q = 1 - chi2.cdf(Q, df_q)
    I2 = max(0, (Q - df_q) / Q * 100) if Q > 0 else 0
    return beta_pooled, se_pooled, ci_lower, ci_upper, p, Q, df_q, p_Q, I2


print("\n  --- Pooled Estimate: Homozygote ---")
bp, sep, cil, ciu, pp, Q, df_q, pQ, I2 = fixed_effect_meta(
    [nacc_homo_beta, adni_homo_beta], [nacc_homo_se, adni_homo_se])
print(f"    Pooled beta = {bp:.2f} mm3/year")
print(f"    SE = {sep:.2f}")
print(f"    95% CI = [{cil:.2f}, {ciu:.2f}]")
print(f"    p = {pp:.6e}")
print(f"    Cochran's Q = {Q:.3f}, df = {df_q}, p = {pQ:.4f}")
print(f"    I2 = {I2:.1f}%")

print("\n  --- Pooled Estimate: Heterozygote ---")
bp_h, sep_h, cil_h, ciu_h, pp_h, Q_h, df_q_h, pQ_h, I2_h = fixed_effect_meta(
    [nacc_het_beta, adni_het_beta], [nacc_het_se, adni_het_se])
print(f"    Pooled beta = {bp_h:.2f} mm3/year")
print(f"    SE = {sep_h:.2f}")
print(f"    95% CI = [{cil_h:.2f}, {ciu_h:.2f}]")
print(f"    p = {pp_h:.6e}")
print(f"    Cochran's Q = {Q_h:.3f}, df = {df_q_h}, p = {pQ_h:.4f}")
print(f"    I2 = {I2_h:.1f}%")

pooled_results = pd.DataFrame({
    'Genotype': ['Homozygote', 'Heterozygote'],
    'Pooled_Beta': [bp, bp_h],
    'Pooled_SE': [sep, sep_h],
    'CI_Lower': [cil, cil_h],
    'CI_Upper': [ciu, ciu_h],
    'P_value': [pp, pp_h],
    'Q': [Q, Q_h],
    'I2': [I2, I2_h],
    'NACC_Beta': [nacc_homo_beta, nacc_het_beta],
    'NACC_SE': [nacc_homo_se, nacc_het_se],
    'ADNI_Beta': [adni_homo_beta, adni_het_beta],
    'ADNI_SE': [adni_homo_se, adni_het_se]
})
pooled_results.to_csv(os.path.join(output_dir, 'Pooled_Meta_Analysis.csv'), index=False)

# ===================================================================
# Part 7: Figures — Gene-Dose (Trajectories + Rates + Forest Plot)
# ===================================================================
print("\n--- Step 6: Generating Gene-Dose Figures ---")

slope_nc = time_beta
slope_het = time_beta + het_beta
slope_homo = time_beta + homo_beta

se_nc = time_se
se_het_slope = np.sqrt(time_se**2 + het_se**2)
se_homo_slope = np.sqrt(time_se**2 + homo_se**2)

# Figure: Longitudinal Trajectories
print("Generating trajectory plot...")
fig1, ax1 = plt.subplots(figsize=(8, 6.5), constrained_layout=True)

np.random.seed(42)
sample_ids = np.random.choice(df['PTID'].unique(), size=min(400, df['PTID'].nunique()), replace=False)
subset = df[df['PTID'].isin(sample_ids)].copy()
subset['Time_Jitter'] = subset['Time'] + np.random.uniform(-0.1, 0.1, size=len(subset))

for group, color in PALETTE.items():
    grp_data = subset[subset['APOE4_Dosage'] == group]
    ax1.scatter(grp_data['Time_Jitter'], grp_data['Hippocampus'],
                color=color, alpha=0.1, s=12, linewidth=0, zorder=1)

time_points = np.linspace(0, 8, 100)
mean_start_vol = df.loc[df['Time'] == 0, 'Hippocampus'].mean()

slopes = [slope_nc, slope_het, slope_homo]
ses_slope = [se_nc, se_het_slope, se_homo_slope]

for i, (group, color) in enumerate(PALETTE.items()):
    y_values = mean_start_vol + slopes[i] * time_points
    ax1.plot(time_points, y_values, color=color, linewidth=3.5, label=group, zorder=3)
    ci_upper = y_values + 1.96 * ses_slope[i] * time_points
    ci_lower = y_values - 1.96 * ses_slope[i] * time_points
    ax1.fill_between(time_points, ci_lower, ci_upper, color=color, alpha=0.22, zorder=2)

ax1.set_xlabel("Time from Baseline (Years)")
ax1.set_ylabel("Hippocampal Volume (mm³)")
ax1.set_title("ADNI: HV Trajectories", pad=12)
ax1.legend(title="APOE4 Genotype", loc='lower left', frameon=False)
ax1.set_xlim(-0.3, 8)
sns.despine(ax=ax1, top=True, right=True)

fig1.savefig(os.path.join(polished_dir, 'Fig4c_ADNI_Trajectories.png'), bbox_inches='tight')
plt.close(fig1)
print("  -> Fig4c_ADNI_Trajectories.png saved")

# Figure: Annual Atrophy Rate Bar Chart
print("Generating rate bar chart...")
fig2, ax2 = plt.subplots(figsize=(8, 6.5), constrained_layout=True)

rates = [slope_nc, slope_het, slope_homo]
errors = ses_slope
groups = ["Non-carrier", "Heterozygote", "Homozygote"]
colors_bar = list(PALETTE.values())

bars = ax2.bar(groups, rates, yerr=errors, color=colors_bar,
               capsize=5, width=0.5, edgecolor='none',
               alpha=0.85, zorder=3,
               error_kw={'elinewidth': 1.8, 'capthick': 1.2})

ax2.axhline(0, color='black', linewidth=0.6, zorder=4)

lowest_point = min([r - e for r, e in zip(rates, errors)])
ax2.set_ylim(lowest_point * 1.6, 30)

def p_text(p):
    if p < 0.001:
        return f"P = {p:.2e}"
    else:
        return f"P = {p:.4f}"

line_y = lowest_point - 20
ax2.plot([0, 0, 2, 2], [line_y, line_y - 8, line_y - 8, line_y], lw=1.2, c='k')
ax2.text(1, line_y - 14, p_text(homo_p), ha='center', va='top', fontweight='bold', fontsize=13)

fold_change = slope_homo / slope_nc
ax2.text(2, rates[2] - errors[2] - 35, f"{fold_change:.1f}x",
         ha='center', va='top', color=PALETTE["Homozygote"], fontweight='bold', fontsize=14)

ax2.set_ylabel("Annual Change Rate (mm³/year)")
ax2.set_title("ADNI: Atrophy Rate", pad=12)
sns.despine(ax=ax2, top=True, right=True)

fig2.savefig(os.path.join(polished_dir, 'Fig4d_ADNI_Rates.png'), bbox_inches='tight')
plt.close(fig2)
print("  -> Fig4d_ADNI_Rates.png saved")

# Figure: Two-Cohort Forest Plot
print("Generating forest plot...")

homo_pooled = fixed_effect_meta([nacc_homo_beta, adni_homo_beta], [nacc_homo_se, adni_homo_se])[:4]
het_pooled = fixed_effect_meta([nacc_het_beta, adni_het_beta], [nacc_het_se, adni_het_se])[:4]

fig3, (ax_homo, ax_het) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True, constrained_layout=True)

def draw_forest(ax, title, nacc_b, nacc_se, adni_b, adni_se, pooled):
    bp_f, sep_f, cil_f, ciu_f = pooled
    studies = ['NACC', 'ADNI', 'Pooled']
    betas = [nacc_b, adni_b, bp_f]
    ci_lows = [nacc_b - 1.96*nacc_se, adni_b - 1.96*adni_se, cil_f]
    ci_highs = [nacc_b + 1.96*nacc_se, adni_b + 1.96*adni_se, ciu_f]

    y_pos = [2, 1, 0]
    colors_fp = ['#4878A6', '#C44E52', '#2D2D2D']
    max_hi = max(ci_highs)

    for i, (y, b, lo, hi, c) in enumerate(zip(y_pos, betas, ci_lows, ci_highs, colors_fp)):
        marker = 'D' if i == 2 else 'o'
        size = 220 if i == 2 else 120
        ax.errorbar(b, y, xerr=[[b - lo], [hi - b]], fmt='none',
                    ecolor=c, elinewidth=2.5, capsize=0, zorder=2)
        ax.scatter(b, y, color=c, s=size, marker=marker, zorder=3, edgecolors='none')
        label = f"β = {b:.1f} [{lo:.1f}, {hi:.1f}]"
        ax.text(max_hi + 12, y, label, va='center', fontsize=14)

    ax.axvline(0, color='grey', linestyle='--', linewidth=0.8, zorder=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(studies)
    ax.set_xlabel("β (mm³/year)")
    ax.set_title(title, fontsize=19)
    ax.set_ylim(-0.7, 2.7)
    sns.despine(ax=ax, left=True, top=True, right=True)
    ax.tick_params(left=False)

draw_forest(ax_homo, "Time x Homozygote",
            nacc_homo_beta, nacc_homo_se, adni_homo_beta, adni_homo_se, homo_pooled)
draw_forest(ax_het, "Time x Heterozygote",
            nacc_het_beta, nacc_het_se, adni_het_beta, adni_het_se, het_pooled)

x_min = min(ax_homo.get_xlim()[0], ax_het.get_xlim()[0])
x_max = max(ax_homo.get_xlim()[1], ax_het.get_xlim()[1])
ax_homo.set_xlim(x_min, x_max)
ax_het.set_xlim(x_min, x_max)

fig3.savefig(os.path.join(polished_dir, 'Fig5a_TwoCohort_Forest.png'), bbox_inches='tight')
plt.close(fig3)
print("  -> Fig5a_TwoCohort_Forest.png saved")

# ===================================================================
# Part 8: CSF Biomarker Interaction Analysis
# ===================================================================
print("\n" + "=" * 70)
print("CSF BIOMARKER INTERACTION ANALYSIS")
print("=" * 70)

def load_biomarker_data(filepath):
    df_bio = pd.read_csv(filepath)
    rename_map = {'CSF_Abeta42': 'csf_abeta', 'CSF_pTau': 'csf_ptau', 'CSF_tTau': 'csf_ttau', 'Hippocampus': 'hippo_vol'}
    df_bio = df_bio.rename(columns=rename_map)
    for col in ['csf_abeta', 'csf_ptau', 'csf_ttau']:
        if col in df_bio.columns:
            df_bio[col] = pd.to_numeric(df_bio[col].astype(str).str.replace(r'[><]', '', regex=True), errors='coerce')

    df_bio['amyloid_status'] = np.where(df_bio['csf_abeta'] < CUTOFFS['Abeta'], "Positive", "Negative")
    df_bio['ptau_status'] = np.where(df_bio['csf_ptau'] > CUTOFFS['pTau'], "Positive", "Negative")
    df_bio['ttau_status'] = np.where(df_bio['csf_ttau'] > CUTOFFS['tTau'], "Positive", "Negative")

    df_bio['Scan_Date'] = pd.to_datetime(df_bio['Scan_Date'], format='%Y%m%d', errors='coerce')
    df_bio = df_bio.sort_values(by=['PTID', 'Scan_Date'])
    df_bio['Baseline_Date'] = df_bio.groupby('PTID')['Scan_Date'].transform('min')
    df_bio['Time'] = (df_bio['Scan_Date'] - df_bio['Baseline_Date']).dt.days / 365.25
    df_bio['Age_bl'] = df_bio.groupby('PTID')['Age'].transform('first')
    df_bio['APOE4_Group'] = df_bio['e4_count'].map({0: 'Non-carrier', 1: 'Heterozygote', 2: 'Homozygote'})
    df_bio['Field_Strength'] = df_bio['Field_Strength'].astype(str)

    baseline_info_bio = df_bio.loc[df_bio.groupby('PTID')['Scan_Date'].idxmin(), ['PTID', 'diagnosis']]
    baseline_info_bio.rename(columns={'diagnosis': 'Baseline_Diagnosis'}, inplace=True)
    df_bio = df_bio.merge(baseline_info_bio, on='PTID', how='left')

    for col in ['Age_bl', 'ICV', 'hippo_vol']:
        df_bio[f'{col}_z'] = (df_bio[col] - df_bio[col].mean()) / df_bio[col].std()

    cols = ['PTID', 'Time', 'hippo_vol', 'hippo_vol_z', 'APOE4_Group', 'e4_count',
            'amyloid_status', 'ptau_status', 'ttau_status',
            'csf_abeta', 'csf_ptau', 'Age_bl_z', 'Sex', 'ICV_z', 'Field_Strength',
            'Baseline_Diagnosis']
    return df_bio[cols]


def perform_biomarker_qc(df_bio):
    df_bio = df_bio.dropna(subset=['hippo_vol', 'Age_bl_z', 'ICV_z', 'csf_abeta', 'csf_ptau', 'Baseline_Diagnosis'])
    counts = df_bio['PTID'].value_counts()
    valid_ptids = counts[counts >= 2].index
    df_bio = df_bio[df_bio['PTID'].isin(valid_ptids)]
    df_bio = df_bio[df_bio['hippo_vol_z'].abs() <= 4]
    for col in ['hippo_vol']:
        df_bio[f'{col}_z'] = (df_bio[col] - df_bio[col].mean()) / df_bio[col].std()
    return df_bio


def fit_best_lmm(data, formula_str):
    mdl = smf.mixedlm(formula_str, data, groups=data["PTID"], re_formula="1 + Time")
    for opt in ['nm', 'lbfgs', 'powell']:
        try:
            res = mdl.fit(method=opt, maxiter=5000, remix=False)
            if res.converged:
                return res, "Random Slope"
        except:
            pass
    try:
        mdl = smf.mixedlm(formula_str, data, groups=data["PTID"], re_formula="1")
        res = mdl.fit(method='lbfgs')
        return res, "Random Intercept"
    except:
        return None, "Failed"


print("\nLoading biomarker data...")
df_bio_raw = load_biomarker_data(data_file)
df_bio = perform_biomarker_qc(df_bio_raw)
print(f"  Biomarker sample: {df_bio['PTID'].nunique()} subjects, {len(df_bio)} observations")

covars = "Age_bl_z + C(Sex) + ICV_z + C(Field_Strength, Treatment('1.5')) + Time:C(Field_Strength, Treatment('1.5')) + C(Baseline_Diagnosis, Treatment('CN')) + Time:C(Baseline_Diagnosis, Treatment('CN'))"
models_config = [
    {'id': 'S_Abeta', 'form': f"hippo_vol_z ~ Time * C(APOE4_Group, Treatment('Non-carrier')) * C(amyloid_status, Treatment('Negative')) + {covars}", 'kw': 'amyloid', 'name': 'Single Factor Model', 'marker': 'Abeta'},
    {'id': 'S_pTau',  'form': f"hippo_vol_z ~ Time * C(APOE4_Group, Treatment('Non-carrier')) * C(ptau_status, Treatment('Negative')) + {covars}", 'kw': 'ptau',    'name': 'Single Factor Model', 'marker': 'pTau'},
    {'id': 'S_tTau',  'form': f"hippo_vol_z ~ Time * C(APOE4_Group, Treatment('Non-carrier')) * C(ttau_status, Treatment('Negative')) + {covars}", 'kw': 'ttau',    'name': 'Single Factor Model', 'marker': 'tTau'},
    {'id': 'J_pTau',  'form': f"hippo_vol_z ~ Time + {covars} + Time * C(APOE4_Group, Treatment('Non-carrier')) * C(amyloid_status, Treatment('Negative')) + Time * C(APOE4_Group, Treatment('Non-carrier')) * C(ptau_status, Treatment('Negative'))",
     'extract': [('amyloid', 'Abeta', 'Joint Model\n(adj. for p-Tau)'), ('ptau', 'pTau', 'Joint Model\n(adj. for Amyloid)')]},
    {'id': 'J_tTau',  'form': f"hippo_vol_z ~ Time + {covars} + Time * C(APOE4_Group, Treatment('Non-carrier')) * C(amyloid_status, Treatment('Negative')) + Time * C(APOE4_Group, Treatment('Non-carrier')) * C(ttau_status, Treatment('Negative'))",
     'extract': [('amyloid', 'Abeta', 'Joint Model\n(adj. for t-Tau)'), ('ttau', 'tTau', 'Joint Model\n(adj. for Amyloid)')]}
]

bio_results = []
all_model_results = []
print("\nFitting biomarker interaction models...")
for cfg in models_config:
    res_bio, tag = fit_best_lmm(df_bio, cfg['form'])
    if not res_bio:
        continue
    params_bio = res_bio.params.index

    for param in params_bio:
        all_model_results.append({
            'Model_ID': cfg['id'],
            'Parameter': param,
            'Coefficient': res_bio.params[param],
            'SE': res_bio.bse[param],
            'P_value': res_bio.pvalues[param],
            'CI_lower': res_bio.params[param] - 1.96 * res_bio.bse[param],
            'CI_upper': res_bio.params[param] + 1.96 * res_bio.bse[param]
        })

    if 'extract' in cfg:
        for kw, marker, disp_name in cfg['extract']:
            term_bio = [p for p in params_bio if 'Time' in p and 'Heterozygote' in p and 'Positive' in p and kw in p][0]
            bio_results.append({'Model': disp_name, 'Marker': marker, 'Coef': res_bio.params[term_bio], 'SE': res_bio.bse[term_bio], 'P_Raw': res_bio.pvalues[term_bio]})
    else:
        term_bio = [p for p in params_bio if 'Time' in p and 'Heterozygote' in p and 'Positive' in p and cfg['kw'] in p][0]
        bio_results.append({'Model': cfg['name'], 'Marker': cfg['marker'], 'Coef': res_bio.params[term_bio], 'SE': res_bio.bse[term_bio], 'P_Raw': res_bio.pvalues[term_bio]})

res_bio_df = pd.DataFrame(bio_results)
_, res_bio_df['P_FDR'], _, _ = multipletests(res_bio_df['P_Raw'], method='fdr_bh')
res_bio_df.to_csv(os.path.join(polished_dir, "Table_Statistics.csv"), index=False)

full_res_df = pd.DataFrame(all_model_results)
full_res_df.to_csv(os.path.join(polished_dir, "Table_Full_Model_Coefficients.csv"), index=False)
print(f"  Exported full coefficients table with {len(full_res_df)} parameters.")

# ===================================================================
# Part 9: Figures — Biomarker (Forest + Trajectories + Distribution)
# ===================================================================
print("\n--- Step 7: Generating Biomarker Figures ---")

# Forest plot
print("Drawing biomarker forest plot...")
fig, ax = plt.subplots(figsize=(10, 7.5))
res_bio_df['Sort1'] = res_bio_df['Marker'].apply(lambda x: 0 if x == 'Abeta' else (1 if x == 'pTau' else 2))
res_bio_df['Sort2'] = res_bio_df['Model'].apply(lambda x: 0 if 'Single' in x else 1)
plot_df = res_bio_df.sort_values(by=['Sort1', 'Sort2'], ascending=[False, False]).reset_index(drop=True)

y_pos_bio = range(len(plot_df))
for i, row in plot_df.iterrows():
    c = COLORS_BIO[row['Marker']]
    ax.errorbar(row['Coef'], i, xerr=row['SE']*1.96, fmt='o', color=c, ecolor=c, markersize=14, capsize=4, elinewidth=2.5)

for i, row in plot_df.iterrows():
    if row['P_Raw'] < 0.001:
        p_txt = f"P={row['P_Raw']:.2e}"
    else:
        p_txt = f"P={row['P_Raw']:.3f}"
    ax.text(0.165, i, p_txt, ha='right', va='center', fontsize=16, fontweight='bold', color='black')

ax.set_yticks(list(y_pos_bio))
ax.set_yticklabels(plot_df['Model'], fontsize=16, fontweight='bold')
ax.axvline(0, color='black', linestyle='--', alpha=0.4)
ax.set_xlabel("Synergistic Atrophy Rate (Standardized β)", fontsize=18)
ax.set_title("ADNI: Biomarker Interaction", fontsize=19, pad=20)
ax.set_xlim(-0.18, 0.18)
patches = [mpatches.Patch(color=COLORS_BIO['Abeta'], label='Amyloid Effect'),
           mpatches.Patch(color=COLORS_BIO['pTau'], label='p-Tau Effect'),
           mpatches.Patch(color=COLORS_BIO['tTau'], label='t-Tau Effect')]
ax.legend(handles=patches, loc='lower left', frameon=False, fontsize=14)
sns.despine(ax=ax, top=True, right=True)
plt.tight_layout()
plt.savefig(os.path.join(polished_dir, "Fig5c_Biomarker_Forest.png"))
plt.close()
print("  -> Fig5c_Biomarker_Forest.png saved")

# Trajectory plot
print("Drawing biomarker trajectory plot...")
subset_bio = df_bio[df_bio['e4_count'].isin([0, 1])].copy()
subset_bio['Group'] = subset_bio.apply(lambda x:
    "A+E+" if (x['amyloid_status']=='Positive' and x['e4_count']==1) else
    "A-E-" if (x['amyloid_status']=='Negative' and x['e4_count']==0) else "Other", axis=1)
subset_bio = subset_bio[subset_bio['Group'] != "Other"]

fig, ax = plt.subplots(figsize=(9, 7))
sns.regplot(data=subset_bio[subset_bio['Group']=="A-E-"], x='Time', y='hippo_vol', ax=ax, scatter=True,
            color=COLORS_BIO['Safe'], label="Aβ- / ε4- (Control)", line_kws={'lw': 4},
            scatter_kws={'s': 15, 'alpha': 0.3})
sns.regplot(data=subset_bio[subset_bio['Group']=="A+E+"], x='Time', y='hippo_vol', ax=ax, scatter=True,
            color=COLORS_BIO['Risk'], label="Aβ+ / ε4+ (Risk)", line_kws={'lw': 4},
            scatter_kws={'s': 15, 'alpha': 0.3})
ax.set_xlabel("Years from Baseline", fontsize=17)
ax.set_ylabel("Hippocampal Volume (mm³)", fontsize=17)
ax.set_title("ADNI: HV by Amyloid × APOE4", fontsize=19, pad=15)
ax.legend(fontsize=14, frameon=False)
sns.despine(ax=ax, top=True, right=True)
plt.tight_layout()
plt.savefig(os.path.join(polished_dir, "Fig5b_Biomarker_Trajectories.png"))
plt.close()
print("  -> Fig5b_Biomarker_Trajectories.png saved")

# Distribution plot
print("Drawing distribution plot...")
full_slopes = []
for ptid, group in df_bio.groupby('PTID'):
    if len(group) < 2:
        continue
    try:
        r = smf.ols("hippo_vol_z ~ Time", data=group).fit()
        base = group.iloc[0]
        if base['e4_count'] == 1:
            a = "A+" if base['amyloid_status'] == 'Positive' else "A-"
            t = "pT+" if base['ptau_status'] == 'Positive' else "pT-"
            full_slopes.append({'Group': f"{a}{t}", 'Rate': r.params['Time']})
    except:
        pass

plot_data = pd.DataFrame(full_slopes)
if not plot_data.empty:
    fig, ax = plt.subplots(figsize=(9, 6))
    order = ['A-pT-', 'A-pT+', 'A+pT-', 'A+pT+']
    pal = {'A-pT-': COLORS_BIO['Safe'], 'A-pT+': COLORS_BIO['pTau'], 'A+pT-': COLORS_BIO['Risk'], 'A+pT+': "#8B0000"}
    sns.violinplot(data=plot_data, x='Group', y='Rate', order=order, palette=pal, alpha=0.3, inner=None, ax=ax)
    sns.swarmplot(data=plot_data, x='Group', y='Rate', order=order, color='k', size=3, alpha=0.6, ax=ax)
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_ylabel("Annual Atrophy Rate (Z-score/year)", fontsize=14)
    ax.set_xlabel("Amyloid / p-Tau Status (in APOE ε4 Carriers)", fontsize=14)
    ax.set_title("ADNI: Atrophy Distribution", fontsize=17)
    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()
    plt.savefig(os.path.join(polished_dir, "Fig5d_Distribution.png"))
    plt.close()
    print("  -> Fig5d_Distribution.png saved")

# ===================================================================
# Summary
# ===================================================================
print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"  N subjects: {df['PTID'].nunique()}")
print(f"  N observations: {len(df)}")
print(f"  Time (NC rate): beta = {time_beta:.2f}, SE = {time_se:.2f}, p = {time_p:.6e}")
print(f"  Time x Homo: beta = {homo_beta:.2f}, SE = {homo_se:.2f}, p = {homo_p:.6e}")
print(f"  Time x Het: beta = {het_beta:.2f}, SE = {het_se:.2f}, p = {het_p:.6e}")
print(f"  Pooled Homo: beta = {bp:.2f}, 95% CI [{cil:.2f}, {ciu:.2f}], p = {pp:.6e}")
print(f"  Pooled Het: beta = {bp_h:.2f}, 95% CI [{cil_h:.2f}, {ciu_h:.2f}], p = {pp_h:.6e}")
print(f"\nResults saved to: {output_dir}")
print(f"Figures saved to: {polished_dir}")
