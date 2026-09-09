# Task 3: Goalkeeping (Save Percentage)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')


# STEP 1: DATA WRANGLING

df = pd.read_csv('goalkeepers_wc2026.csv', encoding='latin-1')

print("=" * 60)
print("STEP 1: DATA WRANGLING")
print("=" * 60)

print("\n[1.1] Raw dataset shape:", df.shape)
print("\nColumn names:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

print("\n[1.2] Missing values per column:")
print(df.isnull().sum())

print("\n[1.3] Data types:")
print(df.dtypes)

df_clean = df[(df['Minutes played'] > 0) & (df['Shots on target faced'] > 0)].copy()

removed = len(df) - len(df_clean)
print(f"\n[1.4] Removed {removed} goalkeeper(s) with 0 minutes played or 0 shots faced.")
print(f"      Remaining records: {len(df_clean)}")

df_clean['Save percentage'] = (df_clean['Saves'] / df_clean['Shots on target faced']) * 100

print("\n[1.5] Save percentage computed: save pct = (Saves / Shots on target faced) x 100")
print("\nSample of computed values:")
print(df_clean[['Goalkeeper name', 'Team', 'Confederation',
                 'Saves', 'Shots on target faced', 'Save percentage']].head(10).to_string(index=False))

uefa     = df_clean[df_clean['Confederation'] == 'UEFA']['Save percentage'].reset_index(drop=True)
non_uefa = df_clean[df_clean['Confederation'] != 'UEFA']['Save percentage'].reset_index(drop=True)

print(f"\n[1.6] Group sizes:")
print(f"      UEFA goalkeepers     : {len(uefa)}")
print(f"      Non-UEFA goalkeepers : {len(non_uefa)}")


# STEP 2: SAMPLING

print("\n" + "=" * 60)
print("STEP 2: SAMPLING")
print("=" * 60)

print(f"\nTotal qualifying goalkeepers (population/sample): {len(df_clean)}")
print(f"  UEFA     : {len(uefa)}")
print(f"  Non-UEFA : {len(non_uefa)}")


# STEP 3: DESCRIPTIVE STATISTICS

print("\n" + "=" * 60)
print("STEP 3: DESCRIPTIVE STATISTICS")
print("=" * 60)

def descriptive_stats(data, label):
    mean   = data.mean()
    median = data.median()
    std    = data.std(ddof=1)
    mn     = data.min()
    mx     = data.max()
    skew   = data.skew()
    kurt   = data.kurtosis()
    q1     = data.quantile(0.25)
    q3     = data.quantile(0.75)
    iqr    = q3 - q1
    n      = len(data)
    print(f"\n  {label} (n = {n})")
    print(f"  Mean               : {mean:.4f}%")
    print(f"  Median             : {median:.4f}%")
    print(f"  Std Dev (sample)   : {std:.4f}%")
    print(f"  Min                : {mn:.4f}%")
    print(f"  Max                : {mx:.4f}%")
    print(f"  Q1                 : {q1:.4f}%")
    print(f"  Q3                 : {q3:.4f}%")
    print(f"  IQR                : {iqr:.4f}%")
    print(f"  Skewness           : {skew:.4f}")
    print(f"  Kurtosis (excess)  : {kurt:.4f}")
    return mean, median, std, mn, mx, skew

stats_uefa     = descriptive_stats(uefa,     "UEFA Goalkeepers")
stats_non_uefa = descriptive_stats(non_uefa, "Non-UEFA Goalkeepers")

fig1, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
fig1.suptitle('Distribution of Save Percentage\nUEFA vs Non-UEFA Goalkeepers : FIFA World Cup 2026',
              fontsize=13, fontweight='bold')

axes[0].hist(uefa,     bins=10, color='#1f77b4', edgecolor='white', alpha=0.85)
axes[0].axvline(uefa.mean(), color='red', linestyle='--', linewidth=1.8, label=f'Mean = {uefa.mean():.1f}%')
axes[0].set_title('UEFA Goalkeepers', fontsize=12)
axes[0].set_xlabel('Save Percentage (%)', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].legend()

axes[1].hist(non_uefa, bins=10, color='#ff7f0e', edgecolor='white', alpha=0.85)
axes[1].axvline(non_uefa.mean(), color='red', linestyle='--', linewidth=1.8, label=f'Mean = {non_uefa.mean():.1f}%')
axes[1].set_title('Non-UEFA Goalkeepers', fontsize=12)
axes[1].set_xlabel('Save Percentage (%)', fontsize=11)
axes[1].legend()

fig1.tight_layout()
fig1.savefig('task3_histogram.png', dpi=150, bbox_inches='tight')
print("\n[Histogram saved as task3_histogram.png]")


# STEP 4: CONFIDENCE INTERVALS (95%)

print("\n" + "=" * 60)
print("STEP 4: 95% CONFIDENCE INTERVALS")
print("=" * 60)

def confidence_interval_95(data, label):
    n      = len(data)
    mean   = data.mean()
    se     = stats.sem(data)
    t_crit = stats.t.ppf(0.975, df=n-1)
    margin = t_crit * se
    lower  = mean - margin
    upper  = mean + margin
    print(f"\n  {label}")
    print(f"  n              : {n}")
    print(f"  Sample mean    : {mean:.4f}%")
    print(f"  Std Error (SE) : {se:.4f}%")
    print(f"  t critical     : {t_crit:.4f}  (df = {n-1}, a/2 = 0.025)")
    print(f"  Margin of error: {margin:.4f}%")
    print(f"  95% CI         : ({lower:.4f}%, {upper:.4f}%)")
    return lower, upper, mean

ci_uefa_lo,     ci_uefa_up,     mean_uefa     = confidence_interval_95(uefa,     "UEFA Goalkeepers")
ci_non_uefa_lo, ci_non_uefa_up, mean_non_uefa = confidence_interval_95(non_uefa, "Non-UEFA Goalkeepers")

overlap = not (ci_uefa_up < ci_non_uefa_lo or ci_non_uefa_up < ci_uefa_lo)
print(f"\n  CI overlap: {'YES: intervals overlap; inconclusive at CI level alone.' if overlap else 'NO: intervals do not overlap; strong evidence of difference.'}")
print("\n  Formula: CI = x_bar +/- t_(a/2, n-1) x (s / sqrt(n))")


# STEP 5: ASSUMPTION CHECKS

print("\n" + "=" * 60)
print("STEP 5: ASSUMPTION CHECKS")
print("=" * 60)

print("\n[5.1] Shapiro-Wilk Normality Test (H0: data is normally distributed)")

sw_stat_u, sw_p_u = stats.shapiro(uefa)
sw_stat_n, sw_p_n = stats.shapiro(non_uefa)

print(f"\n  UEFA     : W = {sw_stat_u:.4f},  p = {sw_p_u:.4f}  =>  {'Normal (fail to reject H0)' if sw_p_u > 0.05 else 'Non-normal (reject H0)'}")
print(f"  Non-UEFA : W = {sw_stat_n:.4f},  p = {sw_p_n:.4f}  =>  {'Normal (fail to reject H0)' if sw_p_n > 0.05 else 'Non-normal (reject H0)'}")
print("\n  Rule: p > 0.05 => normal;  p <= 0.05 => departure from normality")
print("  Note: The t-test is robust to mild departures from normality (Central Limit Theorem).")

fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4))
fig2.suptitle('Q-Q Plots: Save Percentage\nUEFA vs Non-UEFA : FIFA World Cup 2026',
              fontsize=12, fontweight='bold')

stats.probplot(uefa,     dist="norm", plot=axes2[0])
axes2[0].set_title('UEFA Goalkeepers')
axes2[0].get_lines()[1].set(color='red', linewidth=1.5)

stats.probplot(non_uefa, dist="norm", plot=axes2[1])
axes2[1].set_title('Non-UEFA Goalkeepers')
axes2[1].get_lines()[1].set(color='red', linewidth=1.5)

fig2.tight_layout()
fig2.savefig('task3_qqplots.png', dpi=150, bbox_inches='tight')
print("\n[Q-Q plots saved as task3_qqplots.png]")

print("\n[5.2] Levene's Test for Equality of Variances (H0: variance UEFA = variance Non-UEFA)")

lev_stat, lev_p = stats.levene(uefa, non_uefa)
print(f"\n  Levene statistic : {lev_stat:.4f}")
print(f"  p-value          : {lev_p:.4f}")

equal_var = lev_p > 0.05
if equal_var:
    print("  Decision: p > 0.05 => Fail to reject H0. Variances are EQUAL.")
    print("            Use equal_var=True (Student's t-test).")
else:
    print("  Decision: p <= 0.05 => Reject H0. Variances are UNEQUAL.")
    print("            Use equal_var=False (Welch's t-test).")


# STEP 6: TWO-SAMPLE t-TEST

print("\n" + "=" * 60)
print("STEP 6: TWO-SAMPLE t-TEST")
print("=" * 60)

print("\n  H0 (Null)       : mean UEFA = mean Non-UEFA")
print("  H1 (Alternative): mean UEFA != mean Non-UEFA")
print("  Test type       : Two-tailed, two-sample t-test")
print("  Alpha           : 0.05\n")

t_stat, p_value = stats.ttest_ind(uefa, non_uefa, equal_var=equal_var)

print(f"  t-statistic : {t_stat:.4f}")
print(f"  p-value     : {p_value:.4f}")
print(f"  equal_var   : {equal_var}")

print("\n  Result:")
if p_value < 0.05:
    print("  => p < 0.05: REJECT H0.")
    print("     There IS a statistically significant difference in mean save")
    print("     percentage between UEFA and non-UEFA goalkeepers.")
else:
    print("  => p >= 0.05: FAIL TO REJECT H0.")
    print("     There is NO statistically significant difference in mean save")
    print("     percentage between UEFA and non-UEFA goalkeepers.")

effect_size = abs(mean_uefa - mean_non_uefa) / np.sqrt(
    ((len(uefa)-1)*uefa.std(ddof=1)**2 + (len(non_uefa)-1)*non_uefa.std(ddof=1)**2)
    / (len(uefa) + len(non_uefa) - 2))
print(f"\n  Cohen's d (effect size) : {effect_size:.4f}")
if   effect_size < 0.2: effect_label = "negligible"
elif effect_size < 0.5: effect_label = "small"
elif effect_size < 0.8: effect_label = "medium"
else:                   effect_label = "large"
print(f"  Magnitude               : {effect_label}")


# STEP 7: VISUALISATIONS

print("\n" + "=" * 60)
print("STEP 7: VISUALISATIONS")
print("=" * 60)

fig3, ax3 = plt.subplots(figsize=(8, 6))

bp = ax3.boxplot([uefa, non_uefa],
                 patch_artist=True,
                 tick_labels=['UEFA', 'Non-UEFA'],
                 widths=0.4,
                 medianprops=dict(color='black', linewidth=2))

colors = ['#1f77b4', '#ff7f0e']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

np.random.seed(42)
for i, (group, color) in enumerate(zip([uefa, non_uefa], colors), start=1):
    jitter = np.random.normal(0, 0.04, size=len(group))
    ax3.scatter(np.full(len(group), i) + jitter, group,
                alpha=0.55, color=color, s=30, zorder=3)

ax3.axhline(df_clean['Save percentage'].mean(), color='gray',
            linestyle=':', linewidth=1.5,
            label=f'Overall mean ({df_clean["Save percentage"].mean():.1f}%)')

ax3.set_title('Save Percentage by Confederation\nFIFA World Cup 2026', fontsize=13, fontweight='bold')
ax3.set_ylabel('Save Percentage (%)', fontsize=12)
ax3.set_xlabel('Confederation', fontsize=12)
ax3.legend(fontsize=10)
ax3.grid(axis='y', linestyle='--', alpha=0.4)

fig3.tight_layout()
fig3.savefig('task3_boxplot.png', dpi=150, bbox_inches='tight')
print("[Box plot saved as task3_boxplot.png]")

fig4, ax4 = plt.subplots(figsize=(9, 6))

color_map = {'UEFA': '#1f77b4', 'Non-UEFA': '#ff7f0e'}
for label, grp in df_clean.groupby(df_clean['Confederation'].apply(
        lambda x: 'UEFA' if x == 'UEFA' else 'Non-UEFA')):
    ax4.scatter(grp['Shots on target faced'], grp['Saves'],
                label=label, color=color_map[label],
                s=60, alpha=0.75, edgecolors='white', linewidth=0.5)

max_shots = df_clean['Shots on target faced'].max() * 1.05
xs = np.linspace(0, max_shots, 200)
ax4.plot(xs, xs,        color='green', linestyle='--', linewidth=1, label='100% save rate')
ax4.plot(xs, xs * 0.75, color='gray',  linestyle=':',  linewidth=1, label='75% save rate')

ax4.set_title('Saves vs Shots on Target Faced\nColoured by Confederation : FIFA World Cup 2026',
              fontsize=13, fontweight='bold')
ax4.set_xlabel('Shots on Target Faced', fontsize=12)
ax4.set_ylabel('Saves', fontsize=12)
ax4.legend(fontsize=10)
ax4.grid(linestyle='--', alpha=0.35)

fig4.tight_layout()
fig4.savefig('task3_scatter.png', dpi=150, bbox_inches='tight')
print("[Scatter plot saved as task3_scatter.png]")


# STEP 8: RESULTS SUMMARY TABLE

print("\n" + "=" * 60)
print("STEP 8: RESULTS SUMMARY TABLE")
print("=" * 60)

summary = pd.DataFrame({
    'Metric': ['n', 'Mean (%)', 'Median (%)', 'Std Dev (%)', 'Min (%)', 'Max (%)',
               '95% CI Lower', '95% CI Upper', 'Shapiro-Wilk p', 'Levene p',
               't-statistic', 'p-value (t-test)', "Cohen's d"],
    'UEFA': [
        len(uefa), f"{uefa.mean():.2f}", f"{uefa.median():.2f}",
        f"{uefa.std(ddof=1):.2f}", f"{uefa.min():.2f}", f"{uefa.max():.2f}",
        f"{ci_uefa_lo:.2f}", f"{ci_uefa_up:.2f}", f"{sw_p_u:.4f}",
        f"{lev_p:.4f}", f"{t_stat:.4f}", f"{p_value:.4f}", f"{effect_size:.4f}"
    ],
    'Non-UEFA': [
        len(non_uefa), f"{non_uefa.mean():.2f}", f"{non_uefa.median():.2f}",
        f"{non_uefa.std(ddof=1):.2f}", f"{non_uefa.min():.2f}", f"{non_uefa.max():.2f}",
        f"{ci_non_uefa_lo:.2f}", f"{ci_non_uefa_up:.2f}", f"{sw_p_n:.4f}",
        "N/A", "N/A", "N/A", "N/A"
    ]
})

print("\n" + summary.to_string(index=False))

print("\n" + "=" * 60)
print("Analysis complete. Output files generated:")
print("  task3_histogram.png")
print("  task3_qqplots.png")
print("  task3_boxplot.png")
print("  task3_scatter.png")
print("=" * 60)

plt.show()