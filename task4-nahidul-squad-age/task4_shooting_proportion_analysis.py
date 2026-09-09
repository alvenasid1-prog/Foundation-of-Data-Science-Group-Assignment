"""
# HIT140 — Objective 1, Analytic Task 4
# Shooting Conversion Rate of Forwards vs Midfielders at the FIFA World Cup 2026

**Author:** Nahidul Islam <br>
Student ID: S396122 <br>
**Unit:** HIT140 Foundations of Data Science — Assessment 2 (Group Project) <br>
**Task:** 4 of 4 (Objective 1)

## Analytic Question

> Is the mean individual conversion rate (goals scored per shot taken)
> significantly different between forwards and midfielders at the 2026 FIFA
> World Cup?

**Focal point:** scoring efficiency by attacking role.

### Hypotheses
- **H0:** Mean conversion rate is equal for forwards and midfielders.
- **H1:** Mean conversion rate is significantly different between the two groups.

Tested using a **two-sample t-test** on each player's individual conversion rate.
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)

"""## Step 1: Data Wrangling
Source: `player_shooting.csv` [FBref, 2026 FIFA World Cup Shooting Stats](https://fbref.com/en/comps/1/shooting/World-Cup-Stats#all_stats_shooting).
`Rk` (row index) is dropped — no analytical value.
"""

df = pd.read_csv('player_shooting.csv')
print(f"Raw dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

df = df.drop(columns=['Rk'])

print(f"\nMissing values per column:")
print(df.isnull().sum())

"""## Step 2: Defining the Two Populations
Single-position players only (`FW` or `MF`) — hybrids like `FWMF` excluded.
Minimum 3 shots required, since 1–2 shots gives an unstable rate
(e.g. 1 goal / 1 shot = 100%).
"""

MIN_SHOTS = 3

pop_fw = df[(df['Pos'] == 'FW') & (df['Sh'] >= MIN_SHOTS)].copy()
pop_mf = df[(df['Pos'] == 'MF') & (df['Sh'] >= MIN_SHOTS)].copy()

print(f"Population sizes after filtering:")
print(f"  Forwards   : {len(pop_fw)}")
print(f"  Midfielders: {len(pop_mf)}")

"""## Step 3: Sampling
Simple random sample of 40 per group, fixed seed for reproducibility.

n = 40 exceeds the n ≥ 30 CLT rule of thumb — relevant later, since
normality fails for both groups.

Sampling fraction differs by group (81 vs 161 population), since equal
sample *size* was prioritised over equal sample *fraction*, to keep the
two-sample comparison balanced.
"""

SAMPLE_SIZE = 40

sample_fw = pop_fw.sample(n=min(SAMPLE_SIZE, len(pop_fw)), random_state=SEED)
sample_mf = pop_mf.sample(n=min(SAMPLE_SIZE, len(pop_mf)), random_state=SEED)

sample_fw['conv'] = sample_fw['Gls'] / sample_fw['Sh']
sample_mf['conv'] = sample_mf['Gls'] / sample_mf['Sh']

print(f"Sample sizes (simple random sample, seed={SEED}):")
print(f"  Forwards   : {len(sample_fw)} of {len(pop_fw)} population ({len(sample_fw)/len(pop_fw)*100:.1f}%)")
print(f"  Midfielders: {len(sample_mf)} of {len(pop_mf)} population ({len(sample_mf)/len(pop_mf)*100:.1f}%)")

"""## Step 4: Descriptive Statistics
Includes skewness and IQR alongside mean/std, since conversion rate is
bounded and zero-inflated (many players score no goals despite shooting).
"""

def extended_describe(series, label):
    return pd.Series({
        'n': len(series),
        'mean': series.mean(),
        'median': series.median(),
        'std': series.std(),
        'Q1': series.quantile(0.25),
        'Q3': series.quantile(0.75),
        'IQR': series.quantile(0.75) - series.quantile(0.25),
        'min': series.min(),
        'max': series.max(),
        'skewness': series.skew(),
        'zero_conversion_pct': (series == 0).mean() * 100
    }, name=label)

desc = pd.concat([
    extended_describe(sample_fw['conv'], 'Forwards'),
    extended_describe(sample_mf['conv'], 'Midfielders')
], axis=1)
print(desc.round(4))

"""Both groups are positively skewed (FW: 0.28, MF: 0.88). For midfielders
the median is 0 while the mean is 0.105 — over half (57.5%) scored zero
goals despite shooting, versus 37.5% for forwards. This zero-inflation
explains the non-normal distribution confirmed in Step 5.

## Visualisation
"""

fig, ax = plt.subplots(figsize=(7, 5))
bp = ax.boxplot([sample_fw['conv'], sample_mf['conv']],
                 tick_labels=['Forwards', 'Midfielders'], patch_artist=True)
colors = ['#C0392B', '#2874A6']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax.set_ylabel('Individual Conversion Rate (Goals / Shots)')
ax.set_title('Shooting Efficiency by Position: 2026 FIFA World Cup')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig('task4_boxplot.png', dpi=150, bbox_inches='tight')
plt.show()

"""## Step 5: Checking t-test Assumptions
Normality (Shapiro-Wilk) and equal variance (Levene's), to decide between
Student's and Welch's t-test.
"""

sw_fw = stats.shapiro(sample_fw['conv'])
sw_mf = stats.shapiro(sample_mf['conv'])
print(f"Shapiro-Wilk Forwards:    statistic={sw_fw.statistic:.4f}, p={sw_fw.pvalue:.6f}")
print(f"Shapiro-Wilk Midfielders: statistic={sw_mf.statistic:.4f}, p={sw_mf.pvalue:.6f}")

levene_result = stats.levene(sample_fw['conv'], sample_mf['conv'])
print(f"\nLevene's test: statistic={levene_result.statistic:.4f}, p={levene_result.pvalue:.4f}")

"""Both groups fail normality (p < 0.001) — expected, given the
zero-inflated distribution. With n = 40 per group, the CLT supports the
t-test regardless; a Mann-Whitney U check follows as a robustness check.
Levene's test found no significant evidence of unequal variances
(p = 0.3982), so Student's t-test is used.

## Step 6: Confidence Interval for the Difference in Means
"""

n1, n2 = len(sample_fw), len(sample_mf)
mean1, mean2 = sample_fw['conv'].mean(), sample_mf['conv'].mean()
equal_var = levene_result.pvalue > 0.05

if equal_var:
    pooled_var = ((n1-1)*sample_fw['conv'].var(ddof=1) + (n2-1)*sample_mf['conv'].var(ddof=1)) / (n1+n2-2)
    se_diff = np.sqrt(pooled_var * (1/n1 + 1/n2))
    dof = n1 + n2 - 2
else:
    v1, v2 = sample_fw['conv'].var(ddof=1), sample_mf['conv'].var(ddof=1)
    se_diff = np.sqrt(v1/n1 + v2/n2)
    dof = (v1/n1 + v2/n2)**2 / ((v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1))

diff = mean1 - mean2
t_crit = stats.t.ppf(0.975, dof)
ci_low, ci_high = diff - t_crit*se_diff, diff + t_crit*se_diff

print(f"Mean conversion rate — Forwards: {mean1:.4f}, Midfielders: {mean2:.4f}")
print(f"Difference in means: {diff:.4f}")
print(f"95% CI for the difference: [{ci_low:.4f}, {ci_high:.4f}] (df={dof:.2f})")

"""## Step 7: Hypothesis Test — Two-Sample t-test"""

t_stat, p_value = stats.ttest_ind(sample_fw['conv'], sample_mf['conv'], equal_var=equal_var)

def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na-1)*a.var(ddof=1) + (nb-1)*b.var(ddof=1)) / (na+nb-2))
    return (a.mean() - b.mean()) / pooled_std

d = cohens_d(sample_fw['conv'], sample_mf['conv'])

print(f"t-statistic = {t_stat:.4f}")
print(f"p-value = {p_value:.4f}")
print(f"Equal variances assumed: {equal_var}")
print(f"Cohen's d = {d:.4f}")

"""Cohen's d = 0.36 sits between the small (0.2) and medium (0.5)
thresholds — a small-to-moderate effect, worth revisiting with a larger
sample despite non-significance.

## Step 8: Robustness Check — Mann-Whitney U Test
Non-parametric check, since normality was not satisfied.
"""

u_stat, mw_pval = stats.mannwhitneyu(sample_fw['conv'], sample_mf['conv'], alternative='two-sided')
print(f"Mann-Whitney U = {u_stat:.1f}, p-value = {mw_pval:.4f}")

"""## Step 9: Sensitivity Check
Repeats the analysis at a stricter shot threshold (`Sh >= 5`).
"""

pop_fw5 = df[(df['Pos'] == 'FW') & (df['Sh'] >= 5)].copy()
pop_mf5 = df[(df['Pos'] == 'MF') & (df['Sh'] >= 5)].copy()

sample_fw5 = pop_fw5.sample(n=min(SAMPLE_SIZE, len(pop_fw5)), random_state=SEED)
sample_mf5 = pop_mf5.sample(n=min(SAMPLE_SIZE, len(pop_mf5)), random_state=SEED)

sample_fw5['conv'] = sample_fw5['Gls'] / sample_fw5['Sh']
sample_mf5['conv'] = sample_mf5['Gls'] / sample_mf5['Sh']

lev5 = stats.levene(sample_fw5['conv'], sample_mf5['conv'])
t5, p5 = stats.ttest_ind(sample_fw5['conv'], sample_mf5['conv'], equal_var=(lev5.pvalue > 0.05))

print(f"Stricter threshold (Sh >= 5): n={len(sample_fw5)}/{len(sample_mf5)}")
print(f"t = {t5:.4f}, p = {p5:.4f}")
print(f"Conclusion {'UNCHANGED' if (p5 < 0.05) == (p_value < 0.05) else 'CHANGED'} under the stricter threshold.")

"""## Step 10: Alternative Approach Explored — Pooled Proportion Test
Pools goals/shots per group instead of averaging individual rates —
weights players by shot volume rather than treating everyone equally.
"""

goals_fw, shots_fw = int(sample_fw['Gls'].sum()), int(sample_fw['Sh'].sum())
goals_mf, shots_mf = int(sample_mf['Gls'].sum()), int(sample_mf['Sh'].sum())
p_fw, p_mf = goals_fw/shots_fw, goals_mf/shots_mf

p_pool = (goals_fw + goals_mf) / (shots_fw + shots_mf)
se_pool = np.sqrt(p_pool*(1-p_pool)*(1/shots_fw + 1/shots_mf))
z_stat = (p_fw - p_mf) / se_pool
z_pval = 2 * (1 - stats.norm.cdf(abs(z_stat)))

print(f"Pooled proportion — Forwards: {goals_fw}/{shots_fw} = {p_fw:.4f}")
print(f"Pooled proportion — Midfielders: {goals_mf}/{shots_mf} = {p_mf:.4f}")
print(f"Two-proportion z-test: z = {z_stat:.4f}, p = {z_pval:.4f}")

# Fisher's exact test — further confirmation of the pooled result
table = [[goals_fw, shots_fw - goals_fw], [goals_mf, shots_mf - goals_mf]]
odds_ratio, fisher_p = stats.fisher_exact(table)
print(f"Fisher's exact test: odds ratio = {odds_ratio:.4f}, p = {fisher_p:.4f}")


"""## Interpretation

**Primary result (t-test, individual mean conversion rate):** Forwards
averaged 0.156 goals/shot, midfielders 0.105. t = 1.6291, p = 0.1073 →
**fail to reject H0**. Mann-Whitney U agrees on non-significance
(p = 0.0953) — an independent confirmation, not literal agreement, since
it tests distributional tendency rather than means. Holds under the
stricter Sh ≥ 5 threshold (p = 0.1243). Cohen's d = 0.3643 — small-to-moderate
effect, plausibly detectable with a larger sample.

**Alternative approach (pooled proportion):** Forwards convert 17.99% of
pooled shots vs. 10.59% for midfielders — significant (z = 2.4993,
p = 0.012), confirmed by Fisher's exact test (p = 0.0134).

**Why they disagree:** the t-test weights every player equally regardless
of shot volume; the pooled proportion weights by volume, so high-output
forwards pull the pooled result up. Neither is wrong — they answer
different questions about "conversion rate."

**Conclusion:** No significant difference in mean individual conversion
rate (required t-test). Significant difference in pooled tournament-wide
conversion rate (alternative approach). Both are reported, since they
capture genuinely different things.

**Limitations:**
1. No shot-quality (xG) data — conversion rate is a proxy, not a direct measure.
2. Hybrid-position players excluded, potentially removing relevant attackers.
3. Normality not met for the t-test, though CLT and the Mann-Whitney check support its use.
4. Sampling 40 per group (not the full population) introduces variability, though the sensitivity check suggests the conclusion is stable.
"""