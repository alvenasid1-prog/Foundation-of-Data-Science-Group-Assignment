# HIT140 — Objective 1, Analytic Task 2

**Crossing Output of Midfielders at the FIFA World Cup 2026**

Author: **Nasid Alve** · Student ID: **S396257**
Unit: HIT140 Foundations of Data Science · Charles Darwin University
Assessment 2 — Group Project (Objective 1, Task 2 of 4)

---

## Analytic question

> Do midfielders deliver more crosses per 90 minutes than the tournament-wide
> average for all outfield players?

**Focal point: chance creation from wide areas.** This is distinct from the
other three tasks in the project, which examine team discipline (Task 1),
goalkeeping (Task 3) and squad age (Task 4). Crossing is an attacking action
measuring how a player supplies the penalty area.

Passing performance was the original plan, but FBref did not publish passing
data for this tournament — the `Cmp`, `Att` and `Cmp%` columns are empty for
every player. Crossing volume is recorded in full under Miscellaneous Stats, so
the task was moved to that variable while keeping the same statistical design.

---

## Why crosses **per 90 minutes**, not raw counts

This is the central methodological decision.

A player who appeared in seven matches has far more opportunity to cross than
one who played two. Raw counts would therefore measure playing time rather than
crossing output. Dividing each player's crosses by their 90-minute periods
played converts the count into a **rate**, so every player is compared on equal
terms.

---

## Method

| Step | Approach |
|---|---|
| Wrangling | Goalkeepers removed (they essentially never cross and would drag the outfield benchmark down); players under 2 × 90 minutes removed (a single cross in a brief appearance produces an extreme, meaningless rate) |
| Population | All midfielders meeting the playing-time threshold — 217 players |
| Sample | Simple random sample of 50, `random_state=42` for reproducibility |
| Benchmark | Mean crosses per 90 across all outfield players — a census, therefore a known population parameter rather than an estimate |
| Assumption check | Shapiro–Wilk test plus a Q–Q plot |
| Interval | 95% confidence interval using the *t* distribution (population σ unknown) |
| Test | One-sample, **one-tailed** *t*-test (`alternative='greater'`), matching the directional question |
| Robustness | Wilcoxon signed-rank test, plus a second benchmark excluding midfielders |

Hybrid positions such as `DF,MF` and `FWMF` were assigned by first-listed
position, so a player counts as a midfielder only if `MF` appears first.

---

## Results

| Statistic | Value |
|---|---|
| Outfield pool after filtering | 492 players |
| Population (midfielders) | 217 |
| Sample size | 50 (23.0%) |
| Sample mean | 1.874 crosses per 90 |
| Median | 0.964 |
| Standard deviation | 2.217 |
| Min / Max | 0.000 / 10.227 |
| Skewness | 1.692 (strong right skew) |
| Shapiro–Wilk *p* | 0.0000 — normality not met |
| 95% CI | [1.244, 2.504] |
| Benchmark | 1.522 |
| *t* statistic | 1.1237 |
| *p*-value | 0.1333 (one-tailed) |
| Cohen's *d* | 0.159 (negligible) |
| Wilcoxon check | *p* = 0.6084 (agrees with the *t*-test) |
| **Decision** | **Fail to reject H₀** at α = 0.05 |

### Interpretation

There is insufficient evidence that midfielders deliver more crosses per 90
minutes than the outfield average. The difference of +0.352 crosses per 90 is
negligible in effect-size terms, and the non-parametric Wilcoxon test reaches
the same conclusion.

Normality was not met (*p* < 0.0001), which is expected here: crossing rates
are counts bounded below by zero, and many central midfielders never cross at
all. The mean of 1.874 sits well above the median of 0.964, confirming that a
small number of high-volume wide players pull the average upward. With n = 50
the Central Limit Theorem still supports the *t*-test, and the Wilcoxon result
confirms the finding independently of that assumption.

### Sensitivity check

Against a benchmark of defenders and forwards only (1.194), the result reverses:
*t* = 2.1705, *p* = 0.0174, which **rejects** H₀.

The two tests disagree because midfielders make up nearly half the outfield
pool and cross more often than average, so they inflate the very benchmark they
are being tested against. Midfielders do cross more than defenders and
forwards, but not more than an average that already includes them.

The reported conclusion remains **fail to reject**, since that is the test
matching the analytic question as posed. The sensitivity check is reported for
transparency, not substituted for the primary result.

---

## Limitations

1. **Benchmark overlap.** The tournament-wide average includes midfielders, so
   it is pulled toward the sample, making the test conservative. The
   sensitivity check quantifies exactly how much this matters here.
2. **"Midfielder" is not one role.** Wide midfielders cross constantly while
   central midfielders rarely do, yet FBref labels both `MF`. Crossing is
   role-dependent rather than position-dependent, which is the likeliest reason
   for the extreme skew. Full-backs, classified as defenders, are among the
   highest-volume crossers in the tournament.
3. **Volume is not quality.** The data records crosses attempted, not crosses
   completed or chances created, so a wasteful crosser scores the same as an
   accurate one.
4. **Coarse position labels.** Hybrids were resolved by first-listed position;
   a different rule would shift the population slightly.

---

## Files

| File | Contents |
|---|---|
| `task2_crosses_analysis.py` | Full analysis script |
| `S396257_task2_crosses_analysis.ipynb` | Notebook used to run the analysis |
| `wc2026_misc.csv` | Raw data exported from FBref (1039 players) |
| `cleaned_data.csv` | Filtered outfield pool after wrangling (492 players) |
| `sample_used.csv` | The exact 50 midfielders analysed |
| `results_summary.txt` | All results, generated by the script |
| `task2_crosses_analysis.png` | Histogram, Q–Q plot, boxplot, top teams |

## Running it

```bash
pip install pandas numpy scipy matplotlib
python task2_crosses_analysis.py
```

The script expects `wc2026_misc.csv` in the same folder. Because the random
seed is fixed at 42, re-running reproduces every figure above exactly.

---

## Data source

FBref.com (Sports Reference LLC), *2026 FIFA World Cup Miscellaneous Stats*.
Retrieved September 2026 from
`https://fbref.com/en/comps/1/misc/World-Cup-Stats`

## Declaration

Generative AI (Claude) was used to review the statistical design, generate and
comment the analysis script, and draft this documentation. Data collection,
execution of the analysis, interpretation of the results, and the recorded
presentation are my own work. Full details are recorded in the team's AI Usage
Declaration Form.
