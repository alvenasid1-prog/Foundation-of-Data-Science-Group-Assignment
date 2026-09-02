# Task 1 — Discipline (Fouls and Cards)

**Author:** Jahid Hasan
**Unit:** HIT140 Foundations of Data Science — Assessment 2 (Group Project)
**Topic:** FIFA World Cup 2026

## Research Question

Do teams that advanced to the knockout stage commit fewer fouls per match on
average than teams eliminated in the group stage?

## Data

- **Source:** [FBref.com](https://fbref.com/en/comps/1/World-Cup-Stats) —
  Squad Standard Stats & Squad Miscellaneous Stats, 2026 FIFA World Cup
- **Stage classification** (Knockout vs. Eliminated) verified against
  Wikipedia: *2026 FIFA World Cup round of 32*
- **File:** `task1_discipline_data.csv` (48 teams: 32 Knockout, 16 Eliminated
  in the group stage)

| Column | Description |
|---|---|
| Squad | Team name |
| MP (Matches Played) | Matches played in the tournament |
| CrdY / CrdR | Yellow / Red cards |
| Fls (Fouls Committed) | Total fouls committed |
| Fouls per Match | Fls ÷ MP |
| Stage | `Knockout` or `Eliminated (Group Stage)` |

## Method

1. **Data wrangling** — load and validate the dataset, split into two groups
   by `Stage`.
2. **Sampling** — Simple Random Sample: 20 Knockout teams, 12 Eliminated
   teams (`random_state=42` for reproducibility).
3. **Descriptive statistics** — mean, median, standard deviation, range for
   each group.
4. **95% Confidence Interval** for the mean fouls per match in each group.
5. **Assumption checks** — Shapiro-Wilk (normality), Levene's test (equal
   variance).
6. **Two-sample t-test** comparing the two groups, plus Cohen's d effect
   size.

## Results

| Group | n | Mean | 95% CI |
|---|---|---|---|
| Knockout | 20 | 11.55 | (10.44, 12.66) |
| Eliminated | 12 | 12.36 | (10.61, 14.11) |

- Shapiro-Wilk: both groups approximately normal (p > 0.05)
- Levene's test: equal variances assumed (p = 0.734)
- Two-sample t-test: t = -0.878, **p = 0.387**
- Cohen's d = -0.32 (small effect)

**Conclusion:** No statistically significant difference in fouls per match
between teams that reached the knockout stage and teams eliminated in the
group stage. The data does not support the claim that knockout-stage teams
are more disciplined.

## How to Run

```bash
pip install -r requirements.txt
python task1_discipline_analysis_pro.py
```

Optional: point to a different dataset file:

```bash
python task1_discipline_analysis_pro.py --data path/to/other_file.csv
```

## Files in this Folder

- `task1_discipline_analysis_pro.py` — main analysis script
- `task1_discipline_data.csv` — dataset
- `boxplot_fouls_per_match.png` / `histogram_fouls_per_match.png` — generated
  charts
- `requirements.txt` — Python package dependencies
