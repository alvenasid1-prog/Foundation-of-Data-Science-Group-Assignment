# HIT140 — Objective 1, Analytic Task 4

## Shooting Conversion Rate of Forwards vs Midfielders at the FIFA World Cup 2026

**Author:** Nahidul Islam <br>
**Student ID:** S396122 <br>
**Unit:** HIT140 — Foundations of Data Science <br>
**Institution:** Charles Darwin University

---

## 1. Research Question

> **Is the mean individual shooting conversion rate significantly different between forwards and midfielders at the 2026 FIFA World Cup?**

Conversion rate is defined as:

$$
\text{Conversion Rate} = \frac{\text{Goals Scored}}{\text{Shots Taken}}
$$

The study population consists of players classified exclusively as `FW` or `MF` who recorded at least **3 shots**. Hybrid-position players were excluded.

---

## 2. Method

A simple random sample of **40 forwards and 40 midfielders** was selected using `random_state=42`.

The primary analysis uses an independent **two-sample t-test** comparing individual player conversion rates. Descriptive statistics, Shapiro-Wilk normality testing, Levene's test, a 95% confidence interval, and Cohen's \(d\) were also calculated.

A **Mann-Whitney U test** was used as a robustness check, and the analysis was repeated using a stricter threshold of **5 or more shots**.

A pooled two-proportion z-test and Fisher's exact test were additionally conducted as alternative analyses.

---

## 3. Results

| Measure              | Forwards | Midfielders |
| -------------------- | -------: | ----------: |
| n                    |       40 |          40 |
| Mean conversion rate |   0.1561 |      0.1047 |
| Median               |   0.1667 |      0.0000 |
| Std. deviation       |   0.1435 |      0.1383 |

### Primary t-test

* Shapiro-Wilk: \(p < 0.001\) for both groups
* Levene's test: \(p = 0.3982\)
* \(t = 1.6291\)
* \(p = 0.1073\)
* 95% CI: \([-0.0114, 0.1141]\)
* Cohen's \(d = 0.3643\)

**Decision:** Fail to reject \(H_0\) at \(\alpha=0.05\).

The Mann-Whitney U test also produced a non-significant result (\(p=0.0953\)). The sensitivity analysis using \(Sh \geq 5\) likewise remained non-significant (\(p=0.1243\)).

The pooled analysis produced a significant difference (\(z=2.4993,\ p=0.0124\)), illustrating the distinction between the **average player's conversion rate** and the **overall group conversion rate**.

---

## 4. Conclusion

The required individual-level t-test provides **insufficient evidence of a statistically significant difference** in mean shooting conversion rate between forwards and midfielders.

Although forwards had a higher observed mean conversion rate, the difference did not reach statistical significance at the 5% level.

---

## 5. Limitations

* No shot-quality, shot-location, or xG information was available.
* Hybrid-position players were excluded.
* Conversion rates were non-normally distributed.
* Results may vary with a different random sample.
* Individual and pooled conversion rates represent different analytical measures.

---

## 6. How to Run

Create and activate a virtual environment, then install the required dependencies:

```bash
python -m venv .venv
```

**Windows:**

```bash
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the analysis:

```bash
python task4_conversion_rate.py
```

The analysis uses `random_state=42` to ensure reproducibility.
