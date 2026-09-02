"""
HIT140 - Assessment 2: Group Project
Task 1: Discipline (Fouls and Cards)
Author: Jahid Hasan

Research Question:
    Do teams that advanced to the knockout stage commit fewer fouls per match
    on average than teams eliminated in the group stage?

Data Source:
    FBref.com - Squad Standard Stats & Squad Miscellaneous Stats,
    2026 FIFA World Cup.
    Stage classification (Knockout / Eliminated) verified against Wikipedia:
    "2026 FIFA World Cup round of 32".

Usage:
    python task1_discipline_analysis.py
    python task1_discipline_analysis.py --data path/to/other_file.csv
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import shapiro, levene


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = [
    "Squad",
    "MP (Matches Played)",
    "Fls (Fouls Committed)",
    "Fouls per Match",
    "Stage",
]
KNOCKOUT_LABEL = "Knockout"
ELIMINATED_LABEL = "Eliminated (Group Stage)"
CONFIDENCE_LEVEL = 0.95
ALPHA = 0.05
KNOCKOUT_SAMPLE_SIZE = 20
ELIMINATED_SAMPLE_SIZE = 12
RANDOM_SEED = 42


@dataclass
class GroupSummary:
    """Holds descriptive + inferential statistics for one group."""
    label: str
    n: int
    mean: float
    median: float
    std: float
    range_: float
    ci_lower: float
    ci_upper: float
    shapiro_p: float


@dataclass
class TTestResult:
    """Holds the outcome of the two-sample t-test."""
    t_stat: float
    p_value: float
    levene_p: float
    equal_var_used: bool
    cohens_d: float
    significant: bool


# ---------------------------------------------------------------------------
# Step 1: Data wrangling
# ---------------------------------------------------------------------------
def load_data(path: str | Path) -> pd.DataFrame:
    """
    Load and validate the discipline dataset.

    Raises
    ------
    FileNotFoundError
        If the CSV file cannot be found at the given path.
    ValueError
        If required columns are missing or the file contains no rows.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find '{path}'. Make sure the CSV is in the same "
            f"folder as this script (or pass --data <path>)."
        )

    df = pd.read_csv(path)

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required column(s): {missing_cols}")

    if df.empty:
        raise ValueError("Dataset is empty.")

    n_missing = df[REQUIRED_COLUMNS].isnull().sum().sum()
    if n_missing > 0:
        print(f"[Warning] {n_missing} missing value(s) found in required columns.")

    return df


def split_by_stage(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Split 'Fouls per Match' into Knockout and Eliminated populations."""
    knockout = df.loc[df["Stage"] == KNOCKOUT_LABEL, "Fouls per Match"]
    eliminated = df.loc[df["Stage"] == ELIMINATED_LABEL, "Fouls per Match"]

    if knockout.empty or eliminated.empty:
        raise ValueError(
            "One of the two Stage groups is empty. Check the 'Stage' column "
            f"contains exactly '{KNOCKOUT_LABEL}' and '{ELIMINATED_LABEL}'."
        )
    return knockout, eliminated


# ---------------------------------------------------------------------------
# Step 2: Sampling
# ---------------------------------------------------------------------------
def draw_samples(
    knockout: pd.Series,
    eliminated: pd.Series,
    n_knockout: int = KNOCKOUT_SAMPLE_SIZE,
    n_eliminated: int = ELIMINATED_SAMPLE_SIZE,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.Series, pd.Series]:
    """
    Draw a Simple Random Sample from each population.

    A fixed random_state is used so the samples (and therefore every
    downstream result) are reproducible.
    """
    n_knockout = min(n_knockout, len(knockout))
    n_eliminated = min(n_eliminated, len(eliminated))

    np.random.seed(seed)
    knockout_sample = knockout.sample(n=n_knockout, random_state=seed)
    eliminated_sample = eliminated.sample(n=n_eliminated, random_state=seed)
    return knockout_sample, eliminated_sample


# ---------------------------------------------------------------------------
# Step 3 & 4: Descriptive statistics + confidence interval
# ---------------------------------------------------------------------------
def summarise_group(label: str, sample: pd.Series) -> GroupSummary:
    """Compute descriptive statistics, a 95% CI, and a normality check."""
    mean = sample.mean()
    sem = stats.sem(sample)
    ci_lower, ci_upper = stats.t.interval(
        CONFIDENCE_LEVEL, df=len(sample) - 1, loc=mean, scale=sem
    )
    normal_p = shapiro(sample).pvalue

    return GroupSummary(
        label=label,
        n=len(sample),
        mean=mean,
        median=sample.median(),
        std=sample.std(),
        range_=sample.max() - sample.min(),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        shapiro_p=normal_p,
    )


def print_summary(summary: GroupSummary) -> None:
    normal_note = "approximately normal" if summary.shapiro_p > ALPHA else "NOT normal"
    print(
        f"{summary.label:<10} | n={summary.n:>2} | "
        f"mean={summary.mean:6.2f} | median={summary.median:6.2f} | "
        f"std={summary.std:5.2f} | range={summary.range_:5.2f} | "
        f"95% CI=({summary.ci_lower:5.2f}, {summary.ci_upper:5.2f}) | "
        f"Shapiro p={summary.shapiro_p:.4f} ({normal_note})"
    )


# ---------------------------------------------------------------------------
# Step 5 & 6: Assumption checks + two-sample t-test
# ---------------------------------------------------------------------------
def cohens_d(sample_a: pd.Series, sample_b: pd.Series) -> float:
    """
    Effect size for two independent samples (pooled standard deviation).
    Interpretation: ~0.2 small, ~0.5 medium, ~0.8 large.
    """
    n1, n2 = len(sample_a), len(sample_b)
    var1, var2 = sample_a.var(ddof=1), sample_b.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (sample_a.mean() - sample_b.mean()) / pooled_std


def run_two_sample_ttest(sample_a: pd.Series, sample_b: pd.Series) -> TTestResult:
    """Check equal-variance assumption, then run the appropriate t-test."""
    levene_p = levene(sample_a, sample_b).pvalue
    equal_var = levene_p > ALPHA

    t_stat, p_value = stats.ttest_ind(sample_a, sample_b, equal_var=equal_var)
    d = cohens_d(sample_a, sample_b)

    return TTestResult(
        t_stat=t_stat,
        p_value=p_value,
        levene_p=levene_p,
        equal_var_used=equal_var,
        cohens_d=d,
        significant=p_value < ALPHA,
    )


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def plot_boxplot(knockout_sample: pd.Series, eliminated_sample: pd.Series, out_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.boxplot(
        [knockout_sample, eliminated_sample],
        tick_labels=["Knockout", "Eliminated"],
    )
    plt.ylabel("Fouls per Match")
    plt.title("Fouls per Match: Knockout vs Eliminated Teams")
    plt.tight_layout()
    plt.savefig(out_dir / "boxplot_fouls_per_match.png", dpi=150)
    plt.close()


def plot_histogram(knockout_sample: pd.Series, eliminated_sample: pd.Series, out_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(knockout_sample, bins=8, alpha=0.6, label="Knockout")
    plt.hist(eliminated_sample, bins=8, alpha=0.6, label="Eliminated")
    plt.xlabel("Fouls per Match")
    plt.ylabel("Frequency")
    plt.title("Distribution of Fouls per Match")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "histogram_fouls_per_match.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(data_path: str = "task1_discipline_data.csv") -> None:
    try:
        df = load_data(data_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[Error] {exc}")
        sys.exit(1)

    print(f"Dataset loaded: {df.shape[0]} teams, {df.shape[1]} columns.\n")

    knockout, eliminated = split_by_stage(df)
    print(f"Population sizes -> Knockout: {len(knockout)} | Eliminated: {len(eliminated)}")

    knockout_sample, eliminated_sample = draw_samples(knockout, eliminated)
    print(
        f"Sample sizes (Simple Random Sampling, seed={RANDOM_SEED}) -> "
        f"Knockout: {len(knockout_sample)} | Eliminated: {len(eliminated_sample)}\n"
    )

    ko_summary = summarise_group("Knockout", knockout_sample)
    el_summary = summarise_group("Eliminated", eliminated_sample)
    print("--- Descriptive Statistics & 95% Confidence Intervals ---")
    print_summary(ko_summary)
    print_summary(el_summary)

    out_dir = Path(".")
    plot_boxplot(knockout_sample, eliminated_sample, out_dir)
    plot_histogram(knockout_sample, eliminated_sample, out_dir)
    print("\nSaved: boxplot_fouls_per_match.png, histogram_fouls_per_match.png")

    result = run_two_sample_ttest(knockout_sample, eliminated_sample)
    print("\n--- Two-Sample T-Test ---")
    print(f"Levene's test p-value: {result.levene_p:.4f} "
          f"(equal variances {'assumed' if result.equal_var_used else 'NOT assumed'})")
    print(f"t-statistic: {result.t_stat:.4f}")
    print(f"p-value: {result.p_value:.4f}")
    print(f"Cohen's d (effect size): {result.cohens_d:.3f}")

    verdict = "Statistically significant difference (reject H0)" if result.significant \
        else "No statistically significant difference (fail to reject H0)"
    print(f"Result: {verdict}")

    print(
        "\nConclusion: "
        f"Knockout teams averaged {ko_summary.mean:.2f} fouls/match "
        f"(95% CI {ko_summary.ci_lower:.2f}-{ko_summary.ci_upper:.2f}); "
        f"Eliminated teams averaged {el_summary.mean:.2f} fouls/match "
        f"(95% CI {el_summary.ci_lower:.2f}-{el_summary.ci_upper:.2f}). "
        f"The two-sample t-test found {'a' if result.significant else 'no'} "
        f"statistically significant difference (p={result.p_value:.3f}), with a "
        f"{'negligible' if abs(result.cohens_d) < 0.2 else 'small' if abs(result.cohens_d) < 0.5 else 'medium' if abs(result.cohens_d) < 0.8 else 'large'} "
        f"effect size (Cohen's d={result.cohens_d:.2f})."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HIT140 Task 1: Discipline analysis")
    parser.add_argument(
        "--data",
        default="task1_discipline_data.csv",
        help="Path to the Task 1 CSV dataset (default: task1_discipline_data.csv)",
    )
    args = parser.parse_args()
    main(args.data)
