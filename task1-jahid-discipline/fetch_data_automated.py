"""
HIT140 - Assessment 2: Group Project
Task 1: Discipline (Fouls and Cards) - AUTOMATED DATA ACQUISITION

This script replaces manual copy-paste from FBref with a fully automated
Python pipeline: it fetches the required tables directly from FBref's HTML
pages, merges them, computes the derived variable (Fouls per Match), and
scrapes the Wikipedia "Round of 32" page to programmatically classify each
team as Knockout or Eliminated (Group Stage) - instead of a hard-coded list.

Author: Jahid Hasan

Run this once to (re)generate task1_discipline_data.csv, then run
task1_discipline_analysis_pro.py for the statistical analysis.
"""

from __future__ import annotations

import re
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

HEADERS = {
    # FBref blocks requests with no browser-like User-Agent
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

FBREF_STANDARD_URL = "https://fbref.com/en/comps/1/stats/World-Cup-Stats"
FBREF_MISC_URL = "https://fbref.com/en/comps/1/misc/World-Cup-Stats"
WIKI_ROUND_OF_32_URL = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_knockout_stage"

OUTPUT_CSV = "task1_discipline_data.csv"


def fetch_html(url: str) -> str:
    """Download a page's HTML, respecting FBref by pacing requests."""
    response = requests.get(url, headers=HEADERS, timeout=20)
    response.raise_for_status()
    time.sleep(3)  # be polite to FBref's servers between requests
    return response.text


def fetch_fbref_table(url: str, table_id_contains: str) -> pd.DataFrame:
    """
    FBref wraps some tables inside HTML comments (to stop naive scraping).
    This finds the right table whether it's live in the HTML or hidden
    inside a comment block, and flattens the multi-row header FBref uses.
    """
    html = fetch_html(url)

    # FBref hides some tables inside <!-- ... --> comments; search there too
    tables_html = re.findall(r"<!--(.*?)-->", html, flags=re.DOTALL)
    all_html_sources = [html] + tables_html

    for source in all_html_sources:
        try:
            dfs = pd.read_html(StringIO(source))
        except ValueError:
            continue
        for df in dfs:
            # FBref squad tables have a 'Squad' column once headers flatten
            cols = ["_".join([str(c) for c in col]) if isinstance(col, tuple) else str(col)
                    for col in df.columns]
            if any("Squad" in c for c in cols):
                df.columns = cols
                return df

    raise ValueError(f"Could not locate the expected table at {url}")


def clean_squad_name(name: str) -> str:
    """Strip FBref's country-code prefix (e.g. 'ar Argentina' -> 'Argentina')."""
    return re.sub(r"^[a-z]{2,3}\s+", "", str(name)).strip()


def build_dataset() -> pd.DataFrame:
    print("Fetching Squad Standard Stats from FBref...")
    standard = fetch_fbref_table(FBREF_STANDARD_URL, "stats_squads_standard_for")
    standard.columns = [c.split("_")[-1] for c in standard.columns]
    standard = standard[standard["Squad"].notna()]
    standard = standard[standard["Squad"] != "Squad"]  # drop repeated header rows
    standard["Squad"] = standard["Squad"].apply(clean_squad_name)
    standard = standard[["Squad", "MP", "CrdY", "CrdR"]].drop_duplicates("Squad")

    print("Fetching Squad Miscellaneous Stats from FBref...")
    misc = fetch_fbref_table(FBREF_MISC_URL, "stats_squads_misc_for")
    misc.columns = [c.split("_")[-1] for c in misc.columns]
    misc = misc[misc["Squad"].notna()]
    misc = misc[misc["Squad"] != "Squad"]
    misc["Squad"] = misc["Squad"].apply(clean_squad_name)
    misc = misc[["Squad", "Fls"]].drop_duplicates("Squad")

    df = standard.merge(misc, on="Squad", how="inner")
    for col in ["MP", "CrdY", "CrdR", "Fls"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["MP", "Fls"])
    df["Fouls per Match"] = df["Fls"] / df["MP"]

    print(f"Fetched data for {len(df)} teams.")
    return df


def fetch_knockout_teams() -> set[str]:
    """
    Scrape the Wikipedia Round of 32 page and programmatically extract the
    32 team names that reached the knockout stage, instead of hard-coding
    the list by hand.
    """
    print("Fetching Round of 32 team list from Wikipedia...")
    html = fetch_html(WIKI_ROUND_OF_32_URL)
    tables = pd.read_html(StringIO(html))

    teams: set[str] = set()
    # Wikipedia score tables typically have columns like Team1 / Team2 or similar
    for t in tables:
        cols = [str(c) for c in t.columns]
        for col in cols:
            if any(key in col for key in ["Team", "Home", "Away", "Winner"]):
                for val in t[col].dropna():
                    val = re.sub(r"\[.*?\]", "", str(val)).strip()
                    if val and val[0].isupper() and len(val) > 2 and not val.isdigit():
                        teams.add(val)
    return teams


def classify_stage(df: pd.DataFrame, knockout_teams: set[str]) -> pd.DataFrame:
    df = df.copy()
    df["Stage"] = df["Squad"].apply(
        lambda s: "Knockout" if s in knockout_teams else "Eliminated (Group Stage)"
    )
    return df


def main() -> None:
    df = build_dataset()
    knockout_teams = fetch_knockout_teams()

    matched = df["Squad"].isin(knockout_teams).sum()
    print(f"Matched {matched}/{len(df)} teams against the scraped Knockout list.")

    if matched < 28:  # sanity check: expect 32, allow a small margin for name mismatches
        print(
            "[Warning] Fewer knockout teams matched than expected (32). "
            "Team-name formatting on Wikipedia may differ slightly from FBref "
            "(e.g. 'Bosnia and Herzegovina' vs 'Bosnia-Herz'). "
            "Review the output CSV's Stage column manually before analysis."
        )

    df = classify_stage(df, knockout_teams)
    df = df.rename(columns={
        "MP": "MP (Matches Played)",
        "CrdY": "CrdY (Yellow Cards)",
        "CrdR": "CrdR (Red Cards)",
        "Fls": "Fls (Fouls Committed)",
    })
    df = df[[
        "Squad", "MP (Matches Played)", "CrdY (Yellow Cards)", "CrdR (Red Cards)",
        "Fls (Fouls Committed)", "Fouls per Match", "Stage"
    ]]

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved automated dataset to {OUTPUT_CSV}")
    print(df["Stage"].value_counts())


if __name__ == "__main__":
    main()
