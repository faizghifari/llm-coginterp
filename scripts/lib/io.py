"""Load/save the 3 canonical dataset CSVs with one consistent convention.

Every script in this toolkit should go through `load_data()` /
`save_*()` rather than calling pandas directly. That keeps dtype
handling, NaN handling, and line-ending conventions identical across
every tool — and means a future format change (e.g. switching line
endings, adding a column) only has to be made in this one file.
"""
import pandas as pd

from . import config


def load_csv(path):
    """Read a dataset CSV with every column as string and without
    pandas' default NaN-for-empty-string coercion — IDs and scores in
    this dataset must round-trip exactly as written."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def load_data():
    """Load all 3 canonical CSVs. Returns (benchmarks, models, results)."""
    benchmarks = load_csv(config.BENCHMARKS_CSV)
    models = load_csv(config.MODELS_CSV)
    results = load_csv(config.RESULTS_CSV)
    return benchmarks, models, results


def save_csv(df, path):
    """Write a dataset CSV back with CRLF line endings, matching the
    convention already used by every committed file under data/."""
    df.to_csv(path, index=False, lineterminator="\r\n")


def save_benchmarks(df):
    save_csv(df, config.BENCHMARKS_CSV)


def save_models(df):
    save_csv(df, config.MODELS_CSV)


def save_results(df):
    save_csv(df, config.RESULTS_CSV)
