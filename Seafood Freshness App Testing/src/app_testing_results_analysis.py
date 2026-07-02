"""
app_testing_results_analysis.py

End-to-end pipeline for:
- Fetching feedback data from a PostgreSQL database
- Cleaning and enriching the data
- Filtering by date/species/user
- Computing confusion matrices and classification metrics
- Plotting confusion matrices

Designed to be run as a standalone script or imported as a module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extensions import connection as PGConnection
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class DBConfig:
    """
    Configuration for connecting to the PostgreSQL database.
    """

    dbname: str = "postgres"
    user: str = "mobileapi2"
    password: str = "mobileapi2"
    host: str = "mobileapi2.c5gmcqq8k94k.ap-south-1.rds.amazonaws.com"
    port: str = "5432"


# =============================================================================
# Database utilities
# =============================================================================


def get_pg_connection(config: DBConfig) -> PGConnection:
    """
    Create and return a PostgreSQL connection using the given configuration.
    """
    conn = psycopg2.connect(
        dbname=config.dbname,
        user=config.user,
        password=config.password,
        host=config.host,
        port=config.port,
    )
    return conn


def load_feedback_table(config: DBConfig) -> pd.DataFrame:
    """
    Load the entire 'Feedback' table from PostgreSQL into a pandas DataFrame.
    """
    conn = get_pg_connection(config)
    try:
        query = 'SELECT * FROM "Feedback"'
        df = pd.read_sql(query, conn)
    finally:
        conn.close()
    return df


# =============================================================================
# URL parsing & Data enrichment
# =============================================================================


def parse_input_image_url(url: str) -> Tuple[str, str, str, str]:
    """
    Parse an input_image_url and return (date, species, filename, filename_id).

    Assumptions (matching current S3 naming logic):
    - Date is the penultimate segment in the URL path.
    - Filename is the basename of the URL path.
    - Species is the penultimate underscore-separated token in the filename.
    - Filename ID is the third underscore-separated token in the filename.

    Example
    -------
    url = ".../2023-11-23/2023-11-23_09:41:35_(2184)_sardine_input.jpeg"

    Returns
    -------
    date : str          -> "2023-11-23"
    species : str       -> "sardine"
    filename : str      -> "2023-11-23_09:41:35_(2184)_sardine_input.jpeg"
    filename_id : str   -> "(2184)"
    """
    # Date from penultimate path segment
    date = url.split("/")[-2]

    # Filename from URL
    parsed_url = urlparse(url)
    object_key = parsed_url.path[1:]  # strip leading "/"
    filename = os.path.basename(object_key)

    # Species from filename (penultimate underscore token)
    parts = filename.split("_")
    species = parts[-2] if len(parts) >= 2 else ""

    # Filename ID from filename (3rd underscore token)
    filename_id = parts[2] if len(parts) >= 3 else ""

    return date, species, filename, filename_id


def enrich_feedback_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich the Feedback DataFrame with additional derived columns.

    Adds:
    - Date
    - Species
    - filename
    - filename_id

    Also removes:
    - Rows where user_name equals 'test'
    - Rows where user_name is null
    - Duplicate filename_id rows (keeps first)
    """
    df = df.copy()

    parsed = df["input_image_url"].apply(parse_input_image_url)
    df["Date"] = parsed.apply(lambda x: x[0])
    df["Species"] = parsed.apply(lambda x: x[1])
    df["filename"] = parsed.apply(lambda x: x[2])
    df["filename_id"] = parsed.apply(lambda x: x[3])

    df = df[df["user_name"] != "test"]
    df = df.dropna(subset=["user_name"])
    df = df.drop_duplicates(subset="filename_id", keep="first")

    return df


# =============================================================================
# Generic filtering utilities
# =============================================================================


def filter_df(
    df: pd.DataFrame, filters: Optional[Dict[str, Iterable]] = None
) -> pd.DataFrame:
    """
    Filter a pandas DataFrame based on specified conditions for each column.
    """
    if not filters:
        return df

    filtered = df.copy()
    for col, vals in filters.items():
        if not isinstance(vals, (list, tuple, set, np.ndarray)):
            vals = [vals]
        filtered = filtered[filtered[col].isin(vals)]

    return filtered

def get_unique_values(df: pd.DataFrame, column_names: Sequence[str]) -> Dict[str, List]:
    """
    Get unique values for a list of column names in a DataFrame.
    """
    unique_values_dict: Dict[str, List] = {}
    for col in column_names:
        unique_values_dict[col] = df[col].unique().tolist()
    return unique_values_dict


# =============================================================================
# Metrics & confusion matrix utilities
# =============================================================================


def build_confusion_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Construct a confusion matrix for Good/Bad classification across fish1/2/3.
    """
    actual_values = df[
        ["fish1_actual", "fish2_actual", "fish3_actual"]
    ].values.flatten()
    predicted_values = df[["fish1_pred", "fish2_pred", "fish3_pred"]].values.flatten()

    actual_values = actual_values[actual_values != None]  # noqa: E711
    predicted_values = predicted_values[predicted_values != None]  # noqa: E711

    labels = ["G", "B"]
    cm = confusion_matrix(actual_values, predicted_values, labels=labels)
    return cm


def pretty_print_confusion_matrix(cm: np.ndarray, species: str) -> None:
    """
    Print a human-readable confusion matrix with labels.
    """
    conf_df = pd.DataFrame(
        cm,
        index=["Actual Good", "Actual Bad"],
        columns=["Predicted Good", "Predicted Bad"],
    )
    print("\n" + "\t" * 3, f"{species} results\n")
    print(conf_df)


def print_classification_report_from_cm(cm: np.ndarray) -> None:
    """
    Compute and print a classification report from a 2x2 confusion matrix.
    """
    tp, fn, fp, tn = cm.ravel()
    print(f"\n\t\ttp: {tp}, fn: {fn}, fp: {fp}, tn: {tn}\n")

    true_labels = [0] * (tp + fn) + [1] * (fp + tn)
    predicted_labels = [0] * tp + [1] * fn + [0] * fp + [1] * tn

    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=["Good", "Bad"],
        zero_division=1,
    )
    print(report)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_labels: Sequence[str],
    title: str = "Confusion Matrix",
) -> None:
    """
    Plot a confusion matrix heatmap and print the associated classification report.
    """
    print_classification_report_from_cm(cm)

    num_labels = len(class_labels)
    fig_size = max(5, num_labels)

    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    plt.title(title)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.tight_layout()
    plt.show()


# =============================================================================
# High-level workflow helpers
# =============================================================================


def filter_by_species_and_dates(
    df: pd.DataFrame,
    species: str,
    date_range: Optional[Tuple[str, str]] = None,
) -> pd.DataFrame:
    """
    Filter the DataFrame by species and optional inclusive date range.
    """
    filters: Dict[str, Iterable] = {"Species": [species]}

    if date_range is not None:
        start_date, end_date = date_range
        all_dates = sorted(df["Date"].unique())
        date_list = [d for d in all_dates if start_date <= d <= end_date]
        filters["Date"] = date_list

    return filter_df(df, filters)


def analyze_species(
    df: pd.DataFrame,
    species: str,
    date_range: Optional[Tuple[str, str]] = None,
    plot_title: Optional[str] = None,
) -> None:
    """
    Run the full analysis pipeline for a given species and optional date range.
    """
    result_df = filter_by_species_and_dates(df, species=species, date_range=date_range)
    if result_df.empty:
        print(f"No data found for species={species} and date_range={date_range}")
        return

    cm = build_confusion_matrix(result_df)
    # pretty_print_confusion_matrix(cm, species=species)

    if plot_title is None:
        plot_title = f"{species.capitalize()} Test data Confusion Matrix"

    plot_confusion_matrix(cm, ["Good", "Bad"], title=plot_title)


# =============================================================================
# Main entry point
# =============================================================================


def main() -> None:
    """
    Default script entry point.
    """
    config = DBConfig()
    print("Loading Feedback table from database...")
    df_raw = load_feedback_table(config)

    print("Enriching and cleaning DataFrame...")
    df = enrich_feedback_dataframe(df_raw)

    # Example analysis
    date_range = ("2024-12-02", "2025-01-31")
    analyze_species(
        df,
        species="sardine",
        date_range=date_range,
        plot_title="Sardine Test data Confusion Matrix",
    )


if __name__ == "__main__":
    main()
