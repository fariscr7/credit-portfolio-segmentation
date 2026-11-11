"""Stage 1-2 helpers: load, describe and clean the Lending Club loan data.

Every function takes a DataFrame and returns a NEW one (the input is never
changed), so any notebook cell can be re-run safely.
"""

import numpy as np
import pandas as pd

from src.config import DATE_COLUMN, RAW_DATA_FILE, TEXT_COLUMNS

# What each raw column means (from the dataset's published documentation).
COLUMN_DESCRIPTIONS = {
    "id": "Loan id (unique identifier)",
    "issue_d": "Month and year the loan was approved, e.g. Dec-2015",
    "revenue": "Borrower's self-declared annual income (USD)",
    "dti_n": "Debt-to-income ratio excluding mortgage (%)",
    "loan_amnt": "Amount of credit requested (USD)",
    "fico_n": "FICO credit score (average of the low and high range)",
    "experience_c": "Binary flag: whether the borrower is new to Lending "
                    "Club (coding direction not stated in the source docs)",
    "emp_length": "Employment length; 'NI' means no information",
    "purpose": "Loan purpose category",
    "home_ownership_n": "Home ownership: MORTGAGE, RENT, OWN or OTHER",
    "addr_state": "US state of residence (2-letter code)",
    "zip_code": "First 3 digits of the residence zip code, e.g. 190xx",
    "Default": "1 = defaulted (charged off), 0 = fully paid",
    "title": "Loan title written by the borrower (free text)",
    "desc": "Loan description written by the borrower (free text)",
}

# How each column will be used in the project (decided in Stages 1-4).
PLANNED_ROLES = {
    "id": "Identifier - kept for traceability, never a clustering feature",
    "issue_d": "Time - issue year/month for trend analysis",
    "revenue": "Clustering feature (log-transformed)",
    "dti_n": "Clustering feature",
    "loan_amnt": "Clustering feature",
    "fico_n": "Clustering feature",
    "experience_c": "Excluded - almost constant (no information)",
    "emp_length": "Clustering feature (converted to years)",
    "purpose": "Clustering feature (encoded)",
    "home_ownership_n": "Clustering feature (encoded)",
    "addr_state": "Profiling only - geographic analysis after clustering",
    "zip_code": "Excluded - too many values and sensitive location data",
    "Default": "Outcome - post-clustering analysis only, never clustered",
    "title": "Dropped - free text, out of project scope",
    "desc": "Dropped - free text, 91% missing, out of project scope",
}

# Employment length text -> number of years. "NI" (no information) is left
# out on purpose, so it becomes a missing value (NaN) when mapped.
EMP_LENGTH_YEARS = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
    "10+ years": 10,
}
EMP_LENGTH_NO_INFO = "NI"
EMP_LENGTH_ORDER = list(EMP_LENGTH_YEARS) + [EMP_LENGTH_NO_INFO]

DTI_PLACEHOLDER = 999   # code used where the real ratio is unknown
CATEGORY_COLUMNS = ["purpose", "home_ownership_n", "addr_state", "zip_code"]


def load_raw_data(path=RAW_DATA_FILE):
    """Read the raw CSV into a DataFrame.

    low_memory=False makes pandas read the whole file before choosing each
    column's type, which avoids the mixed-type warning on the `desc` column.
    """
    return pd.read_csv(path, low_memory=False)


def first_valid_value(column):
    """Return the first non-missing value of a column, as text."""
    non_missing = column.dropna()
    return str(non_missing.iloc[0]) if len(non_missing) else ""


def describe_columns(df):
    """One row per column: type, missing values, distinct values, an example,
    meaning and planned role. This table is the project's data dictionary.
    Free-text examples are hidden because they can contain personal details.
    """
    examples = [
        "(free text - not shown)" if col in TEXT_COLUMNS
        else first_valid_value(df[col])
        for col in df.columns
    ]
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2),
        "unique_values": df.nunique(),
        "example": examples,
    })
    summary["meaning"] = summary.index.map(COLUMN_DESCRIPTIONS)
    summary["planned_role"] = summary.index.map(PLANNED_ROLES)
    summary.index.name = "column"
    return summary


def iqr_outlier_count(values):
    """Count values outside Tukey's fences: below Q1 - 1.5*IQR or above
    Q3 + 1.5*IQR, where IQR = Q3 - Q1 (the middle 50% of the data)."""
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    outside = (values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)
    return int(outside.sum())


def log_step(log, step, rows_affected, action, reason):
    """Append one entry to the cleaning log (a list of dictionaries)."""
    log.append({
        "step": step,
        "rows_affected": int(rows_affected),
        "action": action,
        "reason": reason,
    })


def drop_text_columns(df):
    """Remove the free-text columns (`title`, `desc`)."""
    return df.drop(columns=TEXT_COLUMNS)


def remove_duplicates(df):
    """Remove exact duplicate rows (identical in every column)."""
    return df.drop_duplicates().reset_index(drop=True)


def add_issue_date_features(df):
    """Turn the text date (e.g. 'Dec-2015') into a real date plus its year
    and month, then drop the original text column."""
    out = df.copy()
    out["issue_date"] = pd.to_datetime(out[DATE_COLUMN], format="%b-%Y")
    out["issue_year"] = out["issue_date"].dt.year
    out["issue_month"] = out["issue_date"].dt.month
    return out.drop(columns=DATE_COLUMN)


def encode_emp_length(df):
    """Add two columns based on `emp_length`:
    - emp_length_years: 0-10 years, or NaN where the value is 'NI'
    - emp_length_unknown: 1 if the value is 'NI', otherwise 0
    The original text column is kept (as an ordered category) for charts."""
    out = df.copy()
    out["emp_length_unknown"] = (
        out["emp_length"] == EMP_LENGTH_NO_INFO
    ).astype(int)
    out["emp_length_years"] = out["emp_length"].map(EMP_LENGTH_YEARS)
    out["emp_length"] = pd.Categorical(
        out["emp_length"], categories=EMP_LENGTH_ORDER,
    )
    return out


def replace_dti_placeholder(df):
    """Replace the placeholder DTI value 999 with NaN (unknown)."""
    out = df.copy()
    out["dti_n"] = out["dti_n"].replace(DTI_PLACEHOLDER, np.nan)
    return out


def fill_missing_zip(df):
    """Label the missing zip code as 'Unknown' (zip is not a model input)."""
    out = df.copy()
    out["zip_code"] = out["zip_code"].fillna("Unknown")
    return out


def convert_to_category(df):
    """Store repeated text labels as the memory-efficient 'category' type."""
    out = df.copy()
    for column in CATEGORY_COLUMNS:
        out[column] = out[column].astype("category")
    return out


def clean_loans(raw):
    """Run every cleaning step in order.

    Returns the clean DataFrame and the cleaning log as a DataFrame, so the
    whole of Stage 2 can be reproduced with one call (e.g. on new data).
    """
    log = []
    df = drop_text_columns(raw)
    log_step(log, "Drop text columns", len(df), "Dropped title, desc",
             "Free text is out of scope; desc is 91% missing")

    before = len(df)
    df = remove_duplicates(df)
    log_step(log, "Remove duplicate rows", before - len(df),
             "Dropped exact duplicates", "Each loan must appear once")

    df = add_issue_date_features(df)
    log_step(log, "Parse issue date", len(df),
             "issue_d -> issue_date, issue_year, issue_month",
             "Text dates cannot be sorted or grouped by year")

    df = encode_emp_length(df)
    log_step(log, "Encode employment length",
             df["emp_length_unknown"].sum(),
             "Text -> years (0-10); 'NI' -> NaN plus a 0/1 flag",
             "'NI' means no information: a hidden missing value")

    n_placeholder = (df["dti_n"] == DTI_PLACEHOLDER).sum()
    df = replace_dti_placeholder(df)
    log_step(log, "Fix DTI placeholder", n_placeholder, "999 -> NaN",
             "999 is a code for 'unknown', not a real ratio")

    n_missing_zip = df["zip_code"].isna().sum()
    df = fill_missing_zip(df)
    log_step(log, "Fill missing zip code", n_missing_zip,
             "NaN -> 'Unknown'", "Only one row; zip is not a model input")

    df = convert_to_category(df)
    log_step(log, "Convert text labels to category", len(df),
             "purpose, home_ownership_n, addr_state, zip_code -> category",
             "Less memory and faster grouping")

    return df, pd.DataFrame(log)
