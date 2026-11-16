"""Stage 9 helpers: find unusual borrower profiles.

Two lenses, as the project asks: density-based (DBSCAN's noise rule) and
feature-level (simple rules on single features and one ratio). An anomaly is
a profile to REVIEW, such as a possible data-entry error or a rare case. It
is never a verdict on a borrower.
"""

import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from src.clustering import DBSCAN_MIN_SAMPLES
from src.config import TARGET_COLUMN

EXTREME_SHARE = 0.001        # top / bottom 0.1% of incomes count as extreme
MAX_LOAN_TO_INCOME = 0.5     # a loan above half of the annual income
MAX_DTI = 50                 # more than half of income already goes to debt

RULE_LABELS = {
    "very_high_income": "Very high income (top 0.1%)",
    "very_low_income": "Very low income (bottom 0.1%)",
    "loan_over_half_income": "Loan > 50% of annual income",
    "dti_over_50": "DTI above 50%",
    "dti_code_999": "DTI recorded as 999 (unknown)",
}


def density_distance(features, sample_positions, eps,
                     min_samples=DBSCAN_MIN_SAMPLES):
    """Distance from every loan to the nearest DBSCAN core point.

    DBSCAN is fitted on the shared sample with the settings of notebook 04.
    DBSCAN's own rule says a point within `eps` of a core point belongs to
    a cluster and any other point is noise, so a loan further than `eps`
    from every core point is a density-based anomaly. This applies the rule
    to all loans without running DBSCAN on 1.35 million points.
    """
    sample = features[sample_positions]
    model = DBSCAN(eps=eps, min_samples=min_samples).fit(sample)
    core_points = sample[model.core_sample_indices_]
    nearest = NearestNeighbors(n_neighbors=1).fit(core_points)
    distances, _ = nearest.kneighbors(features)
    return distances[:, 0], model


def feature_anomaly_flags(loans):
    """One True/False column per feature-level rule, and the thresholds.

    Income thresholds come from the data (top and bottom 0.1%); the
    loan-to-income and DTI limits are business rules; DTI code 999 was
    replaced by a missing value in notebook 01.
    """
    income = loans["revenue"]
    high_income = income.quantile(1 - EXTREME_SHARE)
    low_income = income.quantile(EXTREME_SHARE)
    loan_to_income = loans["loan_amnt"] / income
    flags = pd.DataFrame({
        "very_high_income": income >= high_income,
        "very_low_income": income <= low_income,
        "loan_over_half_income": loan_to_income > MAX_LOAN_TO_INCOME,
        "dti_over_50": loans["dti_n"] > MAX_DTI,
        "dti_code_999": loans["dti_n"].isna(),
    }, index=loans.index)
    thresholds = {
        "very_high_income": f"income >= ${high_income:,.0f}",
        "very_low_income": f"income <= ${low_income:,.0f}",
        "loan_over_half_income": f"loan / income > {MAX_LOAN_TO_INCOME}",
        "dti_over_50": f"DTI > {MAX_DTI}%",
        "dti_code_999": "DTI was 999 in the raw data",
    }
    return flags, thresholds


def anomaly_summary(loans, flags):
    """Loans, share of the portfolio, default rate and lift per flag."""
    overall_rate = loans[TARGET_COLUMN].mean()
    rows = []
    for name in flags.columns:
        mask = flags[name].to_numpy()
        rate = loans.loc[mask, TARGET_COLUMN].mean()
        rows.append({"flag": name, "loans": int(mask.sum()),
                     "share_%": 100 * mask.mean(),
                     "default_rate_%": 100 * rate,
                     "lift": rate / overall_rate})
    return pd.DataFrame(rows).set_index("flag")
