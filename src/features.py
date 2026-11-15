"""Stage 4-5 helpers: build the clustering features and the PCA pipeline.

The pipeline is fitted once (notebook 03), saved to models/, and reused by
the clustering notebook and the dashboard, so every borrower is transformed
in exactly the same way.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import ID_COLUMN, RANDOM_SEED, TARGET_COLUMN

# 14 loan purposes -> 5 groups. Purposes not listed fall into
# "personal_other" (other, major_purchase, medical, car, moving, vacation,
# wedding, renewable_energy, educational).
PURPOSE_GROUPS = {
    "debt_consolidation": "debt_consolidation",
    "credit_card": "credit_card",
    "home_improvement": "home",
    "house": "home",
    "small_business": "small_business",
}
OTHER_PURPOSE_GROUP = "personal_other"

# The columns the clustering is allowed to see, by type.
SCALED_FEATURES = ["log_income", "loan_amount", "fico", "emp_length_years"]
CAPPED_FEATURES = ["dti"]                     # capped, then scaled
FLAG_FEATURES = ["emp_length_unknown"]        # already 0/1
CATEGORY_FEATURES = ["home_ownership", "purpose_group"]
FEATURE_COLUMNS = (
    SCALED_FEATURES + CAPPED_FEATURES + FLAG_FEATURES + CATEGORY_FEATURES
)

# Columns that must never reach the clustering.
FORBIDDEN_COLUMNS = [TARGET_COLUMN, ID_COLUMN]

DTI_CAP_QUANTILE = 0.99
PCA_COMPONENTS = 7


def group_purpose(purpose):
    """Map each loan purpose to one of 5 purpose groups."""
    return (
        purpose.astype(str).map(PURPOSE_GROUPS).fillna(OTHER_PURPOSE_GROUP)
    )


def prepare_feature_frame(loans):
    """Build the clustering input table from cleaned loans.

    Only borrower and loan characteristics are kept. `Default`, `id`,
    dates and location are left out on purpose.
    """
    features = pd.DataFrame({
        "log_income": np.log1p(loans["revenue"]),
        "loan_amount": loans["loan_amnt"].astype(float),
        "fico": loans["fico_n"].astype(float),
        "emp_length_years": loans["emp_length_years"].astype(float),
        "dti": loans["dti_n"].astype(float),
        "emp_length_unknown": loans["emp_length_unknown"].astype(int),
        "home_ownership": loans["home_ownership_n"].astype(str),
        "purpose_group": group_purpose(loans["purpose"]),
    }, index=loans.index)
    assert_no_leakage(features)
    return features[FEATURE_COLUMNS]


def assert_no_leakage(features):
    """Stop with an error if an outcome or identifier column slipped in."""
    leaked = [col for col in FORBIDDEN_COLUMNS if col in features.columns]
    if leaked:
        raise ValueError(f"Columns that must not be clustered: {leaked}")


class UpperCapper(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Cap each column at a high quantile learned during fit (winsorizing).

    Values above the cap are set to the cap, so a few extreme borrowers
    cannot stretch the scale. Missing values stay missing.
    """

    def __init__(self, quantile=0.99):
        self.quantile = quantile

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        self.caps_ = np.nanquantile(values, self.quantile, axis=0)
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X):
        return np.minimum(np.asarray(X, dtype=float), self.caps_)


def build_preprocessor():
    """Impute, cap, scale and encode the feature table.

    Numeric columns are standardised (mean 0, standard deviation 1).
    One-hot and flag columns stay 0/1: standardising a rare category
    (e.g. home ownership OTHER, 0.04% of loans) would turn its 1s into
    values around 50 and let 517 loans dominate the distances.
    """
    scaled = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    capped = Pipeline([
        ("cap", UpperCapper(quantile=DTI_CAP_QUANTILE)),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categories = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    return ColumnTransformer([
        ("scaled", scaled, SCALED_FEATURES),
        ("capped", capped, CAPPED_FEATURES),
        ("flag", "passthrough", FLAG_FEATURES),
        ("category", categories, CATEGORY_FEATURES),
    ], verbose_feature_names_out=False)


def build_feature_pipeline(n_components=PCA_COMPONENTS):
    """Preprocessing followed by PCA, as one reusable object."""
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("pca", PCA(n_components=n_components, random_state=RANDOM_SEED)),
    ])
