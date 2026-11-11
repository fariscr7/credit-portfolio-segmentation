"""Central project settings: paths, file names, column roles and the seed.

Every notebook, script and the Streamlit dashboard imports from this file,
so a path, a column name or the seed only ever has to be changed in one place.
"""

from pathlib import Path

# Project root = the folder that contains this "src" folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data folders.
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Output folders.
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Input and output files.
RAW_DATA_FILE = RAW_DATA_DIR / "loans_granting_model_dataset.csv"
CLEAN_DATA_FILE = PROCESSED_DATA_DIR / "loans_clean.pkl"
DATA_DICTIONARY_FILE = REPORTS_DIR / "data_dictionary.csv"
CLEANING_LOG_FILE = REPORTS_DIR / "cleaning_log.csv"

# Reproducibility: one seed for every random step
# (sampling, K-Means, t-SNE, UMAP).
RANDOM_SEED = 42

# Columns with special roles (confirmed on the real file in Stage 1).
ID_COLUMN = "id"            # identifier -> never a clustering feature
TARGET_COLUMN = "Default"   # outcome -> analysis only, never clustered
DATE_COLUMN = "issue_d"     # text like "Dec-2015" -> parsed in Stage 2
TEXT_COLUMNS = ["title", "desc"]   # free text -> out of scope, dropped
NUMERIC_COLUMNS = ["revenue", "dti_n", "loan_amnt", "fico_n"]
CATEGORICAL_COLUMNS = [
    "emp_length", "purpose", "home_ownership_n", "addr_state", "zip_code",
]
BINARY_COLUMNS = ["experience_c"]
