"""Stage 0 check: confirm that Python, the libraries and the dataset are ready.

Run it from the project root folder:
    python check_environment.py
"""

import importlib
import sys

from src.config import RAW_DATA_FILE

# Oldest Python version this project supports.
MIN_PYTHON = (3, 11)

# Libraries the project needs: {name used in `import`: name used in `pip`}.
REQUIRED_LIBRARIES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "sklearn": "scikit-learn",
    "scipy": "scipy",
    "umap": "umap-learn",
    "streamlit": "streamlit",
}


def check_python_version():
    """Print the Python version and return True if it is new enough."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    if (version.major, version.minor) < MIN_PYTHON:
        print(f"  -> Please install Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} "
              "or newer.")
        return False
    return True


def check_libraries():
    """Try to import every required library and print its version."""
    all_ok = True
    for import_name, pip_name in REQUIRED_LIBRARIES.items():
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, "__version__", "unknown")
            print(f"  [OK]      {pip_name:<14} {version}")
        except ImportError:
            print(f"  [MISSING] {pip_name:<14} -> pip install {pip_name}")
            all_ok = False
    return all_ok


def check_dataset():
    """Confirm the raw CSV exists, then preview its column names."""
    if not RAW_DATA_FILE.exists():
        print(f"  [MISSING] {RAW_DATA_FILE}")
        print("  -> Put the CSV in data/raw/ with the name above.")
        return False

    # Imported here so that a missing pandas is reported in step 2
    # instead of crashing the whole script at the top.
    import pandas as pd

    size_mb = RAW_DATA_FILE.stat().st_size / (1024 ** 2)
    preview = pd.read_csv(RAW_DATA_FILE, nrows=5)  # read 5 rows only
    print(f"  [OK]      {RAW_DATA_FILE.name} ({size_mb:.1f} MB)")
    print(f"  Columns ({preview.shape[1]}): {list(preview.columns)}")
    return True


def main():
    """Run the three checks and print an overall result."""
    print("=== 1. Python ===")
    python_ok = check_python_version()

    print("\n=== 2. Libraries ===")
    libraries_ok = check_libraries()

    print("\n=== 3. Dataset ===")
    dataset_ok = check_dataset()

    print("\n=== Result ===")
    if python_ok and libraries_ok and dataset_ok:
        print("Everything is ready. You can start Stage 1.")
    else:
        print("Fix the items marked above, then run this script again.")


if __name__ == "__main__":
    main()
