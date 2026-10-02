from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET_COLUMN = "CLASS1"
KNOWN_CLASSES = {"BLL", "FSRQ"}
UNCERTAIN_CLASS = "BCU"

# Columns explicitly excluded in the original notebook.
DEFAULT_EXCLUDED_COLUMNS = {
    "Unc_LP_beta",
    "LP_EPeak",
    "Unc_LP_EPeak",
    "Unc_PLEC_Exp_Index",
    "PLEC_EPeak",
    "Unc_PLEC_EPeak",
    "ASSOC_PROB_BAY",
    "Flags",
    "Signif_Peak",
    "Flux_Peak",
    "Unc_Flux_Peak",
    "Time_Peak",
    "Peak_Interval",
}


@dataclass
class DataSplits:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    bcu: pd.DataFrame


@dataclass
class Preprocessor:
    feature_names: list[str]
    imputer: SimpleImputer
    scaler: StandardScaler

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame.reindex(columns=self.feature_names)
        x = self.imputer.transform(x)
        return self.scaler.transform(x).astype(np.float32)


def load_catalog(path: str | Path) -> pd.DataFrame:
    """Load the main table from the Fermi-LAT FITS catalog."""
    from astropy.io import fits
    from astropy.table import Table

    path = Path(path)
    with fits.open(path) as hdul:
        table = Table(hdul[1].data)

    # Keep one-dimensional columns only; multidimensional FITS columns are not
    # directly suitable as scalar MLP features.
    scalar_columns = [name for name in table.colnames if len(table[name].shape) <= 1]
    frame = table[scalar_columns].to_pandas()

    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Catalog does not contain required column {TARGET_COLUMN!r}")

    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype(str).str.strip().str.upper()
    frame = frame[frame[TARGET_COLUMN].isin(KNOWN_CLASSES | {UNCERTAIN_CLASS})].copy()
    return frame.reset_index(drop=True)


def select_numeric_features(
    frame: pd.DataFrame,
    excluded_columns: set[str] | None = None,
) -> pd.DataFrame:
    """Return numeric features plus CLASS1, excluding known problematic columns."""
    excluded = DEFAULT_EXCLUDED_COLUMNS if excluded_columns is None else excluded_columns
    numeric = frame.select_dtypes(include=[np.number]).copy()
    numeric = numeric.drop(columns=[c for c in excluded if c in numeric.columns], errors="ignore")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric[TARGET_COLUMN] = frame[TARGET_COLUMN].values
    return numeric


def make_splits(
    frame: pd.DataFrame,
    test_fraction: float = 0.15,
    val_fraction: float = 0.15,
    random_state: int = 42,
) -> DataSplits:
    """Create stratified train/validation/test splits and keep BCUs separate."""
    if test_fraction <= 0 or val_fraction <= 0 or test_fraction + val_fraction >= 1:
        raise ValueError("test_fraction and val_fraction must be positive and sum to < 1")

    classified = frame[frame[TARGET_COLUMN].isin(KNOWN_CLASSES)].copy()
    bcu = frame[frame[TARGET_COLUMN] == UNCERTAIN_CLASS].copy()

    # Original project convention: BLL = 1, FSRQ = 0.
    y = (classified[TARGET_COLUMN] == "BLL").astype(np.float32)
    X = classified.drop(columns=[TARGET_COLUMN])

    holdout = test_fraction + val_fraction
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=holdout,
        random_state=random_state,
        stratify=y,
    )

    relative_test_fraction = test_fraction / holdout
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test_fraction,
        random_state=random_state,
        stratify=y_temp,
    )

    return DataSplits(X_train, X_val, X_test, y_train, y_val, y_test, bcu)


def fit_preprocessor(X_train: pd.DataFrame) -> Preprocessor:
    """Fit median imputation and standardization on training data only."""
    feature_names = list(X_train.columns)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    imputed = imputer.fit_transform(X_train)
    scaler.fit(imputed)
    return Preprocessor(feature_names, imputer, scaler)
