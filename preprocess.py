import pandas as pd
import numpy as np
from pathlib import Path


RAW_FILE = "iot_dataset_raw.csv"
CLEAN_FILE = "iot_dataset_clean.csv"

# Final shared columns for all models
COLUMNS_TO_KEEP = [
    "src_ip",
    "proto",
    "duration",
    "src_bytes",
    "dst_bytes",
    "missed_bytes",
    "src_pkts",
    "src_ip_bytes",
    "dst_pkts",
    "dst_ip_bytes",
    "dns_qclass",
    "dns_qtype",
    "dns_rejected",
    "ssl_version",
    "ssl_cipher",
    "ssl_resumed",
    "http_trans_depth",
    "http_method",
    "http_version",
    "http_request_body_len",
    "http_response_body_len",
    "label",
    "type",
]

# Columns expected to be numeric
NUMERIC_COLUMNS = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "missed_bytes",
    "src_pkts",
    "src_ip_bytes",
    "dst_pkts",
    "dst_ip_bytes",
    "dns_qclass",
    "dns_qtype",
    "dns_rejected",
    "http_trans_depth",
    "http_request_body_len",
    "http_response_body_len",
]

# Columns expected to be categorical/text
CATEGORICAL_COLUMNS = [
    "src_ip",
    "proto",
    "ssl_version",
    "ssl_cipher",
    "ssl_resumed",
    "http_method",
    "http_version",
    "type",
]

TARGET_COLUMN = "label"


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def normalize_label(series: pd.Series) -> pd.Series:
    """
    Convert label column to binary 0/1 safely.
    Handles common forms like:
    0/1, Benign/Attack, Normal/Malicious, False/True, etc.
    """
    if pd.api.types.is_numeric_dtype(series):
        unique_vals = sorted(series.dropna().unique().tolist())
        if set(unique_vals).issubset({0, 1}):
            return series.astype("Int64")
        raise ValueError(
            f"Numeric label column contains unexpected values: {unique_vals}. "
            "Expected binary 0/1."
        )

    cleaned = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .replace({"nan": np.nan, "": np.nan})
    )

    benign_values = {
        "0", "benign", "normal", "false", "no", "non-malicious", "non_malicious"
    }
    malicious_values = {
        "1", "malicious", "attack", "true", "yes", "anomaly"
    }

    mapped = cleaned.map(
        lambda x: 0 if x in benign_values else (1 if x in malicious_values else np.nan)
    )

    unknown = cleaned[mapped.isna() & cleaned.notna()].unique().tolist()
    if unknown:
        raise ValueError(
            f"Unrecognized label values found: {unknown}. "
            "Please update normalize_label() mapping."
        )

    return mapped.astype("Int64")


def clean_categorical_column(series: pd.Series) -> pd.Series:
    """
    Keep categorical values as strings, but standardize obvious missing markers.
    We do NOT encode here because encoding is model-specific.
    """
    missing_tokens = {
        "-", "--", "---", "", " ", "nan", "none", "null", "na", "n/a", "unknown"
    }

    cleaned = (
        series.astype(str)
        .str.strip()
        .replace(list(missing_tokens), np.nan)
    )

    return cleaned


def clean_numeric_column(series: pd.Series) -> pd.Series:
    """
    Convert to numeric safely. Invalid values become NaN.
    We do NOT impute here because imputation is model-specific.
    """
    return pd.to_numeric(series, errors="coerce")


def remove_constant_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
    if constant_cols:
        df = df.drop(columns=constant_cols)
    return df, constant_cols


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    raw_path = base_dir / RAW_FILE
    clean_path = base_dir / CLEAN_FILE

    print_section("STEP 1 - LOAD RAW DATA")
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_path}")

    df = pd.read_csv(raw_path)
    print(f"Loaded raw dataset from: {raw_path}")
    print(f"Raw shape: {df.shape}")

    print_section("STEP 2 - CHECK REQUIRED COLUMNS")
    missing_required = [col for col in COLUMNS_TO_KEEP if col not in df.columns]
    if missing_required:
        raise ValueError(
            f"These required columns are missing from the raw dataset: {missing_required}"
        )

    # Keep only the agreed final shared columns
    df = df[COLUMNS_TO_KEEP].copy()
    print(f"Shape after keeping selected columns: {df.shape}")

    print_section("STEP 3 - STANDARDIZE MISSING VALUES")
    # First, replace common global placeholders
    df = df.replace(
        ["-", "--", "---", " ", "", "NA", "N/A", "na", "n/a", "None", "none", "null", "Null"],
        np.nan,
    )

    print_section("STEP 4 - CLEAN TARGET COLUMN")
    df[TARGET_COLUMN] = normalize_label(df[TARGET_COLUMN])
    if df[TARGET_COLUMN].isna().any():
        raise ValueError("Target column still contains missing values after normalization.")

    print("Label distribution:")
    print(df[TARGET_COLUMN].value_counts(dropna=False))

    print_section("STEP 5 - CLEAN CATEGORICAL COLUMNS")
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = clean_categorical_column(df[col])

    print_section("STEP 6 - CLEAN NUMERIC COLUMNS")
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = clean_numeric_column(df[col])

    print_section("STEP 7 - REMOVE EXACT DUPLICATES")
    before_duplicates = len(df)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    removed_duplicates = before_duplicates - len(df)
    print(f"Removed exact duplicate rows: {removed_duplicates}")
    print(f"Shape after duplicate removal: {df.shape}")

    print_section("STEP 8 - DROP ROWS WITH MISSING TARGET ONLY")
    before_target_drop = len(df)
    df = df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    dropped_target_rows = before_target_drop - len(df)
    print(f"Dropped rows with missing target: {dropped_target_rows}")

    print_section("STEP 9 - OPTIONAL CONSTANT COLUMN CHECK")
    df, constant_cols = remove_constant_columns(df)
    if constant_cols:
        print(f"Dropped constant columns: {constant_cols}")
    else:
        print("No constant columns found.")

    print_section("STEP 10 - FINAL DATA TYPES")
    # Use pandas nullable integer for binary-ish numeric columns when possible
    integer_like_columns = [
        "src_bytes",
        "dst_bytes",
        "missed_bytes",
        "src_pkts",
        "src_ip_bytes",
        "dst_pkts",
        "dst_ip_bytes",
        "dns_qclass",
        "dns_qtype",
        "dns_rejected",
        "http_trans_depth",
        "http_request_body_len",
        "http_response_body_len",
        TARGET_COLUMN,
    ]

    for col in integer_like_columns:
        if col in df.columns:
            non_null = df[col].dropna()
            if len(non_null) > 0 and np.all(np.isclose(non_null, np.round(non_null))):
                df[col] = df[col].astype("Int64")

    if "duration" in df.columns:
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce")

    print(df.dtypes)

    print_section("STEP 11 - FINAL DIAGNOSTICS")
    print(f"Final shape: {df.shape}")
    print("\nMissing values per column:")
    print(df.isna().sum())

    print("\nLabel distribution:")
    print(df[TARGET_COLUMN].value_counts(normalize=False))
    print("\nLabel ratio:")
    print(df[TARGET_COLUMN].value_counts(normalize=True))

    print("\nCategorical unique counts:")
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            print(f"{col}: {df[col].nunique(dropna=True)} unique values")

    print_section("STEP 12 - SAVE CLEAN DATASET")
    df.to_csv(clean_path, index=False)
    print(f"Clean dataset saved to: {clean_path}")


if __name__ == "__main__":
    main()