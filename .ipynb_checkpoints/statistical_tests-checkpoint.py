from scipy import stats
from scipy.stats import ks_2samp
import pandas as pd
import numpy as np

def test_normality(df, column, alpha=0.05):
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found")

    if not pd.api.types.is_numeric_dtype(df[column].dtype):
        raise TypeError(f"Column '{column}' must be numeric")

    values = df[column].dropna().to_numpy()

    if len(values) < 8:
        raise ValueError("At least 8 observations are required")

    if not np.isfinite(values).all():
        raise ValueError("Column contains infinite values")

    if np.ptp(values) == 0:
        raise ValueError("Cannot test a constant column")

    result = stats.normaltest(values)

    return {
        "statistic": result.statistic,
        "p_value": result.pvalue,
        "is_normal": result.pvalue >= alpha
    }

def ks_test(
    file_a,
    file_b,
    column_a,
    column_b=None,
    alpha=0.05
):
    # Load datasets
    df_a = pd.read_csv(file_a)
    df_b = pd.read_csv(file_b)

    if column_b is None:
        column_b = column_a

    # Extract numerical observations and remove missing values
    sample_a = pd.to_numeric(
        df_a[column_a], errors="coerce"
    ).dropna()

    sample_b = pd.to_numeric(
        df_b[column_b], errors="coerce"
    ).dropna()

    if len(sample_a) == 0 or len(sample_b) == 0:
        raise ValueError("Both columns must contain numerical values.")

    # Perform two-sample KS test
    result = ks_2samp(
        sample_a,
        sample_b,
        alternative="two-sided",
        method="auto"
    )

    if result.pvalue < alpha:
        print("Result: Distributions are significantly different.")
    else:
        print("Result: No significant difference detected.")

    return result