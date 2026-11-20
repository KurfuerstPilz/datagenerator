import pandas as pd

def get_null_ratio(df, column):
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found")

    if df.empty:
        raise ValueError("Cannot calculate NULL ratio of an empty DataFrame")

    return float(df[column].isna().mean())

def count_distinct_values(df, column, include_null=False):
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found")

    return df[column].nunique(dropna=not include_null)

def count_entries(df):
    return len(df)

def get_value_range(df, column):
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found")

    if not pd.api.types.is_integer_dtype(df[column].dtype):
        raise TypeError(f"Column '{column}' must contain integer values")

    if df[column].notna().sum() == 0:
        return None, None

    minimum = df[column].min()
    maximum = df[column].max()

    return minimum, maximum

def get_average(df, column):
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found")

    if not pd.api.types.is_numeric_dtype(df[column].dtype) or \
       pd.api.types.is_bool_dtype(df[column].dtype):
        raise TypeError(f"Column '{column}' must contain numeric values")

    if df[column].notna().sum() == 0:
        return None

    return df[column].mean()

def get_variance(df, column):
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found")

    if not pd.api.types.is_numeric_dtype(df[column].dtype) or \
       pd.api.types.is_bool_dtype(df[column].dtype):
        raise TypeError(f"Column '{column}' must contain numeric values")

    if df[column].notna().sum() == 0:
        return None

    return df[column].var()

def get_top_mcv(df, column, n=10):
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found")

    return df[column].value_counts().head(n)

def get_entry_size_stats(df, column, encoding="utf-8"):
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found")

    values = df[column].dropna()

    if values.empty:
        return None, None

    # Calculate the byte size of each entry
    sizes = values.map(
        lambda value: len(str(value).encode(encoding))
    )

    maximum_size = int(sizes.max())
    average_size = float(sizes.mean())

    return maximum_size, average_size

def get_correlation(df, column1, column2, method="pearson"):
    for column in [column1, column2]:
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found")

        if not pd.api.types.is_numeric_dtype(df[column].dtype):
            raise TypeError(f"Column '{column}' must be numeric")

    if method not in ["pearson", "spearman", "kendall"]:
        raise ValueError("Unsupported correlation method")

    return df[column1].corr(df[column2], method=method)