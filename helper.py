from pathlib import Path
from aggregate import (
    count_entries,
    count_distinct_values,
    get_top_mcv,
    get_entry_size_stats,
    get_value_range,
    get_average,
    get_variance,
    get_correlation
)
import numpy as np
import pandas as pd

def collect_dataset_metrics(
    df,
    columns=None,
    correlation_pairs=None,
    mcv_n=10,
    alpha=0.05,
    encoding="utf-8",
    include_normality=True,
    correlation_methods=("pearson", "spearman", "kendall")
):
    """
    Collect all currently implemented dataset metrics.

    Returns:
        dict: Nested dictionary containing dataset,
              column and correlation metrics.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    if df.empty:
        raise ValueError("DataFrame is empty")

    if columns is None:
        columns = list(df.columns)

    for column in columns:
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found")

    if mcv_n < 1:
        raise ValueError("mcv_n must be at least 1")

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    if correlation_pairs is None:
        correlation_pairs = []

    supported_methods = {"pearson", "spearman", "kendall"}

    if not set(correlation_methods).issubset(supported_methods):
        raise ValueError("Unsupported correlation method")

    metrics = {
        "dataset": {
            "row_count": count_entries(df),
            "column_count": len(df.columns),
            "total_null_count": int(df.isna().sum().sum()),
            "overall_null_ratio": float(
                df.isna().to_numpy().mean()
            )
        },
        "columns": {},
        "correlations": {}
    }

    # Collect individual column metrics
    for column in columns:

        series = df[column]

        # Basic metrics
        column_metrics = {
            "dtype": str(series.dtype),
            "null_count": int(series.isna().sum()),
            "null_ratio": float(series.isna().mean()),

            "distinct_values": count_distinct_values(
                df, column
            ),

            "most_common_values": [
                {"value": value, "count": int(count)}
                for value, count in get_top_mcv(
                    df, column, n=mcv_n
                ).items()
            ]
        }

        # Entry size
        max_size, avg_size = get_entry_size_stats(
            df, column, encoding=encoding
        )

        column_metrics["entry_size"] = {
            "maximum_bytes": max_size,
            "average_bytes": avg_size
        }

        # Integer-specific metrics
        if pd.api.types.is_integer_dtype(series.dtype):

            minimum, maximum = get_value_range(df, column)

            column_metrics["range"] = {
                "minimum": minimum,
                "maximum": maximum
            }
        else:
            column_metrics["range"] = None

        # Numeric metrics
        is_numeric = (
            pd.api.types.is_numeric_dtype(series.dtype)
            and not pd.api.types.is_bool_dtype(series.dtype)
        )

        if is_numeric:

            column_metrics["average"] = get_average(
                df, column
            )

            column_metrics["variance"] = get_variance(
                df, column
            )

            # # Normality test
            # if include_normality:
            #     try:
            #         column_metrics["normality"] = test_normality(
            #             df, column, alpha=alpha
            #         )
            #     except (ValueError, TypeError) as error:
            #         column_metrics["normality"] = {
            #             "tested": False,
            #             "reason": str(error)
            #         }
            # else:
            #     column_metrics["normality"] = None

        else:
            column_metrics["average"] = None
            column_metrics["variance"] = None
            column_metrics["normality"] = None

        metrics["columns"][column] = column_metrics

    # Collect correlations between explicitly specified pairs
    for column1, column2 in correlation_pairs:

        pair_name = f"{column1}__{column2}"

        if column1 not in df.columns or column2 not in df.columns:
            raise KeyError(
                f"Invalid correlation pair: {column1}, {column2}"
            )

        metrics["correlations"][pair_name] = {}

        for method in correlation_methods:

            correlation = get_correlation(
                df, column1, column2, method=method
            )

            metrics["correlations"][pair_name][method] = (
                float(correlation)
                if pd.notna(correlation)
                else None
            )

    return metrics


from IPython.display import display, Markdown

def display_dataset_metrics(metrics):
    """
    Display collected dataset metrics in neatly formatted tables.

    Args:
        metrics (dict): Output of collect_dataset_metrics()

    Returns:
        dict: Dictionary containing the generated display tables.
    """

    if not isinstance(metrics, dict):
        raise TypeError("Metrics must be a dictionary")

    tables = {}

    # --------------------------------------------------
    # 1. Dataset overview
    # --------------------------------------------------
    display(Markdown("## Dataset Overview"))

    dataset = metrics.get("dataset", {})
    tables["dataset"] = pd.DataFrame.from_dict(
        dataset,
        orient="index",
        columns=["Value"]
    )

    display(tables["dataset"])

    # --------------------------------------------------
    # 2. Individual column metrics
    # --------------------------------------------------
    display(Markdown("## Column Metrics"))

    column_rows = {}

    for column, values in metrics.get("columns", {}).items():

        row = {
            "Data type": values.get("dtype"),
            "NULL count": values.get("null_count"),
            "NULL ratio": values.get("null_ratio"),
            "Distinct values": values.get("distinct_values"),
            "Average": values.get("average"),
            "Variance": values.get("variance")
        }

        # Range
        value_range = values.get("range")
        if isinstance(value_range, dict):
            row["Minimum"] = value_range.get("minimum")
            row["Maximum"] = value_range.get("maximum")

        # Entry size
        entry_size = values.get("entry_size")
        if isinstance(entry_size, dict):
            row["Max size (bytes)"] = entry_size.get("maximum_bytes")
            row["Avg size (bytes)"] = entry_size.get("average_bytes")

        # Most common values
        mcv = values.get("most_common_values", [])
        row["Most common values"] = ", ".join(
            f"{item['value']} (n={item['count']})"
            for item in mcv
        )

        # Normality
        normality = values.get("normality")

        if isinstance(normality, dict):
            row["Normality statistic"] = normality.get("statistic")
            row["Normality p-value"] = normality.get("p_value")
            row["Normally distributed"] = normality.get("is_normal")

        column_rows[column] = row

    tables["columns"] = pd.DataFrame.from_dict(
        column_rows,
        orient="index"
    )

    display(tables["columns"])

    # --------------------------------------------------
    # 3. Correlations
    # --------------------------------------------------
    display(Markdown("## Correlations"))

    correlation_rows = []

    for pair, methods in metrics.get("correlations", {}).items():

        row = {"Column pair": pair}
        row.update(methods)
        correlation_rows.append(row)

    tables["correlations"] = pd.DataFrame(
        correlation_rows
    )

    display(tables["correlations"])

    return tables

def load_input_output(input_file, output_file, column):
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output files must be different")

    input_df = pd.read_csv(input_path)

    if column not in input_df.columns:
        raise KeyError(f"Column '{column}' not found in input file")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not output_path.exists():
        shutil.copy2(input_path, output_path)

    output_df = pd.read_csv(output_path)

    if column not in output_df.columns:
        raise KeyError(f"Column '{column}' not found in output file")

    if len(input_df) != len(output_df):
        raise ValueError("Input and output must have the same number of rows")

    return input_df, output_df, output_path


def format_ks_result(
    result,
    alpha=0.05,
    sample_a_name="Sample A",
    sample_b_name="Sample B"
):
    """
    Formats the result of scipy.stats.ks_2samp()
    into a readable command-line report.

    Returns:
        str: Formatted test report.
    """

    statistic = float(result.statistic)
    p_value = float(result.pvalue)

    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1")

    if not (0 <= statistic <= 1 and 0 <= p_value <= 1):
        raise ValueError("Invalid KS test result")

    # Determine statistical conclusion
    if p_value < alpha:
        conclusion = "REJECT H0"
        interpretation = (
            "Statistically significant evidence that the "
            "distributions are different."
        )
    else:
        conclusion = "CANNOT REJECT H0"
        interpretation = (
            "No statistically significant evidence of "
            "a difference between the distributions."
        )

    # Format report
    separator = "=" * 65

    report = [
        separator,
        "          KOLMOGOROV-SMIRNOV TEST REPORT",
        separator,
        "",
        f"Comparing: {sample_a_name} vs. {sample_b_name}",
        "",
        "-" * 65,
        "TEST METRICS",
        "-" * 65,
        f"KS Statistic (D): {statistic:.6f}",
        f"P-value:          {p_value:.6e}",
        f"Significance:     {alpha:.4f}",
        "",
        "-" * 65,
        "INTERPRETATION",
        "-" * 65,
        "",
        "1. KS Statistic (D):",
        "   Maximum distance between the two empirical",
        "   cumulative distribution functions.",
        "   Range: 0 (identical ECDFs) to 1 (maximum separation).",
        "",
        "2. P-value:",
        "   Measures how extreme the observed difference would be",
        "   under the assumption that both samples come from",
        "   the same underlying distribution.",
        "",
        "3. Significance level (alpha):",
        "   Threshold used to decide whether to reject H0.",
        "",
        "-" * 65,
        "TEST RESULT",
        "-" * 65,
        f"Decision: {conclusion}",
        "",
        interpretation,
        "",
        separator
    ]

    # Include location of maximum difference if available
    location = getattr(result, "statistic_location", None)

    if location is not None:
        report.insert(
            10,
            f"Maximum distance occurs at: {location}"
        )

    sign = getattr(result, "statistic_sign", None)

    if sign is not None:
        direction = (
            f"{sample_a_name} ECDF is above {sample_b_name} ECDF."
            if sign == 1
            else f"{sample_b_name} ECDF is above {sample_a_name} ECDF."
        )

        report.insert(11, f"Direction: {direction}")

    return "\n".join(report)

def generate_bounded_noise(
    n,
    reference_variance,
    variance_tolerance=0.05,
    absolute_variance_tolerance=0.0,
    randomness=1.0,
    seed=43,
    rng=None
):
    """
    Generates bounded random perturbations with zero mean.

    The maximum perturbation is derived from the
    permitted variance deviation.

    randomness:
        0.0 = no randomness
        1.0 = maximum permitted randomness
    """

    if n < 2:
        raise ValueError(
            "At least two observations are required"
        )

    if (
        reference_variance < 0
        or not np.isfinite(reference_variance)
    ):
        raise ValueError(
            "Reference variance must be finite and non-negative"
        )

    if (
        variance_tolerance < 0
        or absolute_variance_tolerance < 0
        or not np.isfinite(variance_tolerance)
        or not np.isfinite(absolute_variance_tolerance)
    ):
        raise ValueError(
            "Variance tolerances must be finite and non-negative"
        )

    if not 0 <= randomness <= 1:
        raise ValueError(
            "Randomness must be between 0 and 1"
        )

    if rng is None:
        rng = np.random.default_rng(seed)

    # Calculate allowed variance deviation
    allowed_variance_error = (
        reference_variance * variance_tolerance
        + absolute_variance_tolerance
    )

    reference_std = np.sqrt(reference_variance)

    # Maximum standard deviation of random noise
    max_noise_std = (
        np.sqrt(reference_variance + allowed_variance_error)
        - reference_std
    )

    # Maximum individual perturbation
    max_deviation = (
        max_noise_std * np.sqrt((n - 1) / n)
    )

    if max_deviation == 0 or randomness == 0:
        return np.zeros(n), max_deviation

    # Generate random values in opposite pairs
    # This guarantees that the perturbations sum to zero
    half = rng.uniform(-1, 1, size=n // 2)

    noise = np.concatenate([half, -half])

    if n % 2:
        noise = np.append(noise, 0.0)

    rng.shuffle(noise)

    peak = np.max(np.abs(noise))

    if peak == 0:
        return np.zeros(n), max_deviation

    # Determine actual perturbation strength
    actual_deviation = (
        rng.uniform(0, randomness) * max_deviation
    )

    noise = noise / peak * actual_deviation

    return noise, max_deviation


def get_valid_values(input_df, output_df, column):

    if (
        not pd.api.types.is_numeric_dtype(input_df[column])
        or pd.api.types.is_bool_dtype(input_df[column])
    ):
        raise TypeError("Input column must be numeric")

    if (
        not pd.api.types.is_numeric_dtype(output_df[column])
        or pd.api.types.is_bool_dtype(output_df[column])
    ):
        raise TypeError("Output column must be numeric")

    # Only consider rows that are valid in BOTH datasets
    valid_mask = (
        input_df[column].notna()
        & output_df[column].notna()
    )

    # Extract corresponding values
    input_values = input_df.loc[
        valid_mask, column
    ].to_numpy(dtype=float)

    output_values = output_df.loc[
        valid_mask, column
    ].to_numpy(dtype=float)

    if len(input_values) < 2:
        raise ValueError("At least two valid observations required")

    if not np.isfinite(input_values).all():
        raise ValueError("Input contains infinite values")

    if not np.isfinite(output_values).all():
        raise ValueError("Output contains infinite values")

    return valid_mask, input_values, output_values
