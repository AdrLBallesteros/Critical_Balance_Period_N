# -*- coding: utf-8 -*-
"""
Created on Feb 2026

@author: Adrián López-Ballesteros
"""

import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo


# Folder paths
input_folder_path = r"\outputs\CAMELS\dynamic"
plot_path = r"\outputs\plots\CAMELS-plots"
log_path = r"\outputs\processing_log.txt"
summary_csv_path = r"\outputs\Effective_Errors_summary.csv"

# Variables to analyze.
variables = [
    "streamflow",
]

effective_error_columns = [f"EE_{variable}" for variable in variables]

# Aggregate the daily data to yearly data
# 'mean' "sum"
variable_agg = "mean"

# Disable interactive display during batch processing.
show_plots = False


def find_longest_consecutive_sequence(years_list):
    """Return the longest run of consecutive years."""
    if not years_list:
        return []

    longest = []
    current = [years_list[0]]
    for i in range(1, len(years_list)):
        if years_list[i] == years_list[i - 1] + 1:
            current.append(years_list[i])
        else:
            if len(current) > len(longest):
                longest = current
            current = [years_list[i]]

    if len(current) > len(longest):
        longest = current

    return longest


def read_input_csv(input_file_path):
    """Read one input CSV file once."""
    return pd.read_csv(input_file_path, parse_dates=["time"])


def aggregate_variable_yearly(df_raw, variable, aggregation):
    """Aggregate one daily variable to yearly values."""
    df_variable = df_raw[["time", variable]].dropna(subset=[variable]).copy()
    df_variable.set_index("time", inplace=True)

    yearly_df = df_variable.resample("YE").agg({variable: aggregation})
    yearly_df.reset_index(inplace=True)
    yearly_df["Year"] = yearly_df["time"].dt.year

    return yearly_df[["Year", variable]]


def build_yearly_dataframes(df_raw, selected_variables, aggregation):
    """Build one yearly dataframe per selected variable."""
    yearly_dataframes = []
    for variable in selected_variables:
        yearly_dataframes.append(
            aggregate_variable_yearly(df_raw, variable, aggregation)
        )

    return yearly_dataframes


def get_common_period_dataframe(yearly_dataframes):
    """Merge yearly series and keep the longest common consecutive period."""
    # The analysis only uses years present in every selected variable.
    merged_df = yearly_dataframes[0].copy()
    for yearly_df in yearly_dataframes[1:]:
        merged_df = pd.merge(merged_df, yearly_df, on="Year", how="inner")

    merged_df = merged_df.dropna().copy()
    years_sorted = sorted(merged_df["Year"].unique())
    longest_seq = find_longest_consecutive_sequence(years_sorted)

    if not longest_seq:
        return pd.DataFrame(), []

    common_period_df = merged_df[merged_df["Year"].isin(longest_seq)].copy()
    return common_period_df, longest_seq


def summarize_series(values):
    """Compute the same descriptive statistics used in the original script."""
    series = pd.Series(values, dtype=float)
    stats = {
        "n": series.count(),
        "Min": series.min(),
        "P-25": series.quantile(0.25),
        "Mean": series.mean(),
        "P-75": series.quantile(0.75),
        "Max": series.max(),
        "SD": series.std(),
    }
    stats["CV"] = stats["SD"] / stats["Mean"] if stats["Mean"] != 0 else np.nan
    return stats


def calculate_window_cv_values(
    series_values, cumulative_sum, cumulative_sum_sq, window_size
):
    """Calculate CV values for all windows of a given size."""
    count = len(series_values)

    if window_size <= count:
        window_sums = cumulative_sum[window_size:] - cumulative_sum[:-window_size]
        window_sum_sq = (
            cumulative_sum_sq[window_size:] - cumulative_sum_sq[:-window_size]
        )
        means = window_sums / window_size
        variance = (window_sum_sq - (window_sums**2) / window_size) / (window_size - 1)
        variance = np.maximum(variance, 0)
        std_values = np.sqrt(variance)
        cv_values = np.full(std_values.shape, np.nan, dtype=float)
        valid_means = np.isfinite(means) & (means != 0)
        np.divide(std_values, means, out=cv_values, where=valid_means)
        return cv_values

    full_mean = series_values.mean()
    full_std = pd.Series(series_values).std()
    if full_mean == 0:
        return np.array([np.nan])
    return np.array([full_std / full_mean])


def build_variable_cv_stats(common_period_df, variable):
    """Compute the original row and CV rows for one variable without wide tables."""
    # Work with NumPy arrays here to avoid building many temporary DataFrame columns.
    series_values = common_period_df[variable].to_numpy(dtype=float)
    count = len(series_values)
    cumulative_sum = np.concatenate(([0.0], np.cumsum(series_values)))
    cumulative_sum_sq = np.concatenate(([0.0], np.cumsum(series_values**2)))

    rows = [
        {
            "nn": 1,
            "Variable": variable,
            **summarize_series(series_values),
        }
    ]

    # For n = 2, 3, ..., compute the CV of every possible window and then summarize it.
    for window_size in range(2, count + 2):
        cv_values = calculate_window_cv_values(
            series_values, cumulative_sum, cumulative_sum_sq, window_size
        )
        rows.append(
            {
                "nn": len(rows) + 1,
                "Variable": variable,
                **summarize_series(cv_values),
            }
        )

    return pd.DataFrame(rows)


def build_combined_cv_dataframe(common_period_df, selected_variables):
    """Build the plotting dataframe with CV statistics for all variables."""
    variable_stats = [
        build_variable_cv_stats(common_period_df, variable)
        for variable in selected_variables
    ]
    return pd.concat(variable_stats, ignore_index=True)


def calculate_effective_errors(combined_df, selected_variables):
    """Estimate the effective error for each variable from CV stabilization."""
    effective_errors = {}

    for variable in selected_variables:
        consecutive_stable_count = 0
        # The first nn is kept when the CV slope stays below the threshold 3 times in a row.
        for i in range(1, len(combined_df)):
            if combined_df.loc[i, "Variable"] == variable:
                slope = abs(combined_df.loc[i, "CV"] - combined_df.loc[i - 1, "CV"])
                if slope < 0.01:
                    consecutive_stable_count += 1
                    if consecutive_stable_count == 1:
                        first_stable_nn = combined_df.loc[i - 1, "nn"]
                    if consecutive_stable_count == 3:
                        effective_errors[variable] = first_stable_nn
                        break
                else:
                    consecutive_stable_count = 0

    return effective_errors


def build_effective_errors_text(effective_errors):
    """Format effective errors for the plot title."""
    return "<b>- Effective Errors: </b>" + "; ".join(
        [f"{var} = {nn}" for var, nn in effective_errors.items()]
    )


def create_plot(
    combined_df, input_file_base_name, min_year, max_year, count, effective_errors
):
    """Create the interactive Plotly figure."""
    effective_errors_text = build_effective_errors_text(effective_errors)
    title_text = (
        f"<b>Coefficient of Variation of CV for {input_file_base_name}</b><br>"
        f"<b>- Common Period:</b> {min_year}-{max_year} ({count} years)<br>"
        f"{effective_errors_text}<br>"
    )

    fig = px.line(
        combined_df[combined_df["nn"] != 1],
        x="nn",
        y="CV",
        color="Variable",
        title=title_text,
    )

    for var, nn in effective_errors.items():
        fig.add_trace(
            go.Scatter(
                x=[nn],
                y=[
                    combined_df[
                        (combined_df["Variable"] == var) & (combined_df["nn"] == nn)
                    ]["CV"].values[0]
                ],
                mode="markers",
                marker=dict(size=10, color="black"),
                name=f"Effective Error ({var}) = {nn}",
            )
        )

    return fig


def save_plot(fig, input_file_base_name, output_plot_path, display_plot=None):
    """Optionally show the figure and save it as an HTML file."""
    if display_plot is None:
        display_plot = show_plots

    if display_plot:
        fig.show()
    pyo.plot(
        fig,
        filename=os.path.join(
            output_plot_path, f"CV_plot_combined_{input_file_base_name}.html"
        ),
        auto_open=False,
    )


def build_summary_row(
    input_file_base_name, min_year, max_year, count, effective_errors
):
    """Create one summary row for the output CSV."""
    summary_row = {
        "gauge_id": input_file_base_name,
        "common_period": f"{min_year}-{max_year}",
        "years": count,
    }

    for column_name, variable in zip(effective_error_columns, variables):
        summary_row[column_name] = effective_errors.get(variable)

    return summary_row


def process_input_file(input_file_name, display_plot=None):
    """Process one CSV file and return its outputs or discard reason."""
    if display_plot is None:
        display_plot = show_plots

    input_file_path = os.path.join(input_folder_path, input_file_name)
    input_file_base_name = os.path.splitext(input_file_name)[0]

    # Sequence for one gauge: read daily data, aggregate to yearly, keep the common
    # period, summarize CV behavior, estimate effective errors, then export outputs.
    df_raw = read_input_csv(input_file_path)

    yearly_dataframes = build_yearly_dataframes(df_raw, variables, variable_agg)
    common_period_df, longest_seq = get_common_period_dataframe(yearly_dataframes)

    if not longest_seq:
        return {
            "status": "discarded",
            "message": (
                f"{input_file_name}: no common consecutive years across selected variables"
            ),
        }

    count = len(common_period_df["Year"])
    min_year = common_period_df["Year"].min()
    max_year = common_period_df["Year"].max()

    combined_df = build_combined_cv_dataframe(common_period_df, variables)
    effective_errors = calculate_effective_errors(combined_df, variables)

    fig = create_plot(
        combined_df,
        input_file_base_name,
        min_year,
        max_year,
        count,
        effective_errors,
    )
    save_plot(fig, input_file_base_name, plot_path, display_plot=display_plot)

    return {
        "status": "used",
        "message": f"{input_file_name}: common period {min_year}-{max_year} ({count} years)",
        "summary_row": build_summary_row(
            input_file_base_name,
            min_year,
            max_year,
            count,
            effective_errors,
        ),
    }


def write_processing_log(output_log_path, used_files, discarded_files):
    """Write the text log with used and discarded CSV files."""
    with open(output_log_path, "w", encoding="utf-8") as log_file:
        log_file.write("Used CSV files\n")
        log_file.write("==============\n")
        if used_files:
            log_file.write("\n".join(used_files))
        else:
            log_file.write("None")

        log_file.write("\n\nDiscarded CSV files\n")
        log_file.write("===================\n")
        if discarded_files:
            log_file.write("\n".join(discarded_files))
        else:
            log_file.write("None")


def write_summary_csv(output_summary_csv_path, summary_rows):
    """Write the summary CSV for all processed gauges."""
    summary_df = pd.DataFrame(
        summary_rows,
        columns=["gauge_id", "common_period", "years", *effective_error_columns],
    )
    summary_df.to_csv(output_summary_csv_path, index=False)


def main():
    """Run the complete station analysis workflow."""
    os.makedirs(plot_path, exist_ok=True)

    used_files = []
    discarded_files = []
    summary_rows = []

    # Process every CSV independently so one discarded station does not stop the batch.
    for input_file_name in sorted(os.listdir(input_folder_path)):
        if not input_file_name.endswith(".csv"):
            continue

        result = process_input_file(input_file_name, display_plot=show_plots)
        if result["status"] == "discarded":
            discarded_files.append(result["message"])
            continue

        used_files.append(result["message"])
        summary_rows.append(result["summary_row"])

    # Write the batch-level outputs after all gauges have been evaluated.
    write_processing_log(log_path, used_files, discarded_files)
    write_summary_csv(summary_csv_path, summary_rows)


if __name__ == "__main__":
    main()
