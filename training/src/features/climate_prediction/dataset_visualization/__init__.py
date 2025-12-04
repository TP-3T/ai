"""
Module for visualizing plots of the main features in the full time-series 
climate dataset, as well as the training and testing sets.
"""

import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame

# The below code is ai generated, 
# just so we could visualize the main features from the full time series dataset, 
# as well as from the training and testing set.

# Also for slide images in the final presentation

from features.climate_prediction.constants.climate_prediction_dataset_cols import (  # type: ignore
    ABSOLUTE_GMSL_MM_RELATIVE_TO_JAN_1950,
    CO2_PPM,
    MONTHLY_SEA_SURFACE_TEMP_ANOMALY,
    MONTHLY_SEA_SURFACE_TEMP_ANOMALY_LOWER_BOUND,
    MONTHLY_SEA_SURFACE_TEMP_ANOMALY_UPPER_BOUND,
    ROW_ID,
    TEMP_DEG_C,
    YEAR,
)
from features.climate_prediction.constants.data_file_paths import (  # type: ignore
    TESTING_CLIMATE_DATASET_FILEPATH,
    TRAINING_CLIMATE_DATASET_FILEPATH,
)
from features.climate_prediction.constants.other import CSV_ENCODING  # type: ignore
from features.climate_prediction.data_processing.process_and_train_test_split import (  # type: ignore
    load_interim_climate_dataset,
)

MAJOR_FEATURE_COLUMNS: list[str] = [
    CO2_PPM,
    TEMP_DEG_C,
    ABSOLUTE_GMSL_MM_RELATIVE_TO_JAN_1950,
    MONTHLY_SEA_SURFACE_TEMP_ANOMALY,
    MONTHLY_SEA_SURFACE_TEMP_ANOMALY_LOWER_BOUND,
    MONTHLY_SEA_SURFACE_TEMP_ANOMALY_UPPER_BOUND,
]


def _prepare_dataset_for_visualization(dataset: DataFrame) -> tuple[pd.Series, DataFrame]:
    dataset_no_row_id: DataFrame = dataset.drop(columns=[ROW_ID], errors="ignore")
    dataset_sorted: DataFrame = dataset_no_row_id.sort_values(by=YEAR)

    parsed_year = pd.to_datetime(dataset_sorted[YEAR], errors="coerce")
    x_axis = parsed_year if not parsed_year.isna().any() else dataset_sorted[YEAR]

    return x_axis, dataset_sorted


def _render_feature_time_series(dataset: DataFrame, title: str) -> None:
    x_axis, dataset_sorted = _prepare_dataset_for_visualization(dataset)

    features_to_plot = [feature for feature in MAJOR_FEATURE_COLUMNS if feature in dataset_sorted.columns]
    if not features_to_plot:
        raise ValueError("No major climate features available for visualization.")

    plt.style.use("seaborn-v0_8-whitegrid")

    figure, axes = plt.subplots(  # type: ignore
        len(features_to_plot),
        1,
        figsize=(14, 3.5 * len(features_to_plot)),
        sharex=True,
    )

    if len(features_to_plot) == 1:
        axes = [axes]

    for axis, feature in zip(axes, features_to_plot):
        axis.plot(x_axis, dataset_sorted[feature], linewidth=2.0, color="#1f77b4")
        axis.set_title(feature, fontsize=11, pad=8)
        axis.set_ylabel(feature, fontsize=9)
        axis.grid(alpha=0.35)
        axis.set_xlabel(YEAR, fontsize=10)
        axis.tick_params(axis="x", labelbottom=True, labelrotation=0, labelsize=8)

    figure.suptitle(title, fontsize=16, fontweight="bold")  # type: ignore
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()  # type: ignore


def visualize_full_climate_dataset() -> None:
    """Plot the full interim climate dataset before the chronological split."""
    climate_dataset: DataFrame = load_interim_climate_dataset()
    _render_feature_time_series(climate_dataset, "Climate Dataset Trends (Full Dataset)")


def visualize_training_climate_dataset() -> None:
    """Plot the chronologically first portion used for training."""
    training_dataset: DataFrame = pd.read_csv(TRAINING_CLIMATE_DATASET_FILEPATH, encoding=CSV_ENCODING) # type: ignore
    _render_feature_time_series(training_dataset, "Climate Dataset Trends (Training Set)")


def visualize_testing_climate_dataset() -> None:
    """Plot the chronologically last portion used for testing."""
    testing_dataset: DataFrame = pd.read_csv(TESTING_CLIMATE_DATASET_FILEPATH, encoding=CSV_ENCODING) # type: ignore
    _render_feature_time_series(testing_dataset, "Climate Dataset Trends (Testing Set)")


if __name__ == "__main__":
    visualize_full_climate_dataset()
    # visualize_training_climate_dataset()
    # visualize_testing_climate_dataset()
