from features.climate_prediction.constants.climate_prediction_dataset_cols import DELTA_GMSL_AFTER_3M_TO_PREDICT, DELTA_TEMP_AFTER_3M_TO_PREDICT # type: ignore
from pandas import DataFrame # type: ignore

TARGETS_LIST: list[str] = [DELTA_TEMP_AFTER_3M_TO_PREDICT, DELTA_GMSL_AFTER_3M_TO_PREDICT]

def __separate_features_and_targets(climate_dataset: DataFrame) -> tuple[DataFrame, DataFrame]:
  climate_dataset_features: DataFrame = climate_dataset.drop(columns=TARGETS_LIST)
  climate_dataset_targets: DataFrame = climate_dataset[TARGETS_LIST]
  return (climate_dataset_features, climate_dataset_targets)
