from features.climate_prediction.constants.climate_prediction_dataset_cols import GMSL_AFTER_3M_TO_PREDICT, TEMP_AFTER_3M_TO_PREDICT, ROW_ID # type: ignore
from pandas import DataFrame # type: ignore

# TARGETS_LIST: list[str] = [DELTA_TEMP_AFTER_3M_TO_PREDICT, DELTA_GMSL_AFTER_3M_TO_PREDICT]
TARGETS_LIST: list[str] = [GMSL_AFTER_3M_TO_PREDICT]
# TODO: train 2 models, one for temp, the other for gmsl

def separate_features_and_targets_from_climate_dataset(climate_dataset: DataFrame) -> tuple[DataFrame, DataFrame]:
  """
  Removes row id from the climate prediction dataset, the separates the climate dataset dataframe into two dataframes, one for features and the other for targets. 
  
  :param climate_dataset: The climate dataset.
  :type climate_dataset: DataFrame
  :return: A tuple with two dataframes, the first being the feature set, and the second being the target set.
  :rtype: tuple[DataFrame, DataFrame]
  """
  # remove row id column
  climate_dataset_no_row_ids: DataFrame = climate_dataset.drop(columns=ROW_ID)

  climate_dataset_features:   DataFrame = climate_dataset_no_row_ids.drop(columns=TARGETS_LIST)
  climate_dataset_targets:    DataFrame = climate_dataset_no_row_ids[TARGETS_LIST]

  return (climate_dataset_features, climate_dataset_targets)
