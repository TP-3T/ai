from features.climate_prediction.constants.climate_prediction_dataset_cols import ROW_ID # type: ignore
from pandas import DataFrame # type: ignore

def separate_features_and_targets_from_climate_dataset(climate_dataset: DataFrame, target_col_name: str) -> tuple[DataFrame, DataFrame]:
  """
  Remove row id from the climate prediction dataset, 
  then separates the climate dataset dataframe into two dataframes, 
  one for features and the other for targets. 
  
  :param climate_dataset: The climate dataset.
  :type climate_dataset: DataFrame
  :return: A tuple with two dataframes, the first being the feature set, and the second being the target set.
  :rtype: tuple[DataFrame, DataFrame]
  """
  # remove row id column
  climate_dataset_no_row_ids: DataFrame = climate_dataset.drop(columns=ROW_ID)

  targets_list: list[str] = [target_col_name]

  print(f"Separating features from the target - {target_col_name}")

  climate_dataset_features:   DataFrame = climate_dataset_no_row_ids.drop(columns=targets_list)
  climate_dataset_targets:    DataFrame = climate_dataset_no_row_ids[targets_list]

  return (climate_dataset_features, climate_dataset_targets)
