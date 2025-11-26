"""Main file of second iteration of climate prediction model using gbdt."""

from features.climate_prediction.constants.climate_prediction_dataset_cols import FUTURE_GMSL_TARGET_COL_NAME
# FUTURE_TEMP_TARGET_COL_NAME
from features.climate_prediction.models.gbdt.gbdt_test import test_gbdt_model # type: ignore
from features.climate_prediction.models.gbdt.gbdt_train_and_validate import train_and_validate_gbdt_model # type: ignore
from features.climate_prediction.data_processing.process_and_train_test_split import process_and_train_test_split
from sklearn.pipeline import Pipeline

def main():
  process_and_train_test_split()

  # # temperature ml model
  # gbdt_model: Pipeline = train_and_validate_gbdt_model(FUTURE_TEMP_TARGET_COL_NAME)
  # test_gbdt_model(gbdt_model, FUTURE_TEMP_TARGET_COL_NAME)

  # print("\n\n\n==============\n\n\n")

  # gmsl ml model
  gbdt_model: Pipeline = train_and_validate_gbdt_model(FUTURE_GMSL_TARGET_COL_NAME)
  test_gbdt_model(gbdt_model, FUTURE_GMSL_TARGET_COL_NAME)

if __name__ == "__main__":
  main()
