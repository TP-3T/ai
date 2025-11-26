"""Main file of second iteration of climate prediction model using gbdt."""

from features.climate_prediction.models.gbdt.gbdt_test import test_gbdt_model # type: ignore
from features.climate_prediction.models.gbdt.gbdt_train_and_validate import train_and_validate_gbdt_model # type: ignore
from features.climate_prediction.data_processing.process_and_train_test_split import process_and_train_test_split
from sklearn.pipeline import Pipeline

def main():
  process_and_train_test_split()
  gbdt_model: Pipeline = train_and_validate_gbdt_model()
  test_gbdt_model(gbdt_model)

if __name__ == "__main__":
  main()
