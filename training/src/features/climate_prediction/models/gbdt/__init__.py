"""Main file of second iteration of climate prediction model using gbdt."""

from features.climate_prediction.constants.climate_prediction_dataset_cols import FUTURE_GMSL_TARGET_COL_NAME
# FUTURE_TEMP_TARGET_COL_NAME
from features.climate_prediction.models.gbdt.gbdt_export_to_onnx import export_to_onnx
from features.climate_prediction.models.gbdt.gbdt_test import test_gbdt_model # type: ignore
from features.climate_prediction.models.gbdt.gbdt_train_and_validate import train_and_validate_gbdt_model # type: ignore
from features.climate_prediction.data_processing.process_and_train_test_split import process_and_train_test_split
from sklearn.pipeline import Pipeline

def main():
  # preprocess and split dataset into training and testing datasets
  (training_climate_dataset, testing_climate_dataset) = process_and_train_test_split()
  
  # train the model on training set
  gbdt_model: Pipeline = train_and_validate_gbdt_model(FUTURE_GMSL_TARGET_COL_NAME)
  
  # test the model on testing set
  test_gbdt_model(gbdt_model, FUTURE_GMSL_TARGET_COL_NAME)

  # export the model to onnx format (file can be ran in Unity app using onnxruntime package)
  export_to_onnx(
    gbdt_model, 
    training_climate_dataset=training_climate_dataset,
    testing_climate_dataset=testing_climate_dataset
  )

if __name__ == "__main__":
  main()
