"""Main file of second iteration of climate prediction model using gbdt."""

from features.climate_prediction.constants.climate_prediction_dataset_cols import FUTURE_GMSL_TARGET_COL_NAME # type: ignore
from features.climate_prediction.data_processing.process_and_train_test_split import process_and_train_test_split # type: ignore
from features.climate_prediction.models.gbdt.gbdt_train_and_validate import train_and_validate_gbdt_model # type: ignore
from features.climate_prediction.models.gbdt.gbdt_test import test_regression_ml_model # type: ignore
# from features.climate_prediction.models.gbdt.gbdt_test import test_gbdt_model_interpolation #type: ignore
# from features.climate_prediction.models.gbdt.gbdt_test import test_multiple_synthetic_points # type: ignore
from features.climate_prediction.models.gbdt.gbdt_export_to_onnx import export_to_onnx # type: ignore
from sklearn.pipeline import Pipeline

def main():
  # preprocess and split dataset into training and testing datasets
  (training_climate_dataset, testing_climate_dataset) = process_and_train_test_split(keep_year=False)
  
  # train the model on training set
  gbdt_model: Pipeline = train_and_validate_gbdt_model(FUTURE_GMSL_TARGET_COL_NAME)
  
  # test the model on testing set
  test_regression_ml_model(gbdt_model, FUTURE_GMSL_TARGET_COL_NAME)

  # test_gbdt_model_interpolation(gbdt_model, FUTURE_GMSL_TARGET_COL_NAME)
  # test_multiple_synthetic_points(gbdt_model, FUTURE_GMSL_TARGET_COL_NAME)

  # export the model to onnx format (file can be ran in Unity app using onnxruntime package)
  export_to_onnx(
    gbdt_model, 
    training_climate_dataset=training_climate_dataset,
    testing_climate_dataset=testing_climate_dataset
  )

if __name__ == "__main__":
  main()
