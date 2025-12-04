from constants.climate_prediction_dataset_v3_cols import CHANGE_IN_CO2, CHANGE_IN_SEA_LVL, CHANGE_IN_TEMP, CO2, SEA_LVL, TEMP # type: ignore
from constants.data_file_paths import GBDT_SEA_LEVEL_MODEL_FILEPATH_JSON, GBDT_SEA_LEVEL_MODEL_FILEPATH_ONNX, GBDT_TEMPERATURE_MODEL_FILEPATH_JSON, GBDT_TEMPERATURE_MODEL_FILEPATH_ONNX # type: ignore
from models.climate_prediction.gbdt import run_gbdt # type: ignore
import numpy as np
from pandas import DataFrame
from xgboost import Booster # type: ignore

import onnxruntime as rt # type: ignore
from onnxmltools.convert.common.data_types import FloatTensorType # type: ignore
# from skl2onnx import convert_sklearn, to_onnx, update_registered_converter
# from skl2onnx.common.shape_calculator import (
#     calculate_linear_classifier_output_shapes,
#     calculate_linear_regressor_output_shapes,
# )
# from skl2onnx.convert import may_switch_bases_classes_order
# from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
from onnxmltools.convert import convert_xgboost as convert_xgboost_booster # type: ignore
from sklearn.metrics import mean_absolute_error # type: ignore
import onnx  # type: ignore
from onnx import ModelProto

COLUMNS = [
  CO2,
  TEMP,
  SEA_LVL,
  CHANGE_IN_CO2,
  CHANGE_IN_TEMP,
  CHANGE_IN_SEA_LVL,
]

def create_gbdt_model_onnx_fmt():
  result = run_gbdt()

  models:     tuple[Booster, Booster] = result[0]
  feat_train: DataFrame               = result[1]

  temp_model:       Booster = models[0]
  sea_lvl_model:    Booster = models[1]

  # serialize each model and its learned parameters into json, and store in output directory
  temp_model.save_model(GBDT_TEMPERATURE_MODEL_FILEPATH_JSON)
  sea_lvl_model.save_model(GBDT_SEA_LEVEL_MODEL_FILEPATH_JSON)

  # convert each model to onnx format, and store in output directory
  convert_booster_to_onnx_format_and_save(temp_model, feat_train, GBDT_TEMPERATURE_MODEL_FILEPATH_ONNX)
  convert_booster_to_onnx_format_and_save(sea_lvl_model, feat_train, GBDT_SEA_LEVEL_MODEL_FILEPATH_ONNX)

# Booster -> onnx format
# From: https://onnx.ai/sklearn-onnx/auto_tutorial/plot_gexternal_xgboost.html#same-with-a-booster
def convert_booster_to_onnx_format_and_save(bst: Booster, feat_train: DataFrame, onnx_model_output_path: str):

  print(feat_train)

  # ensure that the features are in the same order as they were for training
  features = feat_train[COLUMNS].to_numpy(dtype=np.float32, copy=False)
  num_features = features.shape[1]

  # rename feature names to f0..f{n-1} to avoid an error requiring a label name pattern f%d
  bst.feature_names = [f"f{i}" for i in range(num_features)]

  initial_type = [("float_input", FloatTensorType([None, feat_train.shape[1]]))]

  # model format that the gbdt Booster model will be converted to
  onnx_model: ModelProto

  try:
    onnx_model = convert_xgboost_booster(bst, "name", initial_types=initial_type)
  except AssertionError as e:
    raise AssertionError("XGBoost is too recent or onnxmltools too old.", e)

  sess = rt.InferenceSession(
      onnx_model.SerializeToString(), providers=["CPUExecutionProvider"]
  )
  input_name = sess.get_inputs()[0].name # type: ignore
  label_name = sess.get_outputs()[0].name # type: ignore
  preds_onx = sess.run([label_name], {input_name: features})[0] # type: ignore
  preds_onx = np.asarray(preds_onx).reshape(-1)

  # ====== GPT5 generated the below code ======

  # It compares the predictions of the model converted to onnx format to the predictions of the original gbdt Booster, 
  # to confirm that the model was converted correctly

  # Compare with native XGBoost predictions on the same features
  import xgboost as xgb  # type: ignore
  # Build a DMatrix with the same feature names as used during conversion
  features_df = DataFrame(features, columns=bst.feature_names)
  dmatrix = xgb.DMatrix(features_df, feature_names=bst.feature_names)
  preds_xgb = bst.predict(dmatrix)
  preds_xgb = np.asarray(preds_xgb).reshape(-1)

  # Ensure shapes line up
  assert preds_onx.shape == preds_xgb.shape, f"Shape mismatch: onnx {preds_onx.shape} vs xgb {preds_xgb.shape}"

  mae = mean_absolute_error(preds_xgb, preds_onx)
  max_abs_diff = float(np.max(np.abs(preds_xgb - preds_onx)))
  print("ONNX conversion validation:")
  print({
  "samples": int(preds_onx.shape[0]),
  "mae": float(mae),
  "max_abs_diff": max_abs_diff,
  })

  # Basic tolerance check (tree models are typically bit-exact or extremely close)
  TOL = 1e-5
  if max_abs_diff > TOL:
    print(f"WARNING: Max abs diff {max_abs_diff:.6g} exceeds tolerance {TOL}")
  else:
    print("ONNX predictions closely match XGBoost predictions (within tolerance).")

  # === save onnx model to file ===
  onnx.save(onnx_model, onnx_model_output_path) # type: ignore


def main(): 
  create_gbdt_model_onnx_fmt()


if __name__ == "__main__":
  main()
