
from features.climate_prediction.constants.climate_prediction_dataset_cols import FUTURE_GMSL_TARGET_COL_NAME
from features.climate_prediction.data_processing.separate_features_and_targets import separate_features_and_targets_from_climate_dataset
import numpy as np
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from skl2onnx import to_onnx, update_registered_converter
from skl2onnx.common.shape_calculator import (
    calculate_linear_regressor_output_shapes,
)
from skl2onnx.convert import may_switch_bases_classes_order
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
import onnxruntime as onnxrt
from features.climate_prediction.constants.data_file_paths import GBDT_MODEL_FILE_NAME, FEATURE_MAPPING_FILE_NAME
import json

def convert_feature_names_to_fx(dataset: DataFrame, save_mapping_to_json=False) -> tuple[DataFrame, dict[str, str]]:
    """
    Convert the feature names in the dataset into formats that are compatible for to_onnx function.
    Compatible format names are f<number>, where number is the current feature num.
    Creates a mapping between feature names and their f<number> label.
    """
    original_cols = dataset.columns.to_list()
    fx_cols = [f"f{i}" for i in range(len(original_cols))]
    
    # Create mapping and store in a json file (the unity app AI module will use this for mapping)
    mapping = dict (zip(original_cols, fx_cols) )
    
    if save_mapping_to_json:
        with open(FEATURE_MAPPING_FILE_NAME, 'w') as json_file:
            json.dump(mapping, json_file)

    return dataset.rename(columns=mapping), mapping

# Source code referenced from:
# https://onnx.ai/sklearn-onnx/auto_tutorial/plot_gexternal_xgboost.html#same-example-with-xgbregressor
def export_to_onnx(model: Pipeline, training_climate_dataset: DataFrame, testing_climate_dataset: DataFrame):
    """
    Export an XGBRegressor sci-kit learn model into ONNX format.
    """

    (train_features, _) = separate_features_and_targets_from_climate_dataset(training_climate_dataset, FUTURE_GMSL_TARGET_COL_NAME)
    (test_features, _) = separate_features_and_targets_from_climate_dataset(testing_climate_dataset, FUTURE_GMSL_TARGET_COL_NAME)

    train_feat_fx, mapping = convert_feature_names_to_fx(train_features, save_mapping_to_json=True)
    test_feat_fx, _ = convert_feature_names_to_fx(test_features)

    update_registered_converter(
        XGBRegressor,
        "XGBoostXGBRegressor",
        calculate_linear_regressor_output_shapes,
        convert_xgboost,
    )
    
    # with may_switch_bases_classes_order(XGBRegressor):
    print("gbdt Model in XGBRegressor format - predictions:", model.predict(test_features[:5]))

    # change the fx names to be f<number> in the model object itself
    fx_names = [feat_name_formatted for feat_name_formatted in mapping.values()]

    model_wrapper: XGBRegressor = None

    # this gets the step in the Pipeline object that is the actual model training result
    for _, step in model.steps:
        if isinstance(step, XGBRegressor):
            model_wrapper = step

    # get the underlying xgboost gbdt class from the sci-kit learn wrapper
    model_fx_names = model_wrapper.get_booster()

    model_fx_names.feature_names = fx_names

    model_onnx = to_onnx(
        model, 
        train_feat_fx.to_numpy(dtype=np.float32), # converts the multi dim. dataframe to a numpy array equivalent with float32 types (no strings)
        target_opset={"": 12, "ai.onnx.ml": 2}
    )

    sess = onnxrt.InferenceSession(model_onnx.SerializeToString(), providers=["CPUExecutionProvider"])
    pred_onx = sess.run(None, {"X": test_feat_fx[:5].to_numpy(dtype=np.float32)})
    print("gbdt Model in ONNX format - predictions:", pred_onx[0].ravel())

    # If both model outputs match, the conversion was successful

    with open(GBDT_MODEL_FILE_NAME, "wb") as f:
        f.write(model_onnx.SerializeToString())
