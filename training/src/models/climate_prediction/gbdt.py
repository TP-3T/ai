"""
Climate prediction model training using the gradient boosted decision tree algorithm.
Contains functions to visualize the climate data, as well as perform the model training, validation and testing.
"""

# (Making notes from research here in case I need to explain the theory in presentation or report)

# Machine Learning (for supervised learning and prediction):
#   - Training data used is a set of multiple features (X) and one target (Y) (inputs and expected output)
#   - Each row in a dataset with the features X and the target Y make up one data point
#   - Overview: A machine learning model learns from training data points to generalize for unseen ones
#     - The model in training makes a prediction of what the target should be
#     - After making a prediction of the target, the model compares its prediction with the expected target value
#     - Based on the amount of error it made, it adjusts its "parameters" (basically the model's setting values) to hopefully make a better prediction on the next prediction
#     - The final parameter values after training is what makes up the machine learning model's ability to predict unforeseen data
#     - Goal is to improve prediction accuracy and minimize the amount of error, by optimizing the model parameters

# Steps to optimize model parameters:
#   1. Calculate and quantify the amount of error (using the loss function)
#   2. Compute the slope of a line that "touches" the loss function at a point - called the gradient 
#   3. Update the model parameters according to the gradient to minimize loss

# Loss function:
#   - A function that quantifies the difference between a model's predictions and actual values
#   - Used to:
#       - Calculate the error (predicted output compared to actual value)
#       - Quantify the amount of error to be able to update parameter values throughout model training
#       - Evaluate the model (compare loss on training, validation, and test datasets)
#   - The same loss function is used to evaluate each prediction during the training process

# Mean Squared Error:
#   - A type of loss function commonly used for regression
#   - Is the sum of squared differences between predicted and actual values
#       - Error (for one data point) = 1/2 x (Observed - Predicted)^2
#   - Purpose of squaring is to prevent negative errors, and emphasize larger errors for the model to correct
#   - Often the error is multiplied by 1/2:     Error = 1/2 x (Observed - Predicted)^2
#   - Purpose of x 1/2 is to produce a simple equation after differentiation of the loss function - result will just be: ( Observed - Predicted )
#   - Differentiate - finding the slope of a line that "touches" the loss function curve at a point
#   - Differentiation is done to produce a value that can be used to adjust model parameters (an algorithm uses the value to calculate the adjustment amounts)

# Train/Test Split and Generalization: #TODO

# Bias-Variance Tradeoff: #TODO

# Gradient Boosting:
#   - An ensemble technique in ML
#   - Does not learn from data independently
#   - It instead combines the predictions of multiple weak learners (models that performs slightly better than randomly guessing)
#     into one more accurate strong learner
#       - An example of weak learner is a decision tree
#   - The strong learner is a combination of many weak learners and is used for the same set of features and target combination (different data points though)

# Gradient boosting algorithm:
#   - Works for tabular data
#   - Supervised learning technique
#   - Can be used for both regression (predicting continous, numerical values) or classification (predicting discrete categories for data - each category being a class label)
#   - Each weak learner is trained using all of the rows of the training data
#   - Each tree uses the features and prediction error, to predict what corrections are needed
#   - Trees don’t predict the final answers. Instead they predict how to correct the errors of previous predictions

# Gradient boosted decision tree:
#   - The Random Forest (RF) technique in ML is an ensemble of decision trees, where each tree is built independently and in parallel.
#       - Then it combines the prediction results at the end
#   - Unlike RF, the Gradient boosted decision tree technique builds trees sequentially.
#       - Because of this, each tree learns from the prediction error of the previous trees in the ensemble, and uses the error when making its own prediction
#   - Uses gradient boosting algorithm

# =============================================================================

# Gradient Boosting Algorithm Steps:
 
# 1. Make an initial prediction
#       - The initial prediction value must be for the target for all of the rows in the dataset
#       - Often simply the mean of the target values
#       - This gives a baseline model for all data points (meaning that it is a benchmark, or a minimum performance level)
#           - Ex. mean(all target values) = 156.2

# 2. Calculate the pseudo-residuals 
#       - Pseudo-residual = Observed value - Current prediction
#       - These represent how much correction each data point needs.
#       - This is done for every row in the dataset
#           - Ex. (for one of the many rows)
#             - Observed value is 117.1
#             - Predicted value is 156.2
#             - Obs - Pred = 117.1 - 156.2 = -39.1

# 3. Build a weak learner (decision tree) that will predict these pseudo-residuals (for indirectly predicting the target vlaue)
#       - Each split from a decision tree node is a binary conditional based on a feature value (ex. CO2 < 301.2)
#       - Each leaf node should be an error correction amount to add to the initial prediction (this amount is one of the pseudo-residuals), which results in the predicted target value
#           - Ex. of ONE leaf node prediction result value in a single DT: -39.1. This result can be added to the initial prediction (the mean of all of the target values) and that is the predicted value.
#           - Ex. 156.2 + (-39.1) = 117.1 -> 117.1 is the indirectly predicted value

# 4. Update the prediction to prevent the model from overfitting
#       - A parameter called the learning rate is used in the calculation of the new prediction to prevent "overfitting" 
#           Overfitting means that the model learns the too much of the dataset patterns, and becomes unable to generalize its learnings to unforeseen data (low bias, high variance)
#       - New prediction of the target = Initial prediction + (learning_rate * DT_leaf_node_value)

# 5. Repeat steps 2-4 with updated pseudo-residuals
#       - Each new tree fixes errors made by the previous trees.
#       - The model gradually improves by correcting mistakes.

# --- During model inference (prediction of real life data): ---
#       - The data from wherever (one "row" with all of the dataset features except the target variable) 
#       - gets input into every the trees in the ensemble, and for each it arrives at a leaf node. 
#       - The leaf node contains the pseudo-residual value, which is the output of that decision tree.
#       - Then all of the the pseudo-residual results from each DT output are added to the initial prediction to produce a predicted value.

# =============================================================================

# Hyperparameters - model "settings":
# - Properties of the ensemble DTs and gradient boosting algorithm that are set before training
# - Determined and "tuned" by ML trainer
#     - Objective - the loss function to use for the algorithm
#     - Learning Rate - controls the contribution of each DT - adjusts the shrinkage factor 
#           - Ex. smaller values means each DT prediction has lower contribution to the final result in the ensemble
#     - Number of trees - number of boosting rounds = number of trees to build in the ensemble
#     - Max tree depth - the number of levels in each DT (including the leaf level)
#     - Minimum number of samples per leaf (or min child weight) - minimum sum of instance weights required in a leaf. Prevents a tree from creating leaves with very few samples.
#     - Subsampling rate - proportion of rows used to train each tree 
#           - Ex. Each DT is trained on a randomly sampled 70% of the rows (a random 70% of the rows, not the same 70% for every DT)
#     - Feature sampling rate - proportion of columns (features) used to train each tree
#           - Ex. Each DT is trained with a randomly sampled 50% of the features (a random 50% of the columns, not the same 50% for every DT)

# will use xgboost since sklearn only supports training using CPU resources

# Dataset normalization not needed for decision trees DT ensembles, 
# since tree-based algorithms (DTs, random forests or GBDTs) are not sensitive to the magnitude of the variables

# =============================================================================

import pandas as pd
from pandas import DataFrame
from pandas import Series
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import r2_score
import xgboost as xgb
from xgboost import Booster, DMatrix
import matplotlib.pyplot as plt

from constants.data_file_paths import CLIMATE_DATASET_FILENAME, GBDT_SEA_LEVEL_MODEL_FILENAME, GBDT_SEA_LEVEL_MODEL_FILENAME_BIN, GBDT_TEMPERATURE_MODEL_FILENAME, GBDT_TEMPERATURE_MODEL_FILENAME_BIN # type: ignore
from constants.climate_prediction_dataset_cols import CO2, FUTURE_SEA_LVL, FUTURE_TEMP, SEA_LVL, TEMP, YEAR, PREDICTED_TEMPERATURE_TUPLE_INDEX, PREDICTED_SEA_LVL_TUPLE_INDEX # type: ignore

COL_AXIS_NUM = 1

TEST_PREDICTION_RESULTS_MSG:      str = "Testing Dataset Prediction Results"
ACTUAL_SEA_LVL:                   str = "Actual Future Sea level (mm)"
ACTUAL_TEMPERATURE:               str = "Actual Future Temperature (deg C)"
PREDICTED_SEA_LVL:                str = "Predicted Sea level (mm)"
PREDICTED_TEMPERATURE:            str = "Predicted Future Temperature (deg C)"

RMSE_TEMP_REPORT_MSG:             str = "RMSE (average error) of future temperature predictions (units - deg C)"
RMSE_SEA_LVL_REPORT_MSG:          str = "RMSE (average error) of future sea level predictions (units - mm)"

COEFF_OF_DET_TEMP_REPORT_MSG:     str = "Model goodness of fit (future temperature)"
COEFF_OF_DET_SEA_LVL_REPORT_MSG:  str = "Model goodness of fit (future sea level)"

# Hyperparameters for training
OBJECTIVE:          str = "reg:squarederror" # use Mean Squared Error (MSE) loss function
LEARNING_RATE:      float = 0.1
NUM_BOOST_ROUNDS:   int = 80 # same as number of trees
MAX_TREE_DEPTH:     int = 3
MIN_CHILD_WEIGHT:   int = 5
SUBSAMPLING_RATE:   float = 0.8
FEAT_SAMPLING_RATE: float = 1.0
TREE_METHOD_SPLIT_ALGORITHM: str = "exact" # what algorithm to use for constructing the individual decision trees - explained in section 3 in the xgboost article
# TREE_OUTPUT_TYPE: str = "multi_output_tree" # multiple targets per decision tree

DEVICE: str = "gpu"



def __load_climate_dataset() -> DataFrame:
  # Load climate dataset
  climate_dataset: DataFrame = pd.read_csv(CLIMATE_DATASET_FILENAME) # type: ignore
  return climate_dataset



def __split_dataset_into_train_test_validation(
      features: DataFrame, 
      target: DataFrame
) -> tuple[DMatrix, DMatrix, list[tuple[DMatrix, str]], DataFrame]:
    # Split the data into training and testing sets (default - 0.75 train size, 0.25 test size) 
    # Reproducible split of data (not random)
    feat_train, feat_test, target_train, target_test = train_test_split(features, target, random_state=1) # type: ignore

    # data is independent vars (expected inputs), label is dependent vars (expected outputs)
    train_data_matrix: DMatrix = xgb.DMatrix(feat_train, label=target_train)
    test_data_matrix:  DMatrix = xgb.DMatrix(feat_test, label=target_test)

    # Validation set to view the model error amount at different points before training ends
    evals = [(train_data_matrix, "train"), (test_data_matrix, "validation")]

    return (train_data_matrix, test_data_matrix, evals, target_test) # type: ignore



def __train_model(
      train_data_matrix: DMatrix, 
      evals: list[tuple[DMatrix, str]], 
      model_hyperparams: dict[str, str]
) -> Booster:
    # Train future temperature prediction model
    model: Booster = xgb.train(
      params=model_hyperparams,
      dtrain=train_data_matrix,
      num_boost_round=NUM_BOOST_ROUNDS,
      evals=evals
    )

    print("")

    return model



def train_validate_and_test_model(climate_dataset: DataFrame) -> tuple[Booster, Booster]:
  """
  Train, validate, and test a regression based ML model for climate prediction. 
  ML technique used for training is the gbdt algorithm.
  The package xgboost is used for the implementation of the above. 
  Dataset splitting and model evaluation is implemented using the sklearn package.
  Returns the trained model.
  """
  # Filter columns for both the feature and target dataframes
  # Features: Current values for:
  #     - Global avg temperature
  #     - Global avg CO2
  #     - Global svg sea level
  # Target: Future values for:
  #     - Global avg temperature
  #     - Global absolute avg sea level
  features:       DataFrame = climate_dataset.drop(columns=[YEAR, FUTURE_TEMP, FUTURE_SEA_LVL], axis=COL_AXIS_NUM)
  temp_target:    DataFrame = climate_dataset.filter(items=[FUTURE_TEMP], axis=COL_AXIS_NUM)
  sea_lvl_target: DataFrame = climate_dataset.filter(items=[FUTURE_SEA_LVL], axis=COL_AXIS_NUM)

  params: dict[str, str] = { # type: ignore
    "device": DEVICE, # use gpu of device for training (will switch to cpu if running device has no compatible gpu)
    "tree_method": TREE_METHOD_SPLIT_ALGORITHM,
    "objective": OBJECTIVE,
    'eta': LEARNING_RATE,
    'max_depth': MAX_TREE_DEPTH,
    'min_child_weight': MIN_CHILD_WEIGHT,
    'subsample': SUBSAMPLING_RATE,
    'colsample_bytree': FEAT_SAMPLING_RATE,
    # 'multi_strategy': TREE_OUTPUT_TYPE
  }

  temp_model_hyperparams:    dict[str, str] = params
  sea_lvl_model_hyperparams: dict[str, str] = params

  train_dmatrix_temp,     test_dmatrix_temp,    evals, future_temp_test_set     = __split_dataset_into_train_test_validation(features, temp_target)
  train_dmatrix_sea_lvl,  test_dmatrix_sea_lvl, evals, future_sea_lvl_test_set = __split_dataset_into_train_test_validation(features, sea_lvl_target)

  future_temp_model:    Booster = __train_model(train_dmatrix_temp, evals, temp_model_hyperparams)
  future_sea_lvl_model: Booster = __train_model(train_dmatrix_sea_lvl, evals, sea_lvl_model_hyperparams)

  # Test model on unseen data (the test data)
    # each is a 1D numpy array
  predicted_future_temps    = future_temp_model.predict(test_dmatrix_temp)
  predicted_future_sea_lvls = future_sea_lvl_model.predict(test_dmatrix_sea_lvl)

  # pred_temps_list:    list[float]  = [prediction_tuple[PREDICTED_TEMPERATURE_TUPLE_INDEX] for prediction_tuple in temp_test_predictions] # type: ignore
  # pred_sea_lvls_list: list[float]  = [prediction_tuple[PREDICTED_SEA_LVL_TUPLE_INDEX] for prediction_tuple in target_test_predictions] # type: ignore

  # predicted_temperatures: Series = pd.Series(pred_temps_list, name=PREDICTED_TEMPERATURE, index=feat_test.index) # type: ignore
  # predicted_sea_lvl:      Series = pd.Series(pred_sea_lvls_list, name=PREDICTED_SEA_LVL, index=feat_test.index) # type: ignore 

  actual_future_temps:    Series = future_temp_test_set[FUTURE_TEMP] # type: ignore
  actual_future_sea_lvls: Series = future_sea_lvl_test_set[FUTURE_SEA_LVL] # type: ignore

    # Create new dataframe with just the predicted and actual test targets
  actual_and_predicted_future_temps_df:    DataFrame = pd.DataFrame(index=future_temp_test_set.index) # type: ignore
  actual_and_predicted_future_sea_lvls_df: DataFrame = pd.DataFrame(index=future_temp_test_set.index) # type: ignore

    # Add the actual and predicted cols - fix the column assignments
  actual_and_predicted_future_temps_df[ACTUAL_TEMPERATURE] = actual_future_temps
  actual_and_predicted_future_temps_df[PREDICTED_TEMPERATURE] = predicted_future_temps

  actual_and_predicted_future_sea_lvls_df[ACTUAL_SEA_LVL] = actual_future_sea_lvls
  actual_and_predicted_future_sea_lvls_df[PREDICTED_SEA_LVL] = predicted_future_sea_lvls

  print(f"\n{TEST_PREDICTION_RESULTS_MSG}:\n")
  print(actual_and_predicted_future_temps_df) # type: ignore
  print(actual_and_predicted_future_sea_lvls_df) # type: ignore

  # Model evaluation

    # Compare predictions with the actual data by calculating the average error - using root mean squared error
  temp_rmse:      float = root_mean_squared_error(actual_future_temps.values, predicted_future_temps) # type: ignore
  sea_level_rmse: float = root_mean_squared_error(actual_future_sea_lvls.values, predicted_future_sea_lvls) # type: ignore

    # Calculate the "goodness of the fit" / coeff of determination of the model (R^2)
    # (how close the model's predicted values are to the actual)
  temp_r2:    float = r2_score(actual_future_temps, predicted_future_temps) # type: ignore
  sea_lvl_r2: float = r2_score(actual_future_sea_lvls, predicted_future_sea_lvls) # type: ignore

  print(f"{RMSE_TEMP_REPORT_MSG}: {temp_rmse:.5f}")
  print(f"{RMSE_SEA_LVL_REPORT_MSG}: {sea_level_rmse:.5f}")
  print(f"{COEFF_OF_DET_TEMP_REPORT_MSG}: {temp_r2:.5f}")
  print(f"{COEFF_OF_DET_SEA_LVL_REPORT_MSG}: {sea_lvl_r2:.5f}")

  return (future_temp_model, future_sea_lvl_model)


def visualize_dataset(climate_dataset: DataFrame):
  """
  Visualize the climate dataset (see the output for the different plots).
  """

  year_col: Series = climate_dataset.loc[:, YEAR]
  temp_col: Series = climate_dataset.loc[:, TEMP]
  sea_lvl_col: Series = climate_dataset.loc[:, SEA_LVL]
  co2_col: Series = climate_dataset.loc[:, CO2]
  # future_sea_lvl_col: Series = climate_dataset.loc[:, FUTURE_SEA_LVL]
  # future_temp_col: Series = climate_dataset.loc[:, FUTURE_TEMP]

  # Create a figure with a 2x2 grid of subplots
  # (figsize adjusts the figure size)
  fig, axs = plt.subplots(4, 4, figsize=(10, 8)) # type: ignore

  # Plot on row 0 col 0
  axs[0, 0].plot(year_col, temp_col, color='blue')
  axs[0, 0].set_title(f"{YEAR} vs {TEMP}")
  axs[0, 0].set_xlabel(YEAR)
  axs[0, 0].set_ylabel(TEMP)

  # Plot on row 0 col 1
  axs[0, 1].plot(year_col, sea_lvl_col, color='red')
  axs[0, 1].set_title(f"{YEAR} vs {SEA_LVL}")
  axs[0, 1].set_xlabel(YEAR)
  axs[0, 1].set_ylabel(SEA_LVL)

  # Plot on row 0 col 2
  axs[0, 2].plot(year_col, co2_col, color='green')
  axs[0, 2].set_title(f"{YEAR} vs {CO2}")
  axs[0, 2].set_xlabel(YEAR)
  axs[0, 2].set_ylabel(CO2)

  #  ------------------------

  # Plot on row 1 col 0
  axs[1, 0].plot(co2_col, temp_col, color='blue')
  axs[1, 0].set_title(f"{CO2} vs {TEMP}")
  axs[1, 0].set_xlabel(CO2)
  axs[1, 0].set_ylabel(TEMP)

  # Plot onrow 2 col 0
  axs[2, 0].plot(co2_col, sea_lvl_col, color='red')
  axs[2, 0].set_title(f"{CO2} vs {SEA_LVL}")
  axs[2, 0].set_xlabel(CO2)
  axs[2, 0].set_ylabel(SEA_LVL)

  # # Plot on row 3 col 0
  axs[3, 0].plot(temp_col, sea_lvl_col, color='green')
  axs[3, 0].set_title(f"{TEMP} vs {SEA_LVL}")
  axs[3, 0].set_xlabel(TEMP)
  axs[3, 0].set_ylabel(SEA_LVL)
  
  # Future temperature and sea levels

  # # # Plot on row 3 col 1
  # axs[3, 1].plot(co2_col, future_temp_col, color='green')
  # axs[3, 1].set_title(f"{CO2} vs {FUTURE_TEMP}")
  # axs[3, 1].set_xlabel(CO2)
  # axs[3, 1].set_ylabel(FUTURE_TEMP)

  #   # # Plot on row 3 col 1
  # axs[3, 2].plot(co2_col, future_sea_lvl_col, color='green')
  # axs[3, 2].set_title(f"{CO2} vs {FUTURE_SEA_LVL}")
  # axs[3, 2].set_xlabel(CO2)
  # axs[3, 2].set_ylabel(FUTURE_SEA_LVL)

  # Adjust layout to prevent overlapping titles/labels
  plt.tight_layout()

  # Display the figure with all the subplots
  plt.show() # type: ignore

def main():
  climate_dataset:  DataFrame = __load_climate_dataset()
  models:           tuple[Booster, Booster] = train_validate_and_test_model(climate_dataset)
  temp_model:       Booster = models[0]
  sea_lvl_model:    Booster = models[1]
  # visualize_dataset(climate_dataset)
  
  # Serialize each model and its learned parameters into json, and store in output directory
  # temp_model.save_model(GBDT_TEMPERATURE_MODEL_FILENAME_BIN)
  # sea_lvl_model.save_model(GBDT_SEA_LEVEL_MODEL_FILENAME_BIN)

if __name__ == "__main__":
  main()
