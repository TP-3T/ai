# Machine Learning (for supervised learning and prediction):
#   - Training data used is a set of multiple features (X) and one target (Y) (inputs and expected output)
#   - Each row in a dataset with the features X and the target Y make up one data point
#   - Overview: A machine learning model learns from training data points to generalize for unseen ones
#     - The model in training makes a prediction of what the target should be
#     - After making a prediction of the target, the model compares its prediction with the expected target value
#     - Based on the amount of error it made, it adjusts its "parameters" (basically the model's setting values) to hopefully make a better prediction on the next data point
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

# Train/Test Split and Generalization:

# Bias-Variance Tradeoff:

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
#   - Each tree uses the features and prediction error, to predict what corrections are needed to be able to 
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
#       - Often simply the mean of the target values.
#       - This gives a baseline model for all data points.

# 2. Calculate the pseudo-residuals
#       - Pseudo-residual = Observed value - Current prediction
#       - These represent how much correction each data point needs.

# 3. Build a weak learner (decision tree) that will predict these pseudo-residuals (for indirectly predicting the target vlaue)
#       - Each split from a decision tree node is based on features (ex. CO2 < 301.2)
#       - Each leaf node should be an error correction amount to add to the initial prediction, which results in the predicted target value

# 4. Update the prediction
#       - New prediction of the target = Previous prediction + (learning_rate * tree_leaf_value)

# 5. Repeat steps 2-4 with updated pseudo-residuals
#       - Each new tree fixes errors made by the previous trees.
#       - The model gradually improves by correcting mistakes.

# =============================================================================

# Hyperparameters - model "settings":
# - Properties of the ensemble DTs and gradient boosting algorithm that are set before training
# - Determined and "tuned" by ML trainer
#     - Objective - the loss function to use for the algorithm
#     - Learning Rate - controls the contribution of each DT - adjusts the shrinkage factor 
#           - Ex. smaller values means each DT prediction has lower contribution to the final result in the ensemble
#     - Number of trees - number of boosting rounds = number of trees to build in the ensemble
#     - Max tree depth - the number of levels in each DT (including the leaf level)
#     - Minimum number of samples per leaf (or min child weight)
#     - Subsampling rate - proportion of rows used to train each tree 
#           - Ex. Each DT is trained on a randomly sampled 70% of the rows (a random 70% of the rows, not the same 70% for every DT)
#     - Feature sampling rate - proportion of columns (features) used to train each tree
#           - Ex. Each DT is trained with a randomly sampled 50% of the features (a random 50% of the columns, not the same 50% for every DT)

# objective - Mean Squared Error (MSE) loss function
# learning rate - 
# # trees - early stopping
# max tree depth - 3
# min child weight - 3
# subsampling rate - 0.8
# feature sampling rate - 1.0 (only have two features)

# will use xgboost since sklearn only supports training using CPU resources

# =============================================================================

import pandas as pd
from pandas import DataFrame
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split # type: ignore
import xgboost as xgb
from xgboost import Booster, DMatrix

from constants.data_file_paths import CLIMATE_DATASET_FILENAME # type: ignore

YEAR_COL_NAME:    str = "Year"
SEA_LVL_COL_NAME: str = "Average change in global sea level (mm)"
COL_AXIS_NUM = 1

# Load climate dataset
climate_dataset: DataFrame = pd.read_csv(CLIMATE_DATASET_FILENAME) # type: ignore

# Extract required features and target
# Features: Global avg temperature, global avg CO2
# Target:   Change in sea level
features: DataFrame = climate_dataset.drop(columns=[YEAR_COL_NAME, SEA_LVL_COL_NAME], axis=COL_AXIS_NUM)
target:   DataFrame = climate_dataset.filter(like=SEA_LVL_COL_NAME, axis=COL_AXIS_NUM)

# Split the data into training and testing sets (default - 0.75 train size, 0.25 test size) 
# Reproducible split of data (not random)
feat_train, feat_test, target_train, target_test = train_test_split(features, target, random_state=1) # type: ignore

dtrain_regr: DMatrix = xgb.DMatrix(feat_train, target_train)
dtest_regr:  DMatrix = xgb.DMatrix(feat_test, target_test)

# Set hyperparameters
OBJECTIVE:          str = "reg:squarederror"
LEARNING_RATE:      float = 0.1
NUM_BOOST_ROUNDS:   int = 100 # same as number of trees
MAX_TREE_DEPTH:     int = 3
MIN_CHILD_WEIGHT:   int = 3
SUBSAMPLING_RATE:   float = 0.8
FEAT_SAMPLING_RATE: float = 1.0

params = {
  "objective": OBJECTIVE,
  "device": "cuda", # use gpu of device for training
  "tree_method": "hist",
}

# Validation sets to view model rmse at current point during training
evals = [(dtrain_regr, "train"), (dtest_regr, "validation")]

# Train model
model: Booster = xgb.train(
  params=params,
  dtrain=dtrain_regr,
  num_boost_round=NUM_BOOST_ROUNDS,
  evals=evals
)

# Test model on unseen data
test_data_predictions = model.predict(dtest_regr)

# Compare predictions with the actual data
rmse = root_mean_squared_error(target_test, test_data_predictions)

print(f"RMSE of the model: {rmse:.3f}")
