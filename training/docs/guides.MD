# Guides and References

Makefile:
https://earthly.dev/blog/python-makefile/

Project structure template
https://cookiecutter-data-science.drivendata.org/#with-pipx-recommended

Guide to the gradient boosting algorithm
https://www.datacamp.com/tutorial/guide-to-the-gradient-boosting-algorithm

xgboost in Python
https://www.datacamp.com/tutorial/xgboost-in-python

xgboost ML package article explanation (only the referenced sections that explain some of the parameters, not all of the math stuff)
https://arxiv.org/pdf/1603.02754

xgboost for predicting multiple target variables
https://xgboosting.com/xgboost-for-multiple-output-regression-with-multi_strategy/

xgboost save model
https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html

onnxmltools to convert model to onnx format
https://onnx.ai/sklearn-onnx/auto_tutorial/plot_gexternal_xgboost.html
https://github.com/onnx/onnxmltools

--------

Cross validation algorithm for time series data (like our climate dataset)
Cross validation with random sampling/shuffling does not work for time series data (see design doc)
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html

Determining the most optimal hyperparameters for an ML model, by testing every combination of them
sci-kit learn has a package called "GridSearchCV", which uses brute force to determine the best possible combination of hyperparameters
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

Dataset normalization (for the polynomial regression model)
https://www.datacamp.com/tutorial/normalization-in-machine-learning
