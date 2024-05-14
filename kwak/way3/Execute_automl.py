import os
import sys
from datetime import datetime
import pickle

from pycaret.datasets import get_data
from pycaret.classification import *
import optuna
import preproc_data

data = preproc_data.merged_data
clf = setup(data, target='S4', session_id=123, use_gpu=True)
# Compare models
best_model = compare_models()

# Create the model (automatically tuned)
tuned_model = tune_model(best_model)

# Finalize the model (including feature selection)
final_model = finalize_model(tuned_model)

# Make predictions
predictions = predict_model(final_model, data=data)

# Evaluate the model
evaluate_model(final_model)