import os
import sys
from datetime import datetime
import pickle

from pycaret.datasets import get_data
from pycaret.classification import *
import optuna
import merged_iss

data = merged_iss.final_merged_data2
clf = setup(data, target='S3', session_id=123, use_gpu=True)
best_model = compare_models()

tuned_model = tune_model(best_model)

final_model = finalize_model(tuned_model)

predictions = predict_model(final_model, data=data)

evaluate_model(final_model)