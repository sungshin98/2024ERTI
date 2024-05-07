import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

import info_processing as info_p
import survey_processing as survey_p

user_info = info_p.user_info
user_survey = survey_p.user_survey

user_data = pd.merge(user_survey, user_info, on='userId', how='left')
