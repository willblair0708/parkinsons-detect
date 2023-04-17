import os
import sys
import random
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster import hierarchy as hc
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

sys.path.append('../data/amp-parkinsons-disease-progression-prediction/')
import amp_pd_peptide

def extract_prediction_info(prediction_id):
    patient_id, visit_month, _, target_name, _, plus_month = prediction_id.split('_')
    return int(patient_id), int(visit_month), f'updrs_{target_name}', int(plus_month)

def update_submission(submission, target_to_prediction):
    submission[['patient_id', 'visit_month', 'target_name', 'plus_month']] = submission['prediction_id'].apply(
        lambda x: pd.Series(extract_prediction_info(x)))

    submission['pred_month'] = submission['visit_month'] + submission['plus_month']
    
    for i in range(1, 5):
        target = f'updrs_{i}'
        mask_target = submission['target_name'] == target
        submission.loc[mask_target, 'rating'] = target_to_prediction[target]

    return submission[['prediction_id', 'rating']]

amp_pd_peptide.make_env.func_dict['__called__'] = False
env = amp_pd_peptide.make_env()   # initialize the environment
iter_test = env.iter_test()    # an iterator which loops over the test files

target_to_prediction = {
    'updrs_1': 3.71, 
    'updrs_2': 5.70, 
    'updrs_3': 20.68, 
    'updrs_4': 2.04
}

iteration_to_data = {}
for iteration, (test_clinical, test_peptides, test_proteins, sample_submission) in enumerate(iter_test):
    iteration_to_data[iteration] = {
        'test_clinical': test_clinical,
        'test_peptides': test_peptides,
        'test_proteins': test_proteins,
        'sample_submission': sample_submission
    }
    
    display(test_clinical.head())
    display(test_peptides.head())
    display(test_proteins.head())
    display(sample_submission.head())

    updated_submission = update_submission(sample_submission, target_to_prediction)
    
    env.predict(updated_submission)
