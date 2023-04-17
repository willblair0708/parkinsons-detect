import os
import sys
import shutil
import random
import warnings
import numpy as np
import pandas as pd
from decimal import *
from pathlib import Path
import glob
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import xgboost as xgb
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster import hierarchy as hc
import sagemaker_datawrangler


def cluster_columns(df, figsize=(10, 6), font_size=12):
    """This function clusters columns using hierarchical clustering and displays a dendrogram.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame with columns to cluster.
    figsize (tuple): A tuple specifying the figure size in inches. Default is (10, 6).
    font_size (int): The font size of the labels on the dendrogram. Default is 12.
    
    Returns:
    None
    """
    corr = np.round(scipy.stats.spearmanr(df).correlation, 4)
    corr_condensed = hc.distance.squareform(1 - corr)
    z = hc.linkage(corr_condensed, method='average')
    fig = plt.figure(figsize=figsize)
    hc.dendrogram(z, labels=df.columns, orientation='left', leaf_font_size=font_size)
    plt.show()


def preprocess_data():
    """This function loads and preprocesses the data needed for Parkinson's disease progression prediction.
    
    Parameters:
    None
    
    Returns:
    train_clinical_drop (pandas.DataFrame): The preprocessed train_clinical_data.csv DataFrame.
    test_clinical (pandas.DataFrame): The test.csv DataFrame.
    train_proteins_role (pandas.DataFrame): The train_proteins.csv DataFrame with a new column 'role'.
    test_proteins_role (pandas.DataFrame): The test_proteins.csv DataFrame with a new column 'role'.
    train_peptides_role (pandas.DataFrame): The train_peptides.csv DataFrame with a new column 'role'.
    test_peptides_role (pandas.DataFrame): The test_peptides.csv DataFrame with a new column 'role'.
    proteins_wide (pandas.DataFrame): The wide-format DataFrame of train_proteins with missing values imputed.
    """
    submission = pd.read_csv('../data/amp-parkinsons-disease-progression-prediction/example_test_files/sample_submission.csv')
    test_clinical = pd.read_csv('../data/amp-parkinsons-disease-progression-prediction/example_test_files/test.csv')
    train_clinical = pd.read_csv('../data/amp-parkinsons-disease-progression-prediction/train_clinical_data.csv')

    train_clinical_drop = train_clinical.drop('upd23b_clinical_state_on_medication', axis=1)

    test_proteins = pd.read_csv('../data/amp-parkinsons-disease-progression-prediction/example_test_files/test_proteins.csv')
    test_proteins_role=test_proteins
    test_proteins_role[['role']]='Test'

    train_proteins = pd.read_csv('../data/amp-parkinsons-disease-progression-prediction/train_proteins.csv')
    train_proteins_role=train_proteins
    train_proteins_role['group_key']=0
    train_proteins_role['role']='Train'
    train_proteins_role['visit_month'] = train_proteins_role['visit_month'].astype(str)
    train_proteins_role['patient_id'] = train_proteins_role['patient_id'].astype(str)

    test_peptides = pd.read_csv('../data/amp-parkinsons-disease-progression-prediction/example_test_files/test_peptides.csv')
    test_peptides_role = test_peptides.assign(role='Test')

    train_peptides = pd.read_csv('../data/amp-parkinsons-disease-progression-prediction/train_peptides.csv')
    train_peptides_role = train_peptides.assign(role='Train')

    train_clinical.loc[train_clinical['updrs_1'].between(0, 10, 'both'), '1 Dl (non-m)'] = 'Mild'
    train_clinical.loc[train_clinical['updrs_1'].between(10, 21, 'right'), '1 Dl (non-m)'] = 'Moderate'
    train_clinical.loc[train_clinical['updrs_1'].between(21, 52, 'right'), '1 Dl (non-m)'] = 'Severe'

    train_clinical.loc[train_clinical['updrs_2'].between(0, 12, 'both'), '2 Dl (mtr)'] = 'Mild'
    train_clinical.loc[train_clinical['updrs_2'].between(12, 29, 'right'), '2 Dl (mtr)'] = 'Moderate'
    train_clinical.loc[train_clinical['updrs_2'].between(29, 52, 'right'), '2 Dl (mtr)'] = 'Severe'

    train_clinical.loc[train_clinical['updrs_3'].between(0, 32, 'both'), '3 Mtr exam'] = 'Mild'
    train_clinical.loc[train_clinical['updrs_3'].between(32, 58, 'right'), '3 Mtr exam'] = 'Moderate'
    train_clinical.loc[train_clinical['updrs_3'].between(58, 132, 'right'), '3 Mtr exam'] = 'Severe'

    train_clinical.loc[train_clinical['updrs_4'].between(0, 4, 'both'), '4 Mtr comp'] = 'Mild'
    train_clinical.loc[train_clinical['updrs_4'].between(4, 12, 'right'), '4 Mtr comp'] = 'Moderate'
    train_clinical.loc[train_clinical['updrs_4'].between(12, 24, 'right'), '4 Mtr comp'] = 'Severe'

    train_clinical['counter'] = 1

    train_clinical = train_clinical.copy()
    train_clinical['updrs_sum'] = train_clinical[['updrs_1', 'updrs_2', 'updrs_3', 'updrs_4']].sum(axis=1)
    train_clinical.sort_values(['patient_id', 'visit_month'], inplace=True)
    train_clinical['diffs_1'] = train_clinical.groupby(['patient_id'])['updrs_1'].transform(lambda x: x.diff())
    train_clinical['diffs_2'] = train_clinical.groupby(['patient_id'])['updrs_2'].transform(lambda x: x.diff())
    train_clinical['diffs_3'] = train_clinical.groupby(['patient_id'])['updrs_3'].transform(lambda x: x.diff())
    train_clinical['diffs_4'] = train_clinical.groupby(['patient_id'])['updrs_4'].transform(lambda x: x.diff())
    train_clinical['diffs_sum'] = train_clinical.groupby(['patient_id'])['updrs_sum'].transform(lambda x: x.diff())

    train_clinical = train_clinical[['visit_id', 'patient_id', 'visit_month',
                                     'updrs_1', '1 Dl (non-m)', 'diffs_1',
                                     'updrs_2', '2 Dl (mtr)', 'diffs_2',
                                     'updrs_3', '3 Mtr exam', 'diffs_3',
                                     'updrs_4', '4 Mtr comp', 'diffs_4',
                                     'updrs_sum', 'diffs_sum',
                                     'counter', 'upd23b_clinical_state_on_medication']]

    proteins_wide = pd.pivot(train_proteins, index=['visit_id', 'visit_month', 'patient_id', 'group_key'], columns='UniProt', values='NPX')
    proteins_wide = proteins_wide.rename_axis(None, axis=1).reset_index()
    proteins_wide = proteins_wide.replace('Train', np.nan)
    proteins_wide = proteins_wide.fillna(proteins_wide.median())
    proteins_wide = proteins_wide.drop(['group_key'], axis=1)

    train_clinical_drop['visit_month'] = train_clinical_drop['visit_month'].astype(str)
    train_clinical_drop['patient_id'] = train_clinical_drop['patient_id'].astype(str)

    train_clinical_proteins = pd.merge(train_clinical_drop, proteins_wide, how='left', on=['visit_id', 'patient_id', 'visit_month'])

    return train_clinical_drop, test_clinical, train_proteins_role, test_proteins_role, train_peptides_role, test_peptides_role, proteins_wide
