import os
import sys
import shutil
import random
import warnings
import numpy as np
import pandas as pd
from decimal import *
from preprocess import *
from pathlib import Path
import glob
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn import tree
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import optuna
from optuna.integration import OptunaSearchCV
from optuna.distributions import FloatDistribution, IntDistribution
from optuna.distributions import LogUniformDistribution, UniformDistribution, IntUniformDistribution
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster import hierarchy as hc
import statsmodels.api as sm
import sagemaker_datawrangler



def define_df_updrs(train_clinical_proteins, proteins_wide, target_updrs, updrs_columns_to_drop):
    df_updrs = train_clinical_proteins.dropna(subset=['Q99435', 'Q99674', 'Q99683', 'Q99829', 'Q99832'])
    df_updrs = df_updrs.drop(updrs_columns_to_drop, axis=1)
    df_updrs = df_updrs.fillna(proteins_wide.median())
    df_updrs = df_updrs[df_updrs[target_updrs].notna()]
    return df_updrs

df_updrs_1 = define_df_updrs(train_clinical_proteins, proteins_wide, 'updrs_1', ['visit_id', 'patient_id', 'visit_month', 'updrs_2', 'updrs_3', 'updrs_4'])
df_updrs_2 = define_df_updrs(train_clinical_proteins, proteins_wide, 'updrs_2', ['visit_id', 'patient_id', 'visit_month', 'updrs_1', 'updrs_3', 'updrs_4'])
df_updrs_3 = define_df_updrs(train_clinical_proteins, proteins_wide, 'updrs_3', ['visit_id', 'patient_id', 'visit_month', 'updrs_1', 'updrs_2', 'updrs_4'])
df_updrs_4 = define_df_updrs(train_clinical_proteins, proteins_wide, 'updrs_4', ['visit_id', 'patient_id', 'visit_month', 'updrs_1', 'updrs_2', 'updrs_3'])

def create_updrs_xgb_keep_list(df_updrs, updrs_type):
    updrs_column = f"updrs_{updrs_type}"
    
    X = df_updrs.drop(updrs_column, axis=1)
    y = df_updrs[updrs_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=15
    )

    y_train = y_train.values.reshape(-1, 1)
    y_test = y_test.values.reshape(-1, 1)

    m = XGBRegressor()
    m.fit(X, y)

    feat_dict = {}
    for col, val in sorted(zip(X_train.columns, m.feature_importances_), key=lambda x: x[1], reverse=True):
        feat_dict[col] = val

    updrs_xgb = pd.DataFrame({'Feature': feat_dict.keys(), 'Importance': feat_dict.values()})
    updrs_xgb['Cum_importance'] = updrs_xgb['Importance'].cumsum()

    updrs_xgb_keep = updrs_xgb.drop(updrs_xgb[updrs_xgb.Cum_importance >= 0.80].index)
    updrs_xgb_keep_list = updrs_xgb_keep.Feature.values.tolist()

    return updrs_xgb_keep_list


updrs_1_xgb_keep_list = create_updrs_xgb_keep_list(df_updrs_1, 1)
updrs_2_xgb_keep_list = create_updrs_xgb_keep_list(df_updrs_2, 2)
updrs_3_xgb_keep_list = create_updrs_xgb_keep_list(df_updrs_3, 3)
updrs_4_xgb_keep_list = create_updrs_xgb_keep_list(df_updrs_4, 4)

updrs_xgb_keep_lists = [
    updrs_1_xgb_keep_list,
    updrs_2_xgb_keep_list,
    updrs_3_xgb_keep_list,
    updrs_4_xgb_keep_list,
]

def process_test_proteins_wide(test_proteins, updrs_xgb_keep_list):
    test_proteins_wide = pd.pivot_table(test_proteins, index=['visit_id', 'visit_month', 'patient_id', 'group_key', 'role'], columns='UniProt', values='NPX')
    test_proteins_wide = test_proteins_wide.rename_axis(None, axis=1).reset_index()
    test_proteins_wide = test_proteins_wide.drop(['visit_id', 'visit_month', 'patient_id', 'group_key', 'role'], axis=1)
    test_proteins_wide = test_proteins_wide[test_proteins_wide.columns.intersection(updrs_xgb_keep_list)]
    test_proteins_wide = test_proteins_wide.replace(np.nan, 0)
    return test_proteins_wide

test_proteins_wide_processed = []

for updrs_xgb_keep_list in updrs_xgb_keep_lists:
    test_proteins_wide_processed.append(process_test_proteins_wide(test_proteins, updrs_xgb_keep_list))

    # Printing the processed test_proteins_wide datasets
for i, test_proteins_wide in enumerate(test_proteins_wide_processed):
    print(f"Processed test_proteins_wide for UPDRS {i+1}:")
    print(test_proteins_wide.head())
    print()

def parkinsons_prediction_model(X, y, test_proteins_wide, updrs_index):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=30)

    # Preprocessing steps
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, X.columns)])

    # Create a list of base models
    base_models = [
        ('ridge', Ridge()),
        ('lasso', Lasso()),
        ('elasticnet', ElasticNet()),
        ('knn', KNeighborsRegressor()),
        ('dt', DecisionTreeRegressor()),
        ('rf', RandomForestRegressor()),
        ('xgb', XGBRegressor()),
        ('lgbm', LGBMRegressor()),
        ('catboost', CatBoostRegressor(verbose=0)),
        ('svr', SVR())
    ]

    # Define the parameter search space for each model
    param_space = {
        'model__ridge__alpha': FloatDistribution(1e-3, 1.0, log=True),
        'model__lasso__alpha': FloatDistribution(1e-3, 1.0, log=True),
        'model__elasticnet__alpha': FloatDistribution(1e-3, 1.0, log=True),
        'model__elasticnet__l1_ratio': FloatDistribution(0.0, 1.0),
        'model__knn__n_neighbors': IntDistribution(1, 20),
        'model__dt__max_depth': IntDistribution(1, 20),
        'model__rf__n_estimators': IntDistribution(10, 100),
        'model__rf__max_depth': IntDistribution(1, 20),
        'model__xgb__n_estimators': IntDistribution(10, 100),
        'model__xgb__max_depth': IntDistribution(1, 20),
        'model__lgbm__n_estimators': IntDistribution(10, 100),
        'model__lgbm__max_depth': IntDistribution(1, 20),
        'model__catboost__iterations': IntDistribution(10, 100),
        'model__catboost__depth': IntDistribution(1, 10),
        'model__svr__C': FloatDistribution(1e-3, 1e3, log=True),
        'model__svr__epsilon': FloatDistribution(1e-3, 1e1, log=True),
    }

    # Create the stacking regressor with the base models
    stacking_regressor = StackingRegressor(estimators=base_models, final_estimator=Ridge())

    # Create the pipeline
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                        ('rfe', RFE(RandomForestRegressor(), n_features_to_select=20)),
                        ('model', stacking_regressor)])

    # Use Optuna for model selection and hyperparameter tuning
    optuna_search = OptunaSearchCV(pipe, param_space, n_trials=50, cv=5, scoring='neg_mean_squared_error', random_state=42, n_jobs=-1)

    # Fit the model on the training data
    optuna_search.fit(X_train, y_train)

    # Print the best parameters and score
    print("Best parameters found: ", optuna_search.best_params_)
    print("Best mean squared error: ", -optuna_search.best_score_)

    # Predict on the test data and calculate the mean squared error
    y_pred = optuna_search.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean squared error on the test data: ", mse)

    # Make predictions on the test_proteins_wide dataset
    updrs_array = optuna_search.predict(test_proteins_wide)
    updrs_pred = pd.DataFrame(data=updrs_array, columns=['Score'])
    updrs_pred.to_csv(f'updrs_{updrs_index}_pred.csv')

    return optuna_search.best_estimator_, updrs_pred

def prepare_updrs_data(train_clinical_proteins, proteins_wide, updrs_columns_to_drop, target_updrs):
    df_updrs = train_clinical_proteins.dropna(subset=['Q99435', 'Q99674', 'Q99683', 'Q99829', 'Q99832'])
    df_updrs = df_updrs.drop(updrs_columns_to_drop, axis=1)
    df_updrs = df_updrs.fillna(proteins_wide.median())
    df_updrs = df_updrs[df_updrs[target_updrs].notna()]
    X = df_updrs.drop(target_updrs, axis=1)
    y = df_updrs[target_updrs]
    return X, y

for i in range(1, 5):
    X_updrs, y_updrs = prepare_updrs_data(train_clinical_proteins, proteins_wide, ['visit_id', 'patient_id', 'visit_month'] + [f'updrs_{j}' for j in range(1, 5) if j != i], f'updrs_{i}')
    best_estimator, updrs_pred = parkinsons_prediction_model(X_updrs, y_updrs, test_proteins_wide, i)
    submission.at[4 * i - 1, 'rating'] = updrs_pred.iloc[0, 0]
    submission.at[4 * i + 14, 'rating'] = updrs_pred.iloc[1, 0]

submission
sample_submission=submission[['prediction_id','rating']]
print(sample_submission)
