import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the datasets
train_clinical_data = pd.read_csv('../data/amp-parkinsons-disease-progression-prediction/train_clinical_data.csv')
train_proteins_data = pd.read_csv('../data/amp-parkinsons-disease-progression-prediction/train_proteins.csv')

# Preprocessing steps
# Preprocessing steps continued...
X = train_proteins_data.drop(columns=['visit_id', 'visit_month', 'patient_id', 'UniProt'])
y = train_clinical_data['updrs_3']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Modeling and evaluation
# Modeling steps
linear_model = LinearRegression()
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)

# We will utilize cross-validation for both models
linear_cv_scores = cross_val_score(linear_model, X_train, y_train, cv=5)
random_forest_cv_scores = cross_val_score(random_forest_model, X_train, y_train, cv=5)

print(f"Linear Regression cross-validation mean score: {linear_cv_scores.mean():.4f}")
print(f"Random Forest cross-validation mean score: {random_forest_cv_scores.mean():.4f}")

# Model selection and evaluation
# Evaluate the model on the validation set
def evaluate(model, X_val, y_val):
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    return mse, mae, r2

# Fit the models on the training set
linear_model.fit(X_train, y_train)
random_forest_model.fit(X_train, y_train)

# Calculate evaluation metrics for both models
linear_mse, linear_mae, linear_r2 = evaluate(linear_model, X_val, y_val)
random_forest_mse, random_forest_mae, random_forest_r2 = evaluate(random_forest_model, X_val, y_val)

print(f"Linear Regression - MSE: {linear_mse:.4f}, MAE: {linear_mae:.4f}, R2: {linear_r2:.4f}")
print(f"Random Forest - MSE: {random_forest_mse:.4f}, MAE: {random_forest_mae:.4f}, R2: {random_forest_r2:.4f}")

# Model optimization and fine-tuning

from sklearn.model_selection import GridSearchCV

# Set the hyperparameters for the grid search
random_forest_params = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 30, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=random_forest_model, param_grid=random_forest_params, cv=5, n_jobs=-1)

# Perform the grid search
grid_search.fit(X_train, y_train)

# Get the best parameters from the grid search
best_params = grid_search.best_params_

print(f"Best parameters from grid search: {best_params}")

# Retrain the random forest model using the optimized parameters
optimized_random_forest_model = RandomForestRegressor(**best_params)
optimized_random_forest_model.fit(X_train, y_train)