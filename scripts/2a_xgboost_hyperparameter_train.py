# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 10:23:04 2025

@author: AD14407
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    # Load merged data
    data_path = '../data/diabetes_merged.csv'
    df = pd.read_csv(data_path)
    
    # Features & Target
    features = ['PovertyRate', 'MedianFamilyIncome', 'LILATracts_1And10', 
               'LowIncomeTracts', 'Years_Since_Survey']
    target = 'Data_value' # BRFSS diabetes prevalence

    # Preprocessing
    X = df[features].fillna(df[features].median())
    y = df[target]
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define the parameter grid
    param_grid = {
        'n_estimators': [100, 400, 600, 1000],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.05, 0.1],
        'colsample_bytree': [0.3, 0.7, 1],
        'subsample': [0.5, 0.7, 1],
        'gamma': [0, 0.1, 0.25],
        'reg_alpha': [0, 0.01, 0.1],
        'reg_lambda': [1, 1.5, 2],
    }
    
    # XGBoost regressor
    model = XGBRegressor(random_state=42, objective='reg:squarederror')

    # Random search
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=50,  # Number of random configurations tested
        scoring='r2',
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1  # Use all cores for parallel processing
    )

    # Fit the random search model
    random_search.fit(X_train, y_train)

    # Best parameters and score
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best R² score: {random_search.best_score_}")

    # Evaluate the best estimator on the test set
    best_model = random_search.best_estimator_
    preds = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r_squared = r2_score(y_test, preds)

    print(f"Optimized XGBoost Performance:")
    print(f"- MAE: {mae:.2f}")
    print(f"- MSE: {mse:.2f}")
    print(f"- RMSE: {rmse:.2f}")
    print(f"- R²: {r_squared:.2f}")

    # Save the best model
    os.makedirs('../models', exist_ok=True)
    joblib.dump(best_model, '../models/optimized_xgboost_model.pkl')

    # Plot predicted vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, preds, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Predicted vs Actual (Optimized XGBoost)')
    plt.tight_layout()
    plt.savefig('../plots/optimized_xgboost_predictions.png')

if __name__ == "__main__":
    main()
