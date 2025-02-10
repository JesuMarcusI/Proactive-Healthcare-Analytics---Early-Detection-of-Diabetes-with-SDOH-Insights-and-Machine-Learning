import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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
    target = 'Data_value' #  diabetes prevalence
    
    # Preprocessing
    X = df[features].fillna(df[features].median())
    y = df[target]
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model Training
    model = XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluation
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r_squared = r2_score(y_test, preds)

    # Calculate 'accuracy' as a measure where prediction tolerance is set
    tolerance = 0.05  # e.g., 5% tolerance
    accuracy = np.mean(np.abs((y_test - preds) / y_test) < tolerance) * 100
    
    print(f"XGBoost Performance:")
    print(f"- MAE: {mae:.2f}")
    print(f"- MSE: {mse:.2f}")
    print(f"- RMSE: {rmse:.2f}")
    print(f"- R²: {r_squared:.2f}")
    print(f"- Accuracy (within ±5%): {accuracy:.2f}%")
    
    # Save Model
    os.makedirs('../models', exist_ok=True)
    joblib.dump(model, '../models/xgboost_model.pkl')
    
    # Feature Importance Plot
    plt.figure(figsize=(10,6))
    pd.Series(model.feature_importances_, index=features).sort_values().plot(kind='barh')
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()
    os.makedirs('../plots', exist_ok=True)
    plt.savefig('../plots/xgboost_importance.png')
    
    # Residual Plot
    plt.figure(figsize=(10,6))
    residuals = y_test - preds
    plt.scatter(preds, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Values')
    plt.tight_layout()
    plt.savefig('../plots/xgboost_residuals.png')
    
    # Prediction vs Actual
    plt.figure(figsize=(10,6))
    plt.scatter(y_test, preds, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Predicted vs Actual')
    plt.tight_layout()
    plt.savefig('../plots/predicted_vs_actual.png')
    
    # Histogram with Bell Curve
    plt.figure(figsize=(10,6))
    sns.histplot(y_test, kde=True, color='blue', label='Actual', stat='density', linewidth=0)
    sns.histplot(preds, kde=True, color='orange', label='Predicted', stat='density', linewidth=0)
    plt.xlabel('Data Value')
    plt.ylabel('Density')
    plt.title('Distribution of Actual vs Predicted Values')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../plots/predictions_distribution.png')

if __name__ == "__main__":
    main()
