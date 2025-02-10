import pandas as pd
from datetime import datetime
import os

def clean_county_name(df, col_name):
    """Standardize county names across datasets"""
    df[col_name] = df[col_name].str.replace(' County', '', regex=False).str.upper().str.strip()
    return df

def merge_datasets():
    # Paths
    brfss_path = '../data/brfss.csv'
    county_path = '../data/county_places.csv'
    food_path = '../data/food_desert.csv'
    output_path = '../data/merged/diabetes_merged.csv'
    
    # Load BRFSS (temporal data)
    brfss = pd.read_csv(brfss_path, parse_dates=['Year'])
    brfss = brfss[brfss['Year'].dt.year >= 2022]
    brfss = clean_county_name(brfss, 'Locationdesc')
    
    # Load County Data (static features)
    county = pd.read_csv(county_path)
    county = clean_county_name(county, 'LocationName')
    county_features = county[['StateDesc', 'LocationName', 'PovertyRate', 'MedianFamilyIncome']]
    
    # Load Food Data (semi-static)
    food = pd.read_csv(food_path)
    food = clean_county_name(food, 'County')
    food_features = food[['State', 'County', 'LILATracts_1And10', 'LowIncomeTracts']]
    
    # Merge BRFSS with County Data
    merged = brfss.merge(
        county_features,
        left_on=['State', 'Locationdesc'],
        right_on=['StateDesc', 'LocationName'],
        how='left'
    )
    
    # Merge with Food Data
    merged = merged.merge(
        food_features,
        left_on=['State', 'Locationdesc'],
        right_on=['State', 'County'],
        how='left'
    )
    
    # Temporal feature
    merged['Years_Since_Survey'] = datetime.now().year - merged['Year'].dt.year
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"Merged data saved to {output_path} | Shape: {merged.shape}")

if __name__ == "__main__":
    merge_datasets()
