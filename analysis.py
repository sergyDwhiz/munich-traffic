import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_data():
    """Load and explore the Munich traffic accident data."""
    
    df = pd.read_csv('data/raw_traffic_data.csv')
    
    print("Dataset Overview:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    print("\nUnique values in key columns:")
    print(f"MONATSZAHL (Categories): {df['MONATSZAHL'].unique()}")
    print(f"AUSPRAEGUNG (Types): {df['AUSPRAEGUNG'].unique()}")
    print(f"Years: {sorted(df['JAHR'].unique())}")
    
    print("\nData types:")
    print(df.dtypes)
    
    print("\nMissing values:")
    print(df.isnull().sum())
    
    return df

def filter_data_for_prediction(df):
    """Filter data for alcohol accidents prediction task."""
    
    # Filter for Alkoholunfälle and insgesamt type
    filtered_df = df[
        (df['MONATSZAHL'] == 'Alkoholunfälle') & 
        (df['AUSPRAEGUNG'] == 'insgesamt')
    ].copy()
    
    # Extract year and month from MONAT column (format: YYYYMM)
    filtered_df['MONAT_str'] = filtered_df['MONAT'].astype(str)
    
    # Filter out non-numeric entries like 'Summe'
    filtered_df = filtered_df[filtered_df['MONAT_str'].str.len() == 6]
    filtered_df = filtered_df[filtered_df['MONAT_str'].str.isdigit()]
    
    filtered_df['JAHR_from_monat'] = filtered_df['MONAT_str'].str[:4].astype(int)
    filtered_df['MONAT_num'] = filtered_df['MONAT_str'].str[4:].astype(int)
    
    # Use the extracted year (more reliable than JAHR column)
    filtered_df['JAHR'] = filtered_df['JAHR_from_monat']
    filtered_df['MONAT'] = filtered_df['MONAT_num']
    
    # Drop records after 2020 as instructed
    filtered_df = filtered_df[filtered_df['JAHR'] <= 2020]
    
    # Remove invalid entries
    filtered_df = filtered_df.dropna(subset=['MONAT', 'WERT'])
    filtered_df = filtered_df[(filtered_df['MONAT'] >= 1) & (filtered_df['MONAT'] <= 12)]
    
    print(f"\nFiltered data shape: {filtered_df.shape}")
    print(f"Year range: {filtered_df['JAHR'].min()} - {filtered_df['JAHR'].max()}")
    
    # Sort by year and month
    filtered_df = filtered_df.sort_values(['JAHR', 'MONAT'])
    
    return filtered_df

def create_visualizations(df, filtered_df):
    """Create visualizations of the traffic accident data."""
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. All categories over time
    categories = df['MONATSZAHL'].unique()
    for category in categories:
        cat_data = df[(df['MONATSZAHL'] == category) & (df['AUSPRAEGUNG'] == 'insgesamt')].copy()
        if not cat_data.empty:
            # Extract year and month from MONAT column
            cat_data['MONAT_str'] = cat_data['MONAT'].astype(str)
            
            # Filter out non-numeric entries like 'Summe'
            cat_data = cat_data[cat_data['MONAT_str'].str.len() == 6]
            cat_data = cat_data[cat_data['MONAT_str'].str.isdigit()]
            
            if not cat_data.empty:
                cat_data['JAHR_extract'] = cat_data['MONAT_str'].str[:4].astype(int)
                cat_data['MONAT_extract'] = cat_data['MONAT_str'].str[4:].astype(int)
                
                # Filter valid months and years up to 2020
                cat_data = cat_data[
                    (cat_data['MONAT_extract'] >= 1) & 
                    (cat_data['MONAT_extract'] <= 12) &
                    (cat_data['JAHR_extract'] <= 2020)
                ]
                
                if not cat_data.empty:
                    cat_data['date'] = pd.to_datetime(cat_data[['JAHR_extract', 'MONAT_extract']].assign(day=1).rename(columns={'JAHR_extract': 'year', 'MONAT_extract': 'month'}))
                    axes[0,0].plot(cat_data['date'], cat_data['WERT'], label=category, marker='o', markersize=3)
    
    axes[0,0].set_title('Traffic Accidents by Category Over Time')
    axes[0,0].set_xlabel('Date')
    axes[0,0].set_ylabel('Number of Accidents')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Alcohol accidents trend
    filtered_df_copy = filtered_df.copy()
    filtered_df_copy['date'] = pd.to_datetime(filtered_df_copy[['JAHR', 'MONAT']].assign(day=1).rename(columns={'JAHR': 'year', 'MONAT': 'month'}))
    axes[0,1].plot(filtered_df_copy['date'], filtered_df_copy['WERT'], 'r-o', markersize=4)
    axes[0,1].set_title('Alcohol-Related Accidents Over Time')
    axes[0,1].set_xlabel('Date')
    axes[0,1].set_ylabel('Number of Accidents')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Seasonal patterns
    monthly_avg = filtered_df_copy.groupby('MONAT')['WERT'].mean()
    axes[1,0].bar(monthly_avg.index, monthly_avg.values)
    axes[1,0].set_title('Average Alcohol Accidents by Month')
    axes[1,0].set_xlabel('Month')
    axes[1,0].set_ylabel('Average Number of Accidents')
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Yearly trend
    yearly_avg = filtered_df_copy.groupby('JAHR')['WERT'].mean()
    axes[1,1].plot(yearly_avg.index, yearly_avg.values, 'g-o', markersize=6)
    axes[1,1].set_title('Average Alcohol Accidents by Year')
    axes[1,1].set_xlabel('Year')
    axes[1,1].set_ylabel('Average Number of Accidents')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/traffic_accidents_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualization saved as 'data/traffic_accidents_analysis.png'")

def prepare_features(df):
    """Prepare features for machine learning."""
    
    df = df.copy()
    
    # MONAT should already be numeric from filter_data_for_prediction
    # Create time-based features
    df['month_sin'] = np.sin(2 * np.pi * df['MONAT'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['MONAT'] / 12)
    
    # Create trend feature (months since start)
    df['time_trend'] = (df['JAHR'] - df['JAHR'].min()) * 12 + df['MONAT']
    
    # Lag features
    df = df.sort_values(['JAHR', 'MONAT'])
    df['lag_1'] = df['WERT'].shift(1)
    df['lag_12'] = df['WERT'].shift(12)  # Same month previous year
    
    # Rolling averages
    df['rolling_3'] = df['WERT'].rolling(window=3, min_periods=1).mean()
    df['rolling_12'] = df['WERT'].rolling(window=12, min_periods=1).mean()
    
    return df

def train_prediction_models(df):
    """Train models to predict alcohol accidents."""
    
    # Prepare features
    df = prepare_features(df)
    
    # Remove rows with NaN values after feature engineering
    df_clean = df.dropna()
    
    if len(df_clean) < 24:  # Need enough data for training
        print("Not enough data for training. Using simple trend model.")
        return train_simple_model(df)
    
    # Features for prediction
    feature_cols = ['time_trend', 'month_sin', 'month_cos', 'lag_1', 'lag_12', 'rolling_3']
    X = df_clean[feature_cols]
    y = df_clean['WERT']
    
    # Split data (use last 12 months for testing)
    split_idx = len(X) - 12
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        results[name] = {
            'model': model,
            'mae': mae,
            'rmse': rmse,
            'predictions': y_pred
        }
        
        print(f"{name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    
    # Select best model (lowest MAE)
    best_model_name = min(results.keys(), key=lambda k: results[k]['mae'])
    best_model = results[best_model_name]['model']
    
    print(f"\nBest model: {best_model_name}")
    
    return best_model, df_clean, feature_cols

def train_simple_model(df):
    """Train a simple trend model when there's insufficient data."""
    
    df = df.sort_values(['JAHR', 'MONAT'])
    df['time_trend'] = (df['JAHR'] - df['JAHR'].min()) * 12 + df['MONAT']
    
    # Simple linear regression on time trend and month
    X = df[['time_trend', 'MONAT']].values
    y = df['WERT'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    return model, df, ['time_trend', 'MONAT']

def predict_future(model, df, feature_cols, target_year=2021, target_month=1):
    """Predict alcohol accidents for a specific month."""
    
    # Calculate time trend for target date
    time_trend = (target_year - df['JAHR'].min()) * 12 + target_month
    
    if 'month_sin' in feature_cols:
        # Full feature set
        month_sin = np.sin(2 * np.pi * target_month / 12)
        month_cos = np.cos(2 * np.pi * target_month / 12)
        
        # Use last known values for lag features
        last_value = df['WERT'].iloc[-1]
        last_year_value = df[df['MONAT'] == target_month]['WERT'].iloc[-1] if len(df[df['MONAT'] == target_month]) > 0 else last_value
        rolling_3 = df['WERT'].tail(3).mean()
        
        features = np.array([[time_trend, month_sin, month_cos, last_value, last_year_value, rolling_3]])
    else:
        # Simple feature set
        features = np.array([[time_trend, target_month]])
    
    prediction = model.predict(features)[0]
    prediction = max(0, round(prediction))  # Ensure non-negative integer
    
    return prediction

if __name__ == "__main__":
    print("Munich Traffic Accident Analysis and Prediction")
    print("=" * 50)
    
    # Load and explore data
    df = load_and_explore_data()
    
    # Filter for alcohol accidents
    filtered_df = filter_data_for_prediction(df)
    
    # Create visualizations
    create_visualizations(df, filtered_df)
    
    # Train prediction model
    model, model_df, feature_cols = train_prediction_models(filtered_df)
    
    # Make prediction for January 2021
    prediction = predict_future(model, model_df, feature_cols, 2021, 1)
    
    print(f"\nPrediction for Alkoholunfälle, insgesamt, January 2021: {prediction}")
    
    # Save model results
    results = {
        'prediction_2021_01': prediction,
        'model_type': type(model).__name__,
        'features_used': feature_cols
    }
    
    print(f"\nModel results: {results}")
