import requests
import pandas as pd
import os
import numpy as np

def create_sample_data():
    """Create sample traffic accident data for development when download fails."""
    print("Creating sample data for development...")

    categories = ['Alkoholunfälle', 'Fluchtunfälle', 'Verkehrsunfälle']
    accident_types = ['insgesamt', 'innerorts', 'außerorts']
    years = range(2010, 2021)
    months = range(1, 13)

    data = []

    for category in categories:
        for acc_type in accident_types:
            for year in years:
                for month in months:
                    # Create realistic sample data with some trend and seasonality
                    base_value = {
                        'Alkoholunfälle': 15,
                        'Fluchtunfälle': 25,
                        'Verkehrsunfälle': 150
                    }[category]

                    # Add trend (slight decrease over years for alcohol accidents)
                    trend = (2020 - year) * 0.1 if category == 'Alkoholunfälle' else 0

                    # Add seasonality (winter months have fewer accidents)
                    seasonality = 1.0 if month in [6, 7, 8] else 0.8 if month in [12, 1, 2] else 0.9

                    # Add random variation
                    noise = np.random.normal(0, 0.1)

                    value = int(base_value * (1 + trend) * seasonality * (1 + noise))
                    value = max(1, value)  # Ensure positive values

                    data.append({
                        'MONATSZAHL': f"{category}",
                        'AUSPRAEGUNG': acc_type,
                        'JAHR': year,
                        'MONAT': month,
                        'WERT': value
                    })

    df = pd.DataFrame(data)

    # Save sample data
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/raw_traffic_data.csv', index=False)

    print(f"Sample data created with shape: {df.shape}")
    return df

def download_munich_traffic_data():
    """Download Munich traffic accident data from the open data portal."""

    # Try multiple potential URLs
    urls = [
        "https://opendata.muenchen.de/dataset/monatszahlen-verkehrsunfaelle/resource/40094bd6-f82d-4979-949b-26c8dc00b9a7/download/monatszahlenverkehrsunfaelle.csv",
        "https://opendata.muenchen.de/dataset/246175dd-b515-4cad-ba53-9a536c37d809/resource/40094bd6-f82d-4979-949b-26c8dc00b9a7/download/monatszahlenverkehrsunfaelle.csv"
    ]

    print("Downloading Munich traffic accident data...")

    for url in urls:
        try:
            response = requests.get(url)
            response.raise_for_status()

            # Create data directory if it doesn't exist
            os.makedirs('data', exist_ok=True)

            # Save the raw data
            with open('data/raw_traffic_data.csv', 'wb') as f:
                f.write(response.content)

            print("Data downloaded successfully!")

            # Load and inspect the data
            df = pd.read_csv('data/raw_traffic_data.csv')
            print(f"\nDataset shape: {df.shape}")
            print(f"\nColumns: {list(df.columns)}")
            print(f"\nFirst few rows:")
            print(df.head())

            return df

        except requests.exceptions.RequestException as e:
            print(f"Failed to download from {url}: {e}")
            continue

    # If all downloads fail, create sample data
    print("All download attempts failed. Creating sample data for development...")
    return create_sample_data()

if __name__ == "__main__":
    df = download_munich_traffic_data()
