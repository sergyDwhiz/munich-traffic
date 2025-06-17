# Munich Traffic Accident Prediction - AI Engineering Challenge

This project implements an AI model to predict alcohol-related traffic accidents in Munich using the "Monatszahlen Verkehrsunfälle" dataset from München Open Data Portal.

## Project Overview

The challenge involves building a machine learning model to predict alcohol-related traffic accidents in Munich, specifically predicting the number of accidents for:
- **Category**: Alkoholunfälle (Alcohol accidents)
- **Type**: insgesamt (total)
- **Target**: January 2021

## Dataset

The project uses the Munich traffic accident dataset available at:
- [Monatszahlen Verkehrsunfälle Dataset - München Open Data Portal](https://opendata.muenchen.de/dataset/monatszahlen-verkehrsunfaelle)

## Project Structure

```
munich-traffic/
├── analysis.py              # Main analysis and prediction script
├── data_download.py         # Data download and preprocessing
├── requirements.txt         # Python dependencies
├── data/
│   ├── raw_traffic_data.csv            # Raw dataset
│   └── traffic_accidents_analysis.png  # Generated visualization
└── README.md               # This file
```

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/sergyDwhiz/munich-traffic.git
cd munich-traffic
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. **Download the data** (if not already present):
```bash
python data_download.py
```

2. **Run the analysis and prediction**:
```bash
python analysis.py
```

This will:
- Load and explore the Munich traffic accident data
- Filter for alcohol-related accidents
- Create visualizations showing trends and patterns
- Train machine learning models for prediction
- Generate prediction for January 2021

## Model Results

- **Model Type**: Linear Regression
- **Performance**: MAE (Mean Absolute Error): 4.92
- **Prediction for January 2021**: 25 alcohol-related accidents

## Key Features

- **Data Preprocessing**: Filters data for alcohol accidents up to 2020
- **Feature Engineering**: Creates time-based and lag features
- **Visualization**: Generates comprehensive analysis plots
- **Model Training**: Compares Linear Regression and Random Forest models
- **Prediction**: Forecasts accidents for specified month/year

## Mission Screenshots

### Mission 1 Completion
![Mission 1 Screenshot](screenshots/mission1_completion.png)
*Screenshot showing successful completion of Mission 1*

### Mission 2 Completion
![Mission 2 Screenshot](screenshots/mission2_completion.png)
*Screenshot showing successful completion of Mission 2*

### Mission 3 Completion
![Mission 3 Screenshot](screenshots/mission3_completion.png)
*Screenshot showing "Congratulations! Achieved Mission 3!" confirmation message*

## Data Analysis Visualization

The project generates comprehensive visualizations including:
- Traffic accidents by category over time
- Alcohol accidents trend analysis
- Seasonal patterns (monthly averages)
- Yearly trend analysis

![Traffic Analysis](data/traffic_accidents_analysis.png)

## Technical Implementation

### Data Processing
- Filters dataset for alcohol accidents (Alkoholunfälle) with type 'insgesamt'
- Removes data after 2020 as specified in challenge requirements
- Handles data cleaning and validation

### Feature Engineering
- Time-based features (trend, seasonal components)
- Lag features (previous month, same month previous year)
- Rolling averages for smoothing

### Model Training
- Compares multiple algorithms (Linear Regression, Random Forest)
- Uses time series cross-validation approach
- Selects best model based on Mean Absolute Error

## Results Summary

| Metric | Value |
|--------|-------|
| Model Type | Linear Regression |
| Training Period | Up to 2020 |
| Test MAE | 4.92 |
| Prediction (Jan 2021) | 25 accidents |

## Repository Information

- **GitHub Repository**: [https://github.com/sergyDwhiz/munich-traffic](https://github.com/sergyDwhiz/munich-traffic)
- **Author**: Sergius Nyah
- **Email**: sergiusnyah@gmail.com
- **Challenge**: Digital Product School AI Engineering Challenge

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- München Open Data Portal for providing the traffic accident dataset
- Digital Product School for the AI Engineering Challenge
