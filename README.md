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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- München Open Data Portal for providing the traffic accident dataset
- Digital Product School for the AI Engineering Challenge
