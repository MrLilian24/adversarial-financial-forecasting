# Adversarial Attacks on Financial Time Series Forecasting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-in%20development-orange.svg)]()

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [References](#references)
- [License](#license)
- [Author](#author)

## Overview

This project investigates the vulnerability of LSTM-based financial forecasting models to adversarial attacks. Using stock price data from Yahoo Finance, a univariate LSTM model is trained to predict the next-day closing price of AAPL stock. The Fast Gradient Sign Method (FGSM) is then applied to generate adversarial inputs, and the impact on model predictions is measured using MAE and RMSE metrics.

This work is inspired by recent research on slope-based adversarial attacks on financial time-series data, scaled down to an empirical study focused on reproducibility and clarity.

## Project Structure

```
adversarial-financial-forecasting/
├── notebooks/
│   ├── 01_data_exploration.ipynb     # EDA* on AAPL stock data
│   └── 02_data_preparation.ipynb     # Splitting, scaling, sequence creation
├── src/                              # Reusable source code (DRY principles)
├── requirements.txt
├── LICENSE
└── README.md
```

_\*EDA = Exploratory Data Analysis_

## Installation

**Requirements:** Python 3.12+

```bash
# Clone the repository
git clone https://github.com/MrLilian24/adversarial-financial-forecasting.git
cd adversarial-financial-forecasting

# Create and activate a virtual environment (venv or conda)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run the notebooks in order:

1. `notebooks/01_data_exploration.ipynb` - Download and explore AAPL stock data (5 years, daily)
2. `notebooks/02_data_preparation.ipynb` - Temporal split (70/20/10), StandardScaler normalization, sliding window sequence creation (lookback = 60 days)

## References

### Academic Papers

- Luszczynski, D. (2025). [Targeted Manipulation: Slope-Based Attacks on Financial Time-Series Data](https://arxiv.org/abs/2511.19330)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Rémi LAVERGNE**

---

> [!NOTE]
> This project is currently under active development.
