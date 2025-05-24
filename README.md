# Stock Price Movement Prediction

A machine learning pipeline to predict next-day stock price direction (Up/Down) using historical market data and technical indicators.

## Features Used

* Scaled Close Price
* Moving Averages (5-day, 10-day, 20-day)
* Exponential Moving Averages (5-day, 10-day)
* RSI (Relative Strength Index)
* Bollinger Bands
* OBV (On-Balance Volume)
* 10-day Price Volatility

## Tools and Libraries

* Python
* yfinance
* pandas, numpy
* scikit-learn
* imbalanced-learn (SMOTE)
* matplotlib, seaborn

## Workflow

1. **Data Fetching**: Download historical data using `yfinance`.
2. **Feature Engineering**: Create indicators from price and volume.
3. **Data Preprocessing**:

   * Fill missing values
   * Scale numeric features
   * Create classification target
4. **Train-Test Split**: 80-20 ratio.
5. **Balancing**: SMOTE used to oversample minority class.
6. **Modeling**: RandomForestClassifier trained on engineered features.
7. **Evaluation**: Confusion matrix and classification report.

## Accuracy Results

* **Accuracy**: 71%
* **Precision**:

  * Class 0 (Down): 72%
  * Class 1 (Up): 69%
* **Recall**:

  * Class 0 (Down): 76%
  * Class 1 (Up): 64%
* **F1-Score**:

  * Class 0 (Down): 74%
  * Class 1 (Up): 67%

## How to Run

1. Install dependencies:

```bash
pip install yfinance pandas scikit-learn imbalanced-learn matplotlib seaborn
```

2. Run the notebook or script in Jupyter or Colab.
