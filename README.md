# 📈 Market Predictor

A Python program that predicts stock and cryptocurrency market prices using machine learning and **only free services**.

## 🌟 Features

- **Free Data Source**: Uses `yfinance` to fetch historical market data at no cost
- **Multiple Asset Classes**: Supports stocks, cryptocurrencies, ETFs, and market indices
- **Machine Learning Models**: 
  - Random Forest Regressor
  - Gradient Boosting Regressor
- **Technical Indicators**: 
  - Moving Averages (SMA, EMA)
  - MACD (Moving Average Convergence Divergence)
  - RSI (Relative Strength Index)
  - Bollinger Bands
  - Volume indicators
  - Momentum indicators
- **Future Predictions**: Forecast prices for the next 7 days (customizable)
- **Performance Metrics**: RMSE, MAE, R², MAPE
- **Visualization**: Beautiful charts comparing predictions vs actual prices

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone or download this repository**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install yfinance pandas numpy scikit-learn matplotlib seaborn
```

### Usage

#### Basic Usage

Run the program with default examples:
```bash
python market_predictor.py
```

This will:
1. Predict Apple (AAPL) stock prices
2. Predict Bitcoin (BTC-USD) prices
3. Prompt you to enter your own ticker symbol

#### Custom Predictions

```python
from market_predictor import MarketPredictor

# Predict Apple stock
predictor = MarketPredictor(
    ticker="AAPL",           # Stock ticker
    period="2y",             # Historical data period
    model_type="random_forest"  # ML model type
)
predictor.run(predict_days=7, show_plots=True)
```

#### Supported Tickers

**Stocks:**
- `AAPL` - Apple Inc.
- `GOOGL` - Alphabet Inc. (Google)
- `MSFT` - Microsoft Corporation
- `TSLA` - Tesla Inc.
- `AMZN` - Amazon.com Inc.
- `META` - Meta Platforms Inc.
- `NVDA` - NVIDIA Corporation
- Any other stock symbol on Yahoo Finance

**Cryptocurrencies:**
- `BTC-USD` - Bitcoin
- `ETH-USD` - Ethereum
- `BNB-USD` - Binance Coin
- `SOL-USD` - Solana
- `ADA-USD` - Cardano

**Indices:**
- `^GSPC` - S&P 500
- `^DJI` - Dow Jones Industrial Average
- `^IXIC` - NASDAQ Composite

## 📊 How It Works

### 1. Data Collection
- Fetches historical price data from Yahoo Finance (free API)
- Downloads OHLCV data (Open, High, Low, Close, Volume)

### 2. Feature Engineering
Creates 26+ technical indicators:
- **Price Features**: Returns, Log Returns
- **Trend Indicators**: SMA (5, 10, 20, 50 days), EMA (12, 26 days)
- **Momentum Indicators**: MACD, Signal Line, RSI, Momentum
- **Volatility Indicators**: Bollinger Bands, Historical Volatility
- **Volume Indicators**: Volume Moving Average, Volume Ratio

### 3. Model Training
- Splits data into training (80%) and testing (20%) sets
- Trains ensemble machine learning models:
  - **Random Forest**: 100 decision trees with optimized parameters
  - **Gradient Boosting**: Sequential ensemble learning
- Uses MinMax scaling for better convergence

### 4. Evaluation
Calculates multiple performance metrics:
- **RMSE** (Root Mean Squared Error): Average prediction error
- **MAE** (Mean Absolute Error): Average absolute error
- **R²** (R-squared): Proportion of variance explained (0-1, higher is better)
- **MAPE** (Mean Absolute Percentage Error): Percentage error

### 5. Prediction
- Predicts future prices for specified number of days
- Shows daily predictions with change direction and percentage
- Generates visual charts saved as PNG files

## 📈 Example Output

```
============================================================
MARKET PREDICTOR - AAPL
============================================================

Fetching data for AAPL...
✓ Successfully fetched 504 days of data
  Date range: 2022-01-03 to 2024-01-03

Creating technical indicators...
✓ Created 31 features

Preparing training data...
✓ Training set: 403 samples
✓ Test set: 101 samples

Training random_forest model...
✓ Model training complete

============================================================
MODEL EVALUATION
============================================================
Root Mean Squared Error (RMSE): $2.45
Mean Absolute Error (MAE): $1.82
R² Score: 0.9234
Mean Absolute Percentage Error (MAPE): 1.23%
============================================================

============================================================
FUTURE PREDICTIONS (Next 7 days)
============================================================
Current Price (2024-01-03): $184.25

Day 1 (2024-01-04): $185.12 ↑ (+0.87, +0.47%)
Day 2 (2024-01-05): $186.03 ↑ (+0.91, +0.49%)
Day 3 (2024-01-06): $185.67 ↓ (-0.36, -0.19%)
Day 4 (2024-01-07): $186.45 ↑ (+0.78, +0.42%)
Day 5 (2024-01-08): $187.23 ↑ (+0.78, +0.42%)
Day 6 (2024-01-09): $187.89 ↑ (+0.66, +0.35%)
Day 7 (2024-01-10): $188.34 ↑ (+0.45, +0.24%)
============================================================

✓ Plot saved as 'AAPL_predictions.png'
✓ Analysis complete!
```

## 🎯 Model Parameters

### Random Forest (Default)
```python
n_estimators=100      # Number of trees
max_depth=10          # Maximum tree depth
min_samples_split=5   # Minimum samples to split
min_samples_leaf=2    # Minimum samples per leaf
```

### Gradient Boosting
```python
n_estimators=100      # Number of boosting stages
max_depth=5           # Maximum tree depth
learning_rate=0.1     # Shrinks contribution of each tree
```

## ⚠️ Important Disclaimers

1. **Not Financial Advice**: This program is for educational and research purposes only. Do not use it as the sole basis for investment decisions.

2. **Past Performance**: Historical data does not guarantee future results. Markets are influenced by many unpredictable factors.

3. **Model Limitations**: 
   - Cannot predict black swan events
   - Doesn't account for news, earnings, or macro events
   - Works best for short-term predictions (1-7 days)
   - Accuracy varies by market conditions

4. **Risk Warning**: Trading stocks and cryptocurrencies involves substantial risk of loss. Only invest what you can afford to lose.

## 🔧 Customization

### Change Prediction Period
```python
predictor = MarketPredictor(ticker="AAPL", period="5y")  # 5 years of data
predictor.run(predict_days=14)  # Predict 14 days ahead
```

### Switch ML Model
```python
predictor = MarketPredictor(ticker="TSLA", model_type="gradient_boosting")
```

### Disable Plots
```python
predictor.run(predict_days=7, show_plots=False)
```

## 📚 Dependencies

All dependencies are free and open-source:

- **yfinance**: Free Yahoo Finance API wrapper
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **matplotlib**: Plotting and visualization
- **seaborn**: Statistical data visualization (optional)

## 🤝 Contributing

Feel free to:
- Add new technical indicators
- Implement additional ML models (LSTM, Prophet, etc.)
- Improve feature engineering
- Enhance visualization
- Add more performance metrics

## 📝 License

This project is provided as-is for educational purposes. Use at your own risk.

## 🐛 Troubleshooting

### "No data found for ticker"
- Check if the ticker symbol is correct
- Verify the asset is available on Yahoo Finance
- Try a different time period

### "Module not found"
- Run: `pip install -r requirements.txt`
- Ensure you're using Python 3.8+

### Plot doesn't display
- The plot is automatically saved as a PNG file
- Check for GUI/display limitations on your system
- Set `show_plots=False` to skip display

## 💡 Tips for Better Predictions

1. **Use more data**: Longer historical periods (2-5 years) generally improve accuracy
2. **Market conditions matter**: Models perform better in stable markets
3. **Combine with other analysis**: Use fundamental analysis and news alongside predictions
4. **Regular retraining**: Retrain models periodically with updated data
5. **Diversification**: Never rely on a single prediction or model

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the example code in `market_predictor.py`
3. Ensure all dependencies are properly installed

---

**Remember**: This is a learning tool. Always do your own research and consult with financial advisors before making investment decisions.

Happy predicting! 📊🚀