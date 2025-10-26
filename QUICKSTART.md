# 🚀 Quick Start Guide - Market Predictor

Get up and running with market predictions in under 5 minutes!

## ⚡ Installation (3 steps)

### Step 1: Install Python
Make sure you have Python 3.8 or higher installed:
```bash
python --version
```

### Step 2: Install Dependencies
```bash
pip install yfinance pandas numpy scikit-learn matplotlib
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

### Step 3: Run Your First Prediction
```bash
python simple_example.py
```

That's it! 🎉

---

## 🎯 Quick Examples

### Example 1: Predict Apple Stock
```python
from market_predictor import MarketPredictor

predictor = MarketPredictor(ticker="AAPL")
predictor.run(predict_days=7)
```

### Example 2: Predict Bitcoin
```python
predictor = MarketPredictor(ticker="BTC-USD", period="1y")
predictor.run(predict_days=7)
```

### Example 3: Predict with Different Model
```python
predictor = MarketPredictor(
    ticker="TSLA",
    model_type="gradient_boosting"
)
predictor.run(predict_days=14)
```

---

## 📊 Popular Tickers to Try

### 🏢 Top Stocks
- **AAPL** - Apple
- **GOOGL** - Google
- **MSFT** - Microsoft
- **TSLA** - Tesla
- **NVDA** - NVIDIA
- **AMZN** - Amazon
- **META** - Meta/Facebook

### 🪙 Cryptocurrencies
- **BTC-USD** - Bitcoin
- **ETH-USD** - Ethereum
- **BNB-USD** - Binance Coin
- **SOL-USD** - Solana
- **ADA-USD** - Cardano
- **DOGE-USD** - Dogecoin

### 📈 Market Indices
- **^GSPC** - S&P 500
- **^DJI** - Dow Jones
- **^IXIC** - NASDAQ

---

## 🎨 Understanding the Output

When you run a prediction, you'll see:

### 1. Data Fetching
```
Fetching data for AAPL...
✓ Successfully fetched 504 days of data
```

### 2. Model Training
```
Training random_forest model...
✓ Model training complete
```

### 3. Performance Metrics
```
Root Mean Squared Error (RMSE): $2.45
Mean Absolute Error (MAE): $1.82
R² Score: 0.9234  ← Higher is better (max 1.0)
MAPE: 1.23%        ← Lower is better
```

**What these mean:**
- **RMSE/MAE**: Average prediction error in dollars
- **R² Score**: How well the model fits (0.9+ is excellent)
- **MAPE**: Percentage error (under 5% is good)

### 4. Future Predictions
```
Day 1 (2024-01-04): $185.12 ↑ (+0.87, +0.47%)
Day 2 (2024-01-05): $186.03 ↑ (+0.91, +0.49%)
```

- **↑** = Price going up
- **↓** = Price going down
- First number = change in dollars
- Second number = percentage change

### 5. Chart Output
A PNG file is saved showing:
- **Top chart**: Model accuracy on test data
- **Bottom chart**: Historical prices + future predictions

---

## 🔧 Common Customizations

### Predict More Days
```python
predictor.run(predict_days=30)  # Predict a month ahead
```

### Use More Historical Data
```python
predictor = MarketPredictor(ticker="AAPL", period="5y")  # 5 years
```

Available periods: `1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `10y`, `max`

### Disable Charts
```python
predictor.run(predict_days=7, show_plots=False)
```

### Try Different Models
```python
# Random Forest (default) - Good for most cases
predictor = MarketPredictor(ticker="AAPL", model_type="random_forest")

# Gradient Boosting - Often more accurate but slower
predictor = MarketPredictor(ticker="AAPL", model_type="gradient_boosting")
```

---

## 💡 Pro Tips

### ✅ DO:
1. **Use 1-2 years of data** for most accurate predictions
2. **Predict 1-7 days ahead** for best accuracy
3. **Retrain regularly** with new data
4. **Compare multiple models** to see which performs best
5. **Check the R² score** - above 0.85 is good

### ❌ DON'T:
1. **Don't predict too far ahead** (accuracy decreases after 7 days)
2. **Don't use during market crashes** (models trained on normal conditions)
3. **Don't trade solely on predictions** (use as one tool among many)
4. **Don't ignore the metrics** (low R² = unreliable predictions)
5. **Don't forget this is educational** (not financial advice!)

---

## 🐛 Troubleshooting

### Problem: "No module named 'yfinance'"
**Solution:**
```bash
pip install yfinance
```

### Problem: "No data found for ticker"
**Solutions:**
- Check ticker spelling (use Yahoo Finance to verify)
- Try a different period: `period="1y"` instead of `period="2y"`
- Some tickers require suffixes (e.g., `BTC-USD` not `BTC`)

### Problem: "ModuleNotFoundError: No module named 'sklearn'"
**Solution:**
```bash
pip install scikit-learn
```

### Problem: Poor predictions (low R² score)
**Solutions:**
- Use more historical data: `period="5y"`
- Try different model: `model_type="gradient_boosting"`
- Some assets are harder to predict (high volatility)
- Check if there was recent major news affecting the asset

### Problem: Chart doesn't show
**Solution:**
- Chart is still saved as PNG file in your directory
- Look for `TICKER_predictions.png`
- Or disable: `show_plots=False`

---

## 📖 Next Steps

### Want to Learn More?
1. Read the full [README.md](README.md) for detailed documentation
2. Explore the code in `market_predictor.py`
3. Try different tickers and compare results

### Want to Customize?
1. **Add new technical indicators** in `create_features()` method
2. **Try other ML models** (SVM, Neural Networks, LSTM)
3. **Add sentiment analysis** from news/social media
4. **Implement ensemble predictions** combining multiple models

### Want to Improve Accuracy?
1. **Feature engineering**: Add more relevant indicators
2. **Hyperparameter tuning**: Optimize model parameters
3. **Ensemble methods**: Combine predictions from multiple models
4. **External data**: Incorporate economic indicators, news sentiment
5. **Regular retraining**: Update model with latest data

---

## ⚠️ Important Reminder

This tool is for **EDUCATIONAL PURPOSES ONLY**:
- Not financial advice
- Past performance ≠ future results
- Markets are unpredictable
- Always do your own research
- Never invest more than you can afford to lose

---

## 🎓 Example Workflow

Here's a typical workflow for making predictions:

```python
from market_predictor import MarketPredictor

# 1. Create predictor for your asset
predictor = MarketPredictor(
    ticker="AAPL",              # What to predict
    period="2y",                # How much history
    model_type="random_forest"  # Which model
)

# 2. Run the complete analysis
predictor.run(
    predict_days=7,    # How far ahead
    show_plots=True    # Show charts
)

# 3. Check the output:
#    - R² score (should be > 0.85)
#    - MAPE (should be < 5%)
#    - Predictions (Day 1-7)
#    - Chart file (TICKER_predictions.png)

# 4. Make informed decisions based on:
#    - Model accuracy metrics
#    - Prediction trends
#    - Your own research
#    - Risk tolerance
```

---

## 🚀 You're Ready!

Now you know how to:
- ✅ Install and run the predictor
- ✅ Interpret the results
- ✅ Customize predictions
- ✅ Troubleshoot common issues

**Start predicting now:**
```bash
python simple_example.py
```

Happy predicting! 📊📈

---

*For detailed documentation, see [README.md](README.md)*
*For code details, see [market_predictor.py](market_predictor.py)*