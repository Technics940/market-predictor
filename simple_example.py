#!/usr/bin/env python3
"""
Simple Market Predictor Example
Quick and easy script to predict stock/crypto prices
"""

from market_predictor import MarketPredictor


def predict_stock(ticker, days=7):
    """
    Simple function to predict a stock or crypto price

    Args:
        ticker (str): Stock/Crypto symbol (e.g., 'AAPL', 'BTC-USD')
        days (int): Number of days to predict ahead
    """
    print(f"\n🔮 Predicting {ticker} for the next {days} days...")
    print("-" * 60)

    # Create predictor
    predictor = MarketPredictor(
        ticker=ticker,
        period="2y",  # Use 2 years of historical data
        model_type="random_forest",
    )

    # Run prediction
    predictor.run(predict_days=days, show_plots=True)

    print(f"\n✅ Prediction complete! Check {ticker}_predictions.png for chart.")


if __name__ == "__main__":
    # Example 1: Predict Apple stock
    predict_stock("AAPL", days=7)

    # Example 2: Predict Bitcoin
    # predict_stock("BTC-USD", days=7)

    # Example 3: Predict Tesla
    # predict_stock("TSLA", days=7)

    # Example 4: Predict Ethereum
    # predict_stock("ETH-USD", days=7)

    # Example 5: Predict S&P 500 Index
    # predict_stock("^GSPC", days=7)

    print("\n" + "=" * 60)
    print("💡 TIP: Edit this file to predict different stocks!")
    print("   Just uncomment the examples above or add your own.")
    print("=" * 60)
