#!/usr/bin/env python3
"""
Batch Market Predictor
Compare predictions for multiple stocks/cryptocurrencies at once
"""

import sys
from market_predictor import MarketPredictor
import pandas as pd
from datetime import datetime


class BatchPredictor:
    """
    Predict multiple tickers and compare results
    """

    def __init__(self, tickers, period="2y", model_type="random_forest"):
        """
        Initialize batch predictor

        Args:
            tickers (list): List of ticker symbols
            period (str): Historical data period
            model_type (str): ML model type
        """
        self.tickers = tickers
        self.period = period
        self.model_type = model_type
        self.results = []

    def predict_all(self, predict_days=7):
        """
        Run predictions for all tickers

        Args:
            predict_days (int): Number of days to predict
        """
        print("\n" + "=" * 70)
        print(f"BATCH MARKET PREDICTOR - Analyzing {len(self.tickers)} assets")
        print("=" * 70 + "\n")

        for i, ticker in enumerate(self.tickers, 1):
            print(f"\n[{i}/{len(self.tickers)}] Processing {ticker}...")
            print("-" * 70)

            try:
                # Create predictor
                predictor = MarketPredictor(
                    ticker=ticker, period=self.period, model_type=self.model_type
                )

                # Fetch and prepare data
                if not predictor.fetch_data():
                    self.results.append(
                        {
                            "Ticker": ticker,
                            "Status": "Failed",
                            "Error": "No data available",
                        }
                    )
                    continue

                predictor.create_features()
                X_train, X_test, y_train, y_test = predictor.prepare_training_data()

                # Train and evaluate
                predictor.train_model(X_train, y_train)
                y_test_actual, y_pred_actual = predictor.evaluate_model(X_test, y_test)

                # Calculate metrics
                from sklearn.metrics import mean_squared_error, r2_score
                import numpy as np

                rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
                r2 = r2_score(y_test_actual, y_pred_actual)
                mape = (
                    np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual))
                    * 100
                )

                # Get predictions
                future_dates, future_predictions = predictor.predict_future(
                    days=predict_days
                )

                # Get current price and first prediction
                current_price = predictor.data["Close"].iloc[-1]
                next_day_price = future_predictions[0]
                week_price = future_predictions[-1]

                # Calculate changes
                day_change = ((next_day_price - current_price) / current_price) * 100
                week_change = ((week_price - current_price) / current_price) * 100

                # Store results
                self.results.append(
                    {
                        "Ticker": ticker,
                        "Status": "Success",
                        "Current_Price": f"${current_price:.2f}",
                        "Next_Day": f"${next_day_price:.2f}",
                        "Day_Change_%": f"{day_change:+.2f}%",
                        f"Day_{predict_days}": f"${week_price:.2f}",
                        f"Day_{predict_days}_Change_%": f"{week_change:+.2f}%",
                        "RMSE": f"${rmse:.2f}",
                        "R2_Score": f"{r2:.4f}",
                        "MAPE_%": f"{mape:.2f}%",
                    }
                )

                print(f"✓ {ticker} completed successfully")

            except Exception as e:
                print(f"✗ {ticker} failed: {str(e)}")
                self.results.append(
                    {"Ticker": ticker, "Status": "Failed", "Error": str(e)}
                )

        print("\n" + "=" * 70)
        print("BATCH PROCESSING COMPLETE")
        print("=" * 70 + "\n")

    def display_summary(self):
        """
        Display summary of all predictions
        """
        if not self.results:
            print("No results to display")
            return

        print("\n" + "=" * 70)
        print("PREDICTION SUMMARY")
        print("=" * 70 + "\n")

        # Separate successful and failed predictions
        successful = [r for r in self.results if r.get("Status") == "Success"]
        failed = [r for r in self.results if r.get("Status") == "Failed"]

        if successful:
            df = pd.DataFrame(successful)
            # Reorder columns
            column_order = ["Ticker", "Current_Price", "Next_Day", "Day_Change_%"]
            other_cols = [col for col in df.columns if col not in column_order]
            df = df[column_order + other_cols]

            print("✅ SUCCESSFUL PREDICTIONS:")
            print(df.to_string(index=False))
            print()

            # Show top movers
            if len(successful) > 1:
                # Extract numeric values for sorting
                df_numeric = df.copy()
                df_numeric["Day_Change_Numeric"] = df_numeric["Day_Change_%"].apply(
                    lambda x: float(x.rstrip("%"))
                )

                print("\n📈 TOP GAINERS (Next Day):")
                top_gainers = df_numeric.nlargest(
                    min(5, len(df_numeric)), "Day_Change_Numeric"
                )
                for _, row in top_gainers.iterrows():
                    print(
                        f"  {row['Ticker']}: {row['Current_Price']} → {row['Next_Day']} ({row['Day_Change_%']})"
                    )

                print("\n📉 TOP LOSERS (Next Day):")
                top_losers = df_numeric.nsmallest(
                    min(5, len(df_numeric)), "Day_Change_Numeric"
                )
                for _, row in top_losers.iterrows():
                    print(
                        f"  {row['Ticker']}: {row['Current_Price']} → {row['Next_Day']} ({row['Day_Change_%']})"
                    )

        if failed:
            print("\n❌ FAILED PREDICTIONS:")
            for r in failed:
                print(f"  {r['Ticker']}: {r.get('Error', 'Unknown error')}")

        print("\n" + "=" * 70)

    def save_to_csv(self, filename=None):
        """
        Save results to CSV file

        Args:
            filename (str): Output filename
        """
        if not self.results:
            print("No results to save")
            return

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"batch_predictions_{timestamp}.csv"

        successful = [r for r in self.results if r.get("Status") == "Success"]
        if successful:
            df = pd.DataFrame(successful)
            df.to_csv(filename, index=False)
            print(f"✓ Results saved to {filename}")
        else:
            print("No successful predictions to save")


def main():
    """
    Main function with example usage
    """
    print("\n" + "=" * 70)
    print("BATCH MARKET PREDICTOR")
    print("Compare multiple stocks and cryptocurrencies at once")
    print("=" * 70)

    # Example 1: Compare major tech stocks
    print("\n1. TECH STOCKS COMPARISON")
    tech_stocks = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "META"]
    batch1 = BatchPredictor(tech_stocks, period="1y", model_type="random_forest")
    batch1.predict_all(predict_days=7)
    batch1.display_summary()
    batch1.save_to_csv("tech_stocks_predictions.csv")

    # Example 2: Compare cryptocurrencies
    print("\n\n2. CRYPTOCURRENCY COMPARISON")
    cryptos = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "ADA-USD"]
    batch2 = BatchPredictor(cryptos, period="1y", model_type="gradient_boosting")
    batch2.predict_all(predict_days=7)
    batch2.display_summary()
    batch2.save_to_csv("crypto_predictions.csv")

    # Example 3: Custom list
    print("\n\n3. CUSTOM COMPARISON")
    custom_list = ["AAPL", "BTC-USD", "^GSPC"]  # Stock, Crypto, Index
    batch3 = BatchPredictor(custom_list, period="2y")
    batch3.predict_all(predict_days=7)
    batch3.display_summary()

    print("\n" + "=" * 70)
    print("💡 TIP: Edit this file to compare your own list of tickers!")
    print("=" * 70 + "\n")


# Predefined lists for easy use
POPULAR_STOCKS = [
    "AAPL",
    "GOOGL",
    "MSFT",
    "AMZN",
    "TSLA",
    "META",
    "NVDA",
    "JPM",
    "V",
    "WMT",
]

POPULAR_CRYPTO = [
    "BTC-USD",
    "ETH-USD",
    "BNB-USD",
    "XRP-USD",
    "SOL-USD",
    "ADA-USD",
    "DOGE-USD",
    "MATIC-USD",
]

MARKET_INDICES = ["^GSPC", "^DJI", "^IXIC", "^RUT"]  # S&P 500, Dow, NASDAQ, Russell

FAANG_STOCKS = ["META", "AAPL", "AMZN", "NFLX", "GOOGL"]

ENERGY_STOCKS = ["XOM", "CVX", "COP", "SLB", "EOG"]

BANK_STOCKS = ["JPM", "BAC", "WFC", "C", "GS"]


if __name__ == "__main__":
    # Choose what to run:

    # Option 1: Run all examples
    # main()

    # Option 2: Quick comparison of popular stocks
    print("\n🚀 Quick Stock Comparison")
    batch = BatchPredictor(POPULAR_STOCKS[:5], period="1y")  # Top 5 stocks
    batch.predict_all(predict_days=7)
    batch.display_summary()
    batch.save_to_csv()

    # Option 3: Your custom list
    # my_tickers = ["AAPL", "TSLA", "BTC-USD"]
    # batch = BatchPredictor(my_tickers, period="2y")
    # batch.predict_all(predict_days=7)
    # batch.display_summary()
