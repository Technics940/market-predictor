#!/usr/bin/env python3
"""
Market Predictor - A Python program that predicts market trends using free services
Uses yfinance for data and machine learning for predictions
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


class MarketPredictor:
    """
    A class to predict stock/crypto market prices using machine learning
    """

    def __init__(self, ticker, period="2y", model_type="random_forest"):
        """
        Initialize the Market Predictor

        Args:
            ticker (str): Stock/Crypto ticker symbol (e.g., 'AAPL', 'BTC-USD')
            period (str): Historical data period ('1y', '2y', '5y', 'max')
            model_type (str): Type of ML model ('random_forest' or 'gradient_boosting')
        """
        self.ticker = ticker
        self.period = period
        self.model_type = model_type
        self.data = None
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()

    def fetch_data(self):
        """
        Fetch historical market data using yfinance (free service)
        """
        print(f"Fetching data for {self.ticker}...")
        try:
            stock = yf.Ticker(self.ticker)
            self.data = stock.history(period=self.period)

            if self.data.empty:
                raise ValueError(f"No data found for ticker {self.ticker}")

            print(f"✓ Successfully fetched {len(self.data)} days of data")
            print(
                f"  Date range: {self.data.index[0].date()} to {self.data.index[-1].date()}"
            )
            return True
        except Exception as e:
            print(f"✗ Error fetching data: {e}")
            return False

    def create_features(self):
        """
        Create technical indicators and features for prediction
        """
        print("Creating technical indicators...")

        df = self.data.copy()

        # Basic price features
        df["Returns"] = df["Close"].pct_change()
        df["Log_Returns"] = np.log(df["Close"] / df["Close"].shift(1))

        # Moving Averages
        df["SMA_5"] = df["Close"].rolling(window=5).mean()
        df["SMA_10"] = df["Close"].rolling(window=10).mean()
        df["SMA_20"] = df["Close"].rolling(window=20).mean()
        df["SMA_50"] = df["Close"].rolling(window=50).mean()

        # Exponential Moving Averages
        df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
        df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

        # MACD
        df["MACD"] = df["EMA_12"] - df["EMA_26"]
        df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

        # RSI (Relative Strength Index)
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df["BB_middle"] = df["Close"].rolling(window=20).mean()
        bb_std = df["Close"].rolling(window=20).std()
        df["BB_upper"] = df["BB_middle"] + (bb_std * 2)
        df["BB_lower"] = df["BB_middle"] - (bb_std * 2)
        df["BB_width"] = df["BB_upper"] - df["BB_lower"]

        # Volatility
        df["Volatility"] = df["Returns"].rolling(window=10).std()

        # Price momentum
        df["Momentum_5"] = df["Close"] - df["Close"].shift(5)
        df["Momentum_10"] = df["Close"] - df["Close"].shift(10)

        # Volume features
        df["Volume_SMA_5"] = df["Volume"].rolling(window=5).mean()
        df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA_5"]

        # High-Low range
        df["HL_Range"] = df["High"] - df["Low"]
        df["HL_Pct"] = (df["High"] - df["Low"]) / df["Close"]

        # Target: Next day's closing price
        df["Target"] = df["Close"].shift(-1)

        # Drop NaN values
        df.dropna(inplace=True)

        self.data = df
        print(f"✓ Created {len(df.columns)} features")

    def prepare_training_data(self):
        """
        Prepare data for training the model
        """
        print("Preparing training data...")

        # Select features
        feature_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Returns",
            "SMA_5",
            "SMA_10",
            "SMA_20",
            "SMA_50",
            "EMA_12",
            "EMA_26",
            "MACD",
            "Signal_Line",
            "RSI",
            "BB_middle",
            "BB_upper",
            "BB_lower",
            "BB_width",
            "Volatility",
            "Momentum_5",
            "Momentum_10",
            "Volume_SMA_5",
            "Volume_Ratio",
            "HL_Range",
            "HL_Pct",
        ]

        X = self.data[feature_columns].values
        y = self.data["Target"].values

        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)
        y_scaled = self.scaler.fit_transform(y.reshape(-1, 1)).flatten()

        # Split data (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.2, shuffle=False
        )

        print(f"✓ Training set: {len(X_train)} samples")
        print(f"✓ Test set: {len(X_test)} samples")

        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        """
        Train the machine learning model
        """
        print(f"Training {self.model_type} model...")

        if self.model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.model.fit(X_train, y_train)
        print("✓ Model training complete")

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate model performance
        """
        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)

        # Make predictions
        y_pred = self.model.predict(X_test)

        # Inverse transform to get actual prices
        y_test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_actual = self.scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

        # Calculate metrics
        mse = mean_squared_error(y_test_actual, y_pred_actual)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_actual, y_pred_actual)
        r2 = r2_score(y_test_actual, y_pred_actual)
        mape = np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100

        print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
        print(f"Mean Absolute Error (MAE): ${mae:.2f}")
        print(f"R² Score: {r2:.4f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        print("=" * 60 + "\n")

        return y_test_actual, y_pred_actual

    def predict_future(self, days=7):
        """
        Predict future prices
        """
        print(f"\n{'=' * 60}")
        print(f"FUTURE PREDICTIONS (Next {days} days)")
        print("=" * 60)

        # Get the latest data point
        latest_data = self.data.iloc[-1:].copy()
        predictions = []
        dates = []

        last_date = self.data.index[-1]
        last_close = self.data["Close"].iloc[-1]

        print(f"Current Price ({last_date.date()}): ${last_close:.2f}\n")

        feature_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Returns",
            "SMA_5",
            "SMA_10",
            "SMA_20",
            "SMA_50",
            "EMA_12",
            "EMA_26",
            "MACD",
            "Signal_Line",
            "RSI",
            "BB_middle",
            "BB_upper",
            "BB_lower",
            "BB_width",
            "Volatility",
            "Momentum_5",
            "Momentum_10",
            "Volume_SMA_5",
            "Volume_Ratio",
            "HL_Range",
            "HL_Pct",
        ]

        for i in range(days):
            # Prepare features
            X_pred = latest_data[feature_columns].values
            X_pred_scaled = self.feature_scaler.transform(X_pred)

            # Make prediction
            y_pred_scaled = self.model.predict(X_pred_scaled)
            y_pred = self.scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))[0][0]

            # Calculate prediction date
            pred_date = last_date + timedelta(days=i + 1)
            predictions.append(y_pred)
            dates.append(pred_date)

            # Calculate change
            if i == 0:
                change = y_pred - last_close
                change_pct = (change / last_close) * 100
            else:
                change = y_pred - predictions[i - 1]
                change_pct = (change / predictions[i - 1]) * 100

            direction = "↑" if change >= 0 else "↓"

            print(
                f"Day {i + 1} ({pred_date.date()}): ${y_pred:.2f} {direction} "
                f"({change:+.2f}, {change_pct:+.2f}%)"
            )

            # Update latest_data for next iteration (simple approach)
            # In a more sophisticated model, we would update all features
            latest_data["Close"] = y_pred

        print("=" * 60 + "\n")

        return dates, predictions

    def plot_results(
        self, y_test_actual, y_pred_actual, future_dates=None, future_predictions=None
    ):
        """
        Plot the results
        """
        print("Generating plots...")

        fig, axes = plt.subplots(2, 1, figsize=(15, 10))

        # Plot 1: Test Set Predictions vs Actual
        ax1 = axes[0]
        test_dates = self.data.index[-len(y_test_actual) :]
        ax1.plot(
            test_dates, y_test_actual, label="Actual Price", color="blue", linewidth=2
        )
        ax1.plot(
            test_dates,
            y_pred_actual,
            label="Predicted Price",
            color="red",
            linewidth=2,
            alpha=0.7,
        )
        ax1.set_title(
            f"{self.ticker} - Model Predictions vs Actual (Test Set)",
            fontsize=14,
            fontweight="bold",
        )
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price ($)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Historical + Future Predictions
        ax2 = axes[1]
        historical_dates = self.data.index[-60:]  # Last 60 days
        historical_prices = self.data["Close"].iloc[-60:].values

        ax2.plot(
            historical_dates,
            historical_prices,
            label="Historical Price",
            color="blue",
            linewidth=2,
        )

        if future_dates and future_predictions:
            # Connect last historical point to first prediction
            all_dates = list(historical_dates) + future_dates
            all_prices = list(historical_prices) + future_predictions

            ax2.plot(
                future_dates,
                future_predictions,
                label="Future Predictions",
                color="green",
                linewidth=2,
                linestyle="--",
                marker="o",
            )
            ax2.axvline(
                x=historical_dates[-1],
                color="gray",
                linestyle=":",
                alpha=0.5,
                label="Today",
            )

        ax2.set_title(
            f"{self.ticker} - Historical Price & Future Predictions",
            fontsize=14,
            fontweight="bold",
        )
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Price ($)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        filename = f"{self.ticker}_predictions.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"✓ Plot saved as '{filename}'")

        try:
            plt.show()
        except:
            print("  (Display not available, plot saved to file)")

    def run(self, predict_days=7, show_plots=True):
        """
        Run the complete prediction pipeline
        """
        print("\n" + "=" * 60)
        print(f"MARKET PREDICTOR - {self.ticker}")
        print("=" * 60 + "\n")

        # Step 1: Fetch data
        if not self.fetch_data():
            return False

        # Step 2: Create features
        self.create_features()

        # Step 3: Prepare training data
        X_train, X_test, y_train, y_test = self.prepare_training_data()

        # Step 4: Train model
        self.train_model(X_train, y_train)

        # Step 5: Evaluate model
        y_test_actual, y_pred_actual = self.evaluate_model(X_test, y_test)

        # Step 6: Predict future
        future_dates, future_predictions = self.predict_future(days=predict_days)

        # Step 7: Plot results
        if show_plots:
            self.plot_results(
                y_test_actual, y_pred_actual, future_dates, future_predictions
            )

        print("✓ Analysis complete!\n")
        return True


def main():
    """
    Main function with example usage
    """
    print("\n" + "=" * 60)
    print("MARKET PREDICTION PROGRAM")
    print("Using Free Services: yfinance + scikit-learn")
    print("=" * 60)

    # Example 1: Predict Apple stock (AAPL)
    print("\n1. Predicting Apple Stock (AAPL)...")
    predictor_aapl = MarketPredictor(
        ticker="AAPL", period="2y", model_type="random_forest"
    )
    predictor_aapl.run(predict_days=7, show_plots=False)

    # Example 2: Predict Bitcoin (BTC-USD)
    print("\n2. Predicting Bitcoin (BTC-USD)...")
    predictor_btc = MarketPredictor(
        ticker="BTC-USD", period="1y", model_type="gradient_boosting"
    )
    predictor_btc.run(predict_days=7, show_plots=False)

    # Example 3: User input
    print("\n" + "=" * 60)
    print("CUSTOM PREDICTION")
    print("=" * 60)
    print("\nAvailable tickers:")
    print("  Stocks: AAPL, GOOGL, MSFT, TSLA, AMZN, META, NVDA, etc.")
    print("  Crypto: BTC-USD, ETH-USD, etc.")
    print("  Indices: ^GSPC (S&P 500), ^DJI (Dow Jones), ^IXIC (NASDAQ)")

    try:
        ticker = (
            input("\nEnter ticker symbol (or press Enter for AAPL): ").strip().upper()
        )
        if not ticker:
            ticker = "AAPL"

        model_type = (
            input(
                "Choose model (random_forest/gradient_boosting) [default: random_forest]: "
            )
            .strip()
            .lower()
        )
        if model_type not in ["random_forest", "gradient_boosting"]:
            model_type = "random_forest"

        predictor = MarketPredictor(ticker=ticker, period="2y", model_type=model_type)
        predictor.run(predict_days=7, show_plots=True)

    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
