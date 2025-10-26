#!/usr/bin/env python3
"""
Market Utilities
Helper functions for market analysis and data processing
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")


def get_ticker_info(ticker):
    """
    Get detailed information about a ticker

    Args:
        ticker (str): Ticker symbol

    Returns:
        dict: Ticker information
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        return {
            "Symbol": ticker,
            "Name": info.get("longName", "N/A"),
            "Sector": info.get("sector", "N/A"),
            "Industry": info.get("industry", "N/A"),
            "Market Cap": info.get("marketCap", "N/A"),
            "Currency": info.get("currency", "USD"),
            "Exchange": info.get("exchange", "N/A"),
        }
    except Exception as e:
        return {"Symbol": ticker, "Error": str(e)}


def validate_ticker(ticker):
    """
    Check if a ticker symbol is valid

    Args:
        ticker (str): Ticker symbol

    Returns:
        bool: True if valid, False otherwise
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")
        return not hist.empty
    except:
        return False


def get_current_price(ticker):
    """
    Get the current/latest price for a ticker

    Args:
        ticker (str): Ticker symbol

    Returns:
        float: Current price
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if not hist.empty:
            return hist["Close"].iloc[-1]
        return None
    except:
        return None


def calculate_returns(prices, period="daily"):
    """
    Calculate returns from price series

    Args:
        prices (pd.Series): Price series
        period (str): 'daily', 'weekly', 'monthly'

    Returns:
        pd.Series: Returns
    """
    if period == "daily":
        return prices.pct_change()
    elif period == "weekly":
        return prices.resample("W").last().pct_change()
    elif period == "monthly":
        return prices.resample("M").last().pct_change()
    else:
        return prices.pct_change()


def calculate_volatility(prices, window=30):
    """
    Calculate rolling volatility

    Args:
        prices (pd.Series): Price series
        window (int): Rolling window size

    Returns:
        pd.Series: Volatility
    """
    returns = prices.pct_change()
    return returns.rolling(window=window).std() * np.sqrt(252)  # Annualized


def calculate_sharpe_ratio(prices, risk_free_rate=0.02):
    """
    Calculate Sharpe Ratio

    Args:
        prices (pd.Series): Price series
        risk_free_rate (float): Annual risk-free rate

    Returns:
        float: Sharpe ratio
    """
    returns = prices.pct_change().dropna()
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate

    if returns.std() == 0:
        return 0

    return np.sqrt(252) * excess_returns.mean() / returns.std()


def calculate_max_drawdown(prices):
    """
    Calculate maximum drawdown

    Args:
        prices (pd.Series): Price series

    Returns:
        float: Maximum drawdown as percentage
    """
    cumulative = (1 + prices.pct_change()).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min() * 100


def get_support_resistance(prices, window=20):
    """
    Calculate support and resistance levels

    Args:
        prices (pd.DataFrame): DataFrame with High, Low, Close
        window (int): Lookback window

    Returns:
        dict: Support and resistance levels
    """
    recent_high = prices["High"].tail(window).max()
    recent_low = prices["Low"].tail(window).min()
    current = prices["Close"].iloc[-1]

    return {
        "resistance": recent_high,
        "support": recent_low,
        "current": current,
        "distance_to_resistance": ((recent_high - current) / current) * 100,
        "distance_to_support": ((current - recent_low) / current) * 100,
    }


def calculate_moving_average_signal(prices, short_window=50, long_window=200):
    """
    Calculate moving average crossover signal

    Args:
        prices (pd.Series): Price series
        short_window (int): Short MA period
        long_window (int): Long MA period

    Returns:
        str: 'bullish', 'bearish', or 'neutral'
    """
    short_ma = prices.rolling(window=short_window).mean()
    long_ma = prices.rolling(window=long_window).mean()

    if len(prices) < long_window:
        return "neutral"

    current_short = short_ma.iloc[-1]
    current_long = long_ma.iloc[-1]
    prev_short = short_ma.iloc[-2]
    prev_long = long_ma.iloc[-2]

    # Golden cross (bullish)
    if prev_short <= prev_long and current_short > current_long:
        return "bullish"
    # Death cross (bearish)
    elif prev_short >= prev_long and current_short < current_long:
        return "bearish"
    # Currently above long MA
    elif current_short > current_long:
        return "bullish"
    # Currently below long MA
    else:
        return "bearish"


def calculate_rsi_signal(rsi_value):
    """
    Interpret RSI value

    Args:
        rsi_value (float): RSI value (0-100)

    Returns:
        str: 'oversold', 'overbought', or 'neutral'
    """
    if rsi_value < 30:
        return "oversold"
    elif rsi_value > 70:
        return "overbought"
    else:
        return "neutral"


def get_market_hours_status(ticker):
    """
    Check if market is open for a given ticker

    Args:
        ticker (str): Ticker symbol

    Returns:
        dict: Market status information
    """
    now = datetime.now()
    weekday = now.weekday()
    hour = now.hour

    # Determine market based on ticker
    if ticker.endswith("-USD"):  # Crypto
        is_open = True
        market_type = "Cryptocurrency (24/7)"
    else:  # Stocks (US market hours)
        is_open = (weekday < 5) and (9 <= hour < 16)  # Mon-Fri, 9am-4pm EST
        market_type = "Stock Market (US)"

    return {
        "is_open": is_open,
        "market_type": market_type,
        "current_time": now.strftime("%Y-%m-%d %H:%M:%S"),
    }


def compare_tickers(tickers, period="1y"):
    """
    Compare performance of multiple tickers

    Args:
        tickers (list): List of ticker symbols
        period (str): Time period

    Returns:
        pd.DataFrame: Comparison data
    """
    results = []

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)

            if hist.empty:
                continue

            start_price = hist["Close"].iloc[0]
            end_price = hist["Close"].iloc[-1]
            total_return = ((end_price - start_price) / start_price) * 100

            volatility = calculate_volatility(hist["Close"])
            sharpe = calculate_sharpe_ratio(hist["Close"])
            max_dd = calculate_max_drawdown(hist["Close"])

            results.append(
                {
                    "Ticker": ticker,
                    "Start Price": f"${start_price:.2f}",
                    "End Price": f"${end_price:.2f}",
                    "Total Return %": f"{total_return:.2f}%",
                    "Volatility": f"{volatility.iloc[-1]:.2f}"
                    if not volatility.empty
                    else "N/A",
                    "Sharpe Ratio": f"{sharpe:.2f}",
                    "Max Drawdown %": f"{max_dd:.2f}%",
                }
            )
        except Exception as e:
            results.append({"Ticker": ticker, "Error": str(e)})

    return pd.DataFrame(results)


def get_correlation_matrix(tickers, period="1y"):
    """
    Calculate correlation matrix for multiple tickers

    Args:
        tickers (list): List of ticker symbols
        period (str): Time period

    Returns:
        pd.DataFrame: Correlation matrix
    """
    price_data = {}

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            if not hist.empty:
                price_data[ticker] = hist["Close"]
        except:
            continue

    if not price_data:
        return None

    df = pd.DataFrame(price_data)
    return df.corr()


def detect_trend(prices, window=20):
    """
    Detect price trend

    Args:
        prices (pd.Series): Price series
        window (int): Window for trend calculation

    Returns:
        dict: Trend information
    """
    if len(prices) < window:
        return {"trend": "insufficient_data"}

    # Calculate linear regression slope
    recent_prices = prices.tail(window).values
    x = np.arange(len(recent_prices))
    slope = np.polyfit(x, recent_prices, 1)[0]

    # Calculate percentage change
    pct_change = ((prices.iloc[-1] - prices.iloc[-window]) / prices.iloc[-window]) * 100

    # Determine trend
    if abs(pct_change) < 2:
        trend = "sideways"
    elif pct_change > 0:
        if pct_change > 10:
            trend = "strong_uptrend"
        else:
            trend = "uptrend"
    else:
        if pct_change < -10:
            trend = "strong_downtrend"
        else:
            trend = "downtrend"

    return {
        "trend": trend,
        "slope": slope,
        "percent_change": pct_change,
        "window_days": window,
    }


def calculate_fibonacci_levels(high, low):
    """
    Calculate Fibonacci retracement levels

    Args:
        high (float): Period high
        low (float): Period low

    Returns:
        dict: Fibonacci levels
    """
    diff = high - low

    return {
        "0.0%": high,
        "23.6%": high - 0.236 * diff,
        "38.2%": high - 0.382 * diff,
        "50.0%": high - 0.500 * diff,
        "61.8%": high - 0.618 * diff,
        "78.6%": high - 0.786 * diff,
        "100.0%": low,
    }


def get_news_sentiment_placeholder(ticker):
    """
    Placeholder for news sentiment analysis
    Note: Real implementation would require news API (many are paid)

    Args:
        ticker (str): Ticker symbol

    Returns:
        dict: Sentiment information
    """
    return {
        "ticker": ticker,
        "sentiment": "neutral",
        "note": "News sentiment requires additional API integration",
        "suggestion": "Check financial news websites manually",
    }


def format_large_number(num):
    """
    Format large numbers with K, M, B, T suffixes

    Args:
        num (float): Number to format

    Returns:
        str: Formatted number
    """
    if num >= 1e12:
        return f"${num / 1e12:.2f}T"
    elif num >= 1e9:
        return f"${num / 1e9:.2f}B"
    elif num >= 1e6:
        return f"${num / 1e6:.2f}M"
    elif num >= 1e3:
        return f"${num / 1e3:.2f}K"
    else:
        return f"${num:.2f}"


def get_technical_summary(ticker, period="3mo"):
    """
    Get comprehensive technical summary for a ticker

    Args:
        ticker (str): Ticker symbol
        period (str): Time period

    Returns:
        dict: Technical analysis summary
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)

        if hist.empty:
            return {"error": "No data available"}

        current_price = hist["Close"].iloc[-1]

        # Calculate various indicators
        sma_50 = (
            hist["Close"].rolling(window=50).mean().iloc[-1]
            if len(hist) >= 50
            else None
        )
        sma_200 = (
            hist["Close"].rolling(window=200).mean().iloc[-1]
            if len(hist) >= 200
            else None
        )

        # RSI
        delta = hist["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).iloc[-1]

        # Trend
        trend_info = detect_trend(hist["Close"])

        # Support/Resistance
        sr = get_support_resistance(hist)

        # Volatility
        volatility = calculate_volatility(hist["Close"]).iloc[-1]

        # MA Signal
        ma_signal = (
            calculate_moving_average_signal(hist["Close"])
            if len(hist) >= 200
            else "insufficient_data"
        )

        return {
            "ticker": ticker,
            "current_price": f"${current_price:.2f}",
            "sma_50": f"${sma_50:.2f}" if sma_50 else "N/A",
            "sma_200": f"${sma_200:.2f}" if sma_200 else "N/A",
            "rsi": f"{rsi:.2f}",
            "rsi_signal": calculate_rsi_signal(rsi),
            "trend": trend_info["trend"],
            "trend_change": f"{trend_info['percent_change']:.2f}%",
            "support": f"${sr['support']:.2f}",
            "resistance": f"${sr['resistance']:.2f}",
            "volatility": f"{volatility:.2f}%",
            "ma_signal": ma_signal,
        }

    except Exception as e:
        return {"error": str(e)}


def print_technical_summary(ticker, period="3mo"):
    """
    Print formatted technical summary

    Args:
        ticker (str): Ticker symbol
        period (str): Time period
    """
    summary = get_technical_summary(ticker, period)

    if "error" in summary:
        print(f"Error: {summary['error']}")
        return

    print(f"\n{'=' * 60}")
    print(f"TECHNICAL SUMMARY - {ticker}")
    print(f"{'=' * 60}")
    print(f"Current Price:     {summary['current_price']}")
    print(f"50-Day SMA:        {summary['sma_50']}")
    print(f"200-Day SMA:       {summary['sma_200']}")
    print(f"RSI (14):          {summary['rsi']} ({summary['rsi_signal']})")
    print(f"Trend:             {summary['trend']}")
    print(f"Trend Change:      {summary['trend_change']}")
    print(f"Support Level:     {summary['support']}")
    print(f"Resistance Level:  {summary['resistance']}")
    print(f"Volatility:        {summary['volatility']}")
    print(f"MA Signal:         {summary['ma_signal']}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    """
    Example usage of utility functions
    """
    print("Market Utilities - Example Usage\n")

    # Example 1: Get ticker info
    print("1. Ticker Information:")
    info = get_ticker_info("AAPL")
    for key, value in info.items():
        print(f"   {key}: {value}")

    # Example 2: Validate ticker
    print("\n2. Ticker Validation:")
    print(f"   AAPL is valid: {validate_ticker('AAPL')}")
    print(f"   INVALID is valid: {validate_ticker('INVALID')}")

    # Example 3: Get current price
    print("\n3. Current Price:")
    price = get_current_price("AAPL")
    print(f"   AAPL: ${price:.2f}" if price else "   Unable to fetch price")

    # Example 4: Compare tickers
    print("\n4. Compare Tickers:")
    comparison = compare_tickers(["AAPL", "GOOGL", "MSFT"], period="1mo")
    print(comparison.to_string(index=False))

    # Example 5: Technical summary
    print("\n5. Technical Summary:")
    print_technical_summary("AAPL", period="3mo")
