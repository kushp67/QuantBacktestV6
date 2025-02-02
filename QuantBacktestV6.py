# -*- coding: utf-8 -*- 
"""
Created on Sat Feb  1 12:14:03 2025

Enhanced by Kush Patel
Updated by ChatGPT for additional position sizing, optimization, and bug fixes.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.api as sm
from datetime import datetime
from scipy.stats import norm, ttest_ind
from statsmodels.tsa.stattools import adfuller, acf
import itertools

# ------------------------------
# Set page configuration
# ------------------------------
st.set_page_config(
    page_title="QuantBacktest Pro",
    page_icon="📈",
    layout="wide",
)

# ------------------------------
# Advanced Custom CSS Styling
# ------------------------------
def local_css():
    st.markdown(
        """
        <style>
        /* Global Styles */
        html, body {
            background: #E7F7EE; /* Mint Green background */
            font-family: 'Open Sans', sans-serif;
            color: #333333;
            margin: 0;
            padding: 0;
        }
        .reportview-container .main {
            background: #E7F7EE;
        }
        /* Custom Header */
        header {
            background-color: #5EB583; /* Accent Green */
            padding: 1rem;
            text-align: center;
            color: #FFFFFF;
            font-size: 2.5rem;
            font-weight: 700;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-bottom: 2px solid #4AA371;
            margin-bottom: 1rem;
        }
        /* Sidebar Styling */
        .css-1d391kg { 
            background-color: #FFFFFF; /* Clean White */
            padding: 20px;
            border-radius: 10px;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
            border: 1px solid #E1F0E5;
        }
        /* Metric Cards */
        .metric-card {
            background: #FFFFFF;
            border: 1px solid #E1F0E5;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            padding: 20px;
            text-align: center;
            transition: transform 0.2s ease;
            margin: 10px;
        }
        .metric-card:hover {
            transform: scale(1.03);
        }
        .metric-card h3 {
            margin-bottom: 10px;
            color: #333333;
        }
        .metric-card p {
            font-size: 1.8rem;
            font-weight: 600;
            color: #5EB583;
        }
        /* Buttons */
        button, .stButton>button {
            background-color: #5EB583;
            color: #FFFFFF;
            border: none;
            border-radius: 6px;
            padding: 10px 20px;
            transition: background-color 0.3s ease, box-shadow 0.3s ease;
        }
        button:hover, .stButton>button:hover {
            background-color: #4AA371;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }
        /* Tab Headers */
        .stTabs [data-baseweb="tab-list"] button {
            font-weight: 600;
            color: #333333;
            background: #FFFFFF;
            border: 1px solid #E1F0E5;
            border-radius: 6px;
            padding: 8px 16px;
            margin-right: 4px;
            transition: background 0.3s ease;
        }
        .stTabs [data-baseweb="tab-list"] button:focus {
            outline: none;
        }
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
            background: #5EB583;
            color: #FFFFFF;
            border-color: #5EB583;
        }
        /* Chart Containers */
        .chart-container {
            background: #FFFFFF;
            border: 1px solid #E1F0E5;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        /* Links */
        a {
            color: #5EB583;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #E7F7EE;
        }
        ::-webkit-scrollbar-thumb {
            background: #C5E8D4;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #A9DAB6;
        }
        </style>
        """, unsafe_allow_html=True
    )

local_css()

# ------------------------------
# Data Acquisition Module
# ------------------------------
class DataFetcher:
    """
    Fetch historical stock price data using yfinance.
    """
    def __init__(self, ticker: str, start_date: str, end_date: str):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

    def fetch(self) -> pd.DataFrame:
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        if data.empty:
            raise ValueError("No data fetched. Please check ticker and date range.")
        data.dropna(inplace=True)
        return data

# ------------------------------
# Strategy Builder Module (Stock Strategies)
# ------------------------------
class Strategy:
    """
    Base Strategy class. Subclasses should implement generate_signals.
    """
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement generate_signals()")

class SMACrossoverStrategy(Strategy):
    def __init__(self, short_window: int = 50, long_window: int = 200):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['short_mavg'] = data['Close'].rolling(window=self.short_window, min_periods=1).mean()
        signals['long_mavg'] = data['Close'].rolling(window=self.long_window, min_periods=1).mean()
        signals['signal'] = np.where(signals['short_mavg'] > signals['long_mavg'], 1.0, 0.0)
        signals['positions'] = pd.Series(signals['signal']).diff().fillna(0.0)
        return signals

class RSITradingStrategy(Strategy):
    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def compute_rsi(self, data: pd.DataFrame) -> pd.Series:
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=self.period, min_periods=self.period).mean()
        avg_loss = loss.rolling(window=self.period, min_periods=self.period).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['rsi'] = self.compute_rsi(data)
        sig = [1 if r < self.oversold else 0 for r in signals['rsi']]
        signals['signal'] = sig
        signals['positions'] = pd.Series(sig, index=signals.index).diff().fillna(0.0)
        return signals

class BollingerBandsStrategy(Strategy):
    def __init__(self, window: int = 20, std_multiplier: float = 2.0):
        self.window = window
        self.std_multiplier = std_multiplier

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        rolling_mean = data['Close'].rolling(window=self.window, min_periods=1).mean()
        rolling_std = data['Close'].rolling(window=self.window, min_periods=1).std()
        upper_band = rolling_mean + self.std_multiplier * rolling_std
        lower_band = rolling_mean - self.std_multiplier * rolling_std
        sig = []
        current = 0
        for price, lb, ub in zip(data['Close'], lower_band, upper_band):
            if price < lb:
                current = 1
            elif price > ub:
                current = 0
            sig.append(current)
        signals['signal'] = sig
        signals['positions'] = pd.Series(sig, index=signals.index).diff().fillna(0.0)
        signals['rolling_mean'] = rolling_mean
        signals['upper_band'] = upper_band
        signals['lower_band'] = lower_band
        return signals

class SecondDerivativeMAStrategy(Strategy):
    def __init__(self, ma_window: int = 50, threshold: float = 0.1):
        self.ma_window = ma_window
        self.threshold = threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['ma'] = data['Close'].rolling(window=self.ma_window, min_periods=1).mean()
        signals['second_deriv'] = signals['ma'].diff().diff()
        sig = []
        prev = 0
        for val in signals['second_deriv']:
            if pd.isna(val):
                sig.append(prev)
            elif val > self.threshold:
                prev = 1
                sig.append(1)
            elif val < -self.threshold:
                prev = 0
                sig.append(0)
            else:
                sig.append(prev)
        signals['signal'] = sig
        signals['positions'] = pd.Series(sig, index=signals.index).diff().fillna(0.0)
        return signals

# ------------------------------
# Stock Backtesting Engine
# ------------------------------
class Backtester:
    def __init__(self, data: pd.DataFrame, signals: pd.DataFrame, initial_capital: float = 100000.0, shares: int = 100,
                 position_sizing: str = "fixed", risk_fraction: float = 0.01):
        self.data = data
        self.signals = signals
        self.initial_capital = initial_capital
        self.shares = shares
        self.position_sizing = position_sizing.lower()  # "fixed", "dynamic", or "fixed fractional"
        self.risk_fraction = risk_fraction

    def run_backtest(self) -> pd.DataFrame:
        cash = self.initial_capital
        position = 0
        portfolio_values = []
        # Iterate using index positions
        for i in range(len(self.data)):
            price = float(self.data['Close'].iloc[i])
            signal = self.signals['signal'].iloc[i]
            if position == 0 and signal == 1:
                if self.position_sizing == "fixed":
                    shares_to_buy = self.shares
                elif self.position_sizing == "dynamic":
                    shares_to_buy = int(cash // price)
                elif self.position_sizing == "fixed fractional":
                    shares_to_buy = int((cash * self.risk_fraction) // price)
                else:
                    shares_to_buy = self.shares
                cash -= shares_to_buy * price
                position = shares_to_buy
            elif position > 0 and signal == 0:
                cash += position * price
                position = 0
            total = cash + position * price
            portfolio_values.append(total)
        portfolio = pd.DataFrame(index=self.data.index, data={'total': portfolio_values})
        portfolio['returns'] = portfolio['total'].pct_change().fillna(0.0)
        return portfolio

    def run_backtest_custom(self, profit_target: float, stop_loss: float) -> pd.DataFrame:
        cash = self.initial_capital
        position = 0
        entry_price = None
        total_values = []
        for i in range(len(self.data)):
            price = float(self.data['Close'].iloc[i])
            signal = self.signals['signal'].iloc[i]
            if position == 0:
                if signal == 1:
                    if self.position_sizing == "fixed":
                        shares_to_buy = self.shares
                    elif self.position_sizing == "dynamic":
                        shares_to_buy = int(cash // price)
                    elif self.position_sizing == "fixed fractional":
                        shares_to_buy = int((cash * self.risk_fraction) // price)
                    else:
                        shares_to_buy = self.shares
                    position = shares_to_buy
                    entry_price = price
                    cash -= price * shares_to_buy
            else:
                if price >= entry_price * (1 + profit_target) or price <= entry_price * (1 - stop_loss) or signal == 0:
                    cash += position * price
                    position = 0
                    entry_price = None
            total = cash + position * price
            total_values.append(total)
        portfolio = pd.DataFrame(index=self.data.index)
        portfolio['total'] = total_values
        portfolio['returns'] = portfolio['total'].pct_change().fillna(0.0)
        return portfolio

# ------------------------------
# Advanced Analytics Functions
# ------------------------------
def compute_sharpe_ratio(returns, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate/252
    return np.sqrt(252) * excess_returns.mean() / (returns.std() + 1e-9)

def compute_sortino_ratio(returns, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate/252
    negative_returns = excess_returns[excess_returns < 0]
    downside_std = negative_returns.std()
    return np.sqrt(252) * excess_returns.mean() / (downside_std + 1e-9)

def compute_calmar_ratio(annual_return, max_drawdown):
    return annual_return / abs(max_drawdown) if max_drawdown != 0 else np.nan

def compute_drawdown_metrics(portfolio):
    total = portfolio['total']
    peak = total.cummax()
    drawdown = (total - peak) / peak
    max_drawdown = drawdown.min()
    avg_drawdown = drawdown[drawdown < 0].mean()
    recovery_times = []
    trough = total.iloc[0]
    trough_date = total.index[0]
    for date, value in total.items():
        if value < trough:
            trough = value
            trough_date = date
        if value >= peak.loc[date] and trough < value:
            recovery_times.append((date - trough_date).days)
            trough = value
    avg_recovery = np.mean(recovery_times) if recovery_times else np.nan
    return max_drawdown, avg_drawdown, avg_recovery

def monte_carlo_simulation(returns, initial_value, num_simulations, horizon):
    simulated_values = []
    daily_returns = np.ravel(returns.dropna().values)
    for _ in range(num_simulations):
        sim_return = np.random.choice(daily_returns, size=int(horizon), replace=True)
        sim_growth = np.prod(1 + sim_return)
        simulated_values.append(initial_value * sim_growth)
    return np.array(simulated_values)

def compute_VaR_CVaR(simulated_values, confidence_level=0.95):
    VaR = np.percentile(simulated_values, (1 - confidence_level) * 100)
    CVaR = simulated_values[simulated_values <= VaR].mean()
    return VaR, CVaR

def compute_greeks(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        delta = norm.cdf(d1)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365.0
    else:
        delta = norm.cdf(d1) - 1
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365.0
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100.0
    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}

def get_market_data(ticker="SPY", start_date="2020-01-01", end_date="2021-01-01"):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.dropna(inplace=True)
    return data

def perform_statistical_tests(strategy_returns, market_returns):
    a = np.ravel(strategy_returns.dropna().values)
    b = np.ravel(market_returns.dropna().values)
    t_stat, p_value = ttest_ind(a, b, equal_var=False)
    adf_result = adfuller(strategy_returns.dropna())
    autocorr = acf(strategy_returns.dropna(), nlags=20)
    return t_stat, p_value, adf_result, autocorr

def compute_beta(strategy_returns, market_returns):
    common_index = strategy_returns.index.intersection(market_returns.index)
    if len(common_index) < 2:
        return np.nan
    a = np.ravel(strategy_returns.loc[common_index].dropna().values)
    b = np.ravel(market_returns.loc[common_index].dropna().values)
    if len(b) < 2:
        return np.nan
    covariance = np.cov(a, b)[0, 1]
    variance = np.var(b)
    return covariance / variance if variance != 0 else np.nan

# ------------------------------
# Visualization Functions
# ------------------------------
def plot_results(portfolio: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(portfolio.index, portfolio['total'], label="Strategy Portfolio", color="#2980b9")
    ax.set_title("Portfolio Performance", fontsize=16, fontweight='600')
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Value")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    return fig

def plot_buy_hold_comparison(portfolio: pd.DataFrame, data: pd.DataFrame, initial_capital: float):
    bh_shares = initial_capital / data['Close'].iloc[0]
    buy_hold = bh_shares * data['Close']
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(portfolio.index, portfolio['total'], label="Strategy Portfolio", color="#27ae60")
    ax.plot(data.index, buy_hold, label="Buy & Hold", linestyle='--', color="#c0392b")
    ax.set_title("Strategy vs. Buy & Hold Comparison", fontsize=16, fontweight='600')
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    return fig

def plot_monte_carlo(simulated_values):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(simulated_values, bins=50, alpha=0.7, color='#f39c12')
    ax.set_title("Monte Carlo Simulation: Final Portfolio Values", fontsize=16, fontweight='600')
    ax.set_xlabel("Final Portfolio Value")
    ax.set_ylabel("Frequency")
    return fig

def plot_beta_comparison(strategy_returns, market_returns):
    if hasattr(strategy_returns, "squeeze"):
        strategy_returns = strategy_returns.squeeze()
    if hasattr(market_returns, "squeeze"):
        market_returns = market_returns.squeeze()
    df = pd.DataFrame({
        'Strategy Returns': strategy_returns,
        'Market Returns': market_returns
    }).dropna()
    fig = px.scatter(
        df, 
        x='Market Returns', 
        y='Strategy Returns',
        trendline='ols',
        title='Strategy vs. Market Returns (Interactive Beta Analysis)',
        labels={'Market Returns': 'Market Returns', 'Strategy Returns': 'Strategy Returns'}
    )
    return fig

def plot_qq(returns):
    fig, ax = plt.subplots(figsize=(8, 6))
    sm.qqplot(returns, line='s', ax=ax, alpha=0.5)
    ax.set_title('QQ-Plot of Strategy Returns', fontsize=16, fontweight='600')
    fig.tight_layout()
    return fig

def generate_report(portfolio, market_data, annual_return, max_dd, avg_dd, rec_time, sharpe, sortino, calmar, beta):
    report_dict = {
        "Total Return (%)": [(portfolio['total'].iloc[-1] - portfolio['total'].iloc[0]) / portfolio['total'].iloc[0] * 100],
        "Annualized Return (%)": [annual_return * 100],
        "Sharpe Ratio": [sharpe],
        "Sortino Ratio": [sortino],
        "Calmar Ratio": [calmar],
        "Max Drawdown (%)": [max_dd * 100],
        "Average Drawdown (%)": [avg_dd * 100],
        "Average Recovery Time (days)": [rec_time],
        "Portfolio Beta": [beta]
    }
    report_df = pd.DataFrame(report_dict)
    return report_df

# ------------------------------
# Optimization Functions
# ------------------------------
def optimize_strategy(strategy_name, data, initial_capital, shares, sizing_method, risk_fraction, metric="Total Return"):
    best_metric = -np.inf
    best_params = None
    best_portfolio = None

    if strategy_name in ["SMA Crossover", "Custom Profit/Stop"]:
        # Grid for SMA parameters
        for short_window in [20, 30, 40, 50]:
            for long_window in [100, 150, 200, 250]:
                if short_window >= long_window:
                    continue
                strat = SMACrossoverStrategy(short_window=short_window, long_window=long_window)
                signals = strat.generate_signals(data)
                backtester = Backtester(data, signals, initial_capital, shares, sizing_method, risk_fraction)
                if strategy_name == "Custom Profit/Stop":
                    for pt in [0.05, 0.10, 0.15]:
                        for sl in [0.03, 0.05, 0.10]:
                            portfolio = backtester.run_backtest_custom(profit_target=pt, stop_loss=sl)
                            total_return = (portfolio['total'].iloc[-1] - portfolio['total'].iloc[0]) / portfolio['total'].iloc[0]
                            days = (portfolio.index[-1] - portfolio.index[0]).days
                            annual_return = (1 + total_return) ** (365.0 / days) - 1 if days > 0 else np.nan
                            sharpe = compute_sharpe_ratio(portfolio['returns'])
                            score = total_return if metric=="Total Return" else annual_return if metric=="Annualized Return" else sharpe
                            if score > best_metric:
                                best_metric = score
                                best_params = {"short_window": short_window, "long_window": long_window,
                                               "profit_target": pt, "stop_loss": sl}
                                best_portfolio = portfolio
                else:
                    portfolio = backtester.run_backtest()
                    total_return = (portfolio['total'].iloc[-1] - portfolio['total'].iloc[0]) / portfolio['total'].iloc[0]
                    days = (portfolio.index[-1] - portfolio.index[0]).days
                    annual_return = (1 + total_return) ** (365.0 / days) - 1 if days > 0 else np.nan
                    sharpe = compute_sharpe_ratio(portfolio['returns'])
                    score = total_return if metric=="Total Return" else annual_return if metric=="Annualized Return" else sharpe
                    if score > best_metric:
                        best_metric = score
                        best_params = {"short_window": short_window, "long_window": long_window}
                        best_portfolio = portfolio

    elif strategy_name == "RSI Trading":
        for period in [10, 14, 20]:
            for oversold in [20, 30, 40]:
                for overbought in [60, 70, 80]:
                    if oversold >= overbought:
                        continue
                    strat = RSITradingStrategy(period=period, oversold=oversold, overbought=overbought)
                    signals = strat.generate_signals(data)
                    backtester = Backtester(data, signals, initial_capital, shares, sizing_method, risk_fraction)
                    portfolio = backtester.run_backtest()
                    total_return = (portfolio['total'].iloc[-1] - portfolio['total'].iloc[0]) / portfolio['total'].iloc[0]
                    days = (portfolio.index[-1] - portfolio.index[0]).days
                    annual_return = (1 + total_return) ** (365.0 / days) - 1 if days > 0 else np.nan
                    sharpe = compute_sharpe_ratio(portfolio['returns'])
                    score = total_return if metric=="Total Return" else annual_return if metric=="Annualized Return" else sharpe
                    if score > best_metric:
                        best_metric = score
                        best_params = {"period": period, "oversold": oversold, "overbought": overbought}
                        best_portfolio = portfolio

    elif strategy_name == "Bollinger Bands":
        for window in [10, 20, 30]:
            for std_multiplier in [1.5, 2.0, 2.5]:
                strat = BollingerBandsStrategy(window=window, std_multiplier=std_multiplier)
                signals = strat.generate_signals(data)
                backtester = Backtester(data, signals, initial_capital, shares, sizing_method, risk_fraction)
                portfolio = backtester.run_backtest()
                total_return = (portfolio['total'].iloc[-1] - portfolio['total'].iloc[0]) / portfolio['total'].iloc[0]
                days = (portfolio.index[-1] - portfolio.index[0]).days
                annual_return = (1 + total_return) ** (365.0 / days) - 1 if days > 0 else np.nan
                sharpe = compute_sharpe_ratio(portfolio['returns'])
                score = total_return if metric=="Total Return" else annual_return if metric=="Annualized Return" else sharpe
                if score > best_metric:
                    best_metric = score
                    best_params = {"window": window, "std_multiplier": std_multiplier}
                    best_portfolio = portfolio

    elif strategy_name == "Second Derivative MA":
        for ma_window in [10, 20, 30, 50]:
            for threshold in [0.1, 0.5, 1.0]:
                strat = SecondDerivativeMAStrategy(ma_window=ma_window, threshold=threshold)
                signals = strat.generate_signals(data)
                backtester = Backtester(data, signals, initial_capital, shares, sizing_method, risk_fraction)
                portfolio = backtester.run_backtest()
                total_return = (portfolio['total'].iloc[-1] - portfolio['total'].iloc[0]) / portfolio['total'].iloc[0]
                days = (portfolio.index[-1] - portfolio.index[0]).days
                annual_return = (1 + total_return) ** (365.0 / days) - 1 if days > 0 else np.nan
                sharpe = compute_sharpe_ratio(portfolio['returns'])
                score = total_return if metric=="Total Return" else annual_return if metric=="Annualized Return" else sharpe
                if score > best_metric:
                    best_metric = score
                    best_params = {"ma_window": ma_window, "threshold": threshold}
                    best_portfolio = portfolio

    return best_params, best_metric, best_portfolio

# ------------------------------
# Streamlit Web App
# ------------------------------
def main():
    # Custom Header
    st.markdown("<header>QuantBacktest Pro</header>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; font-size: 1.2rem; margin-bottom: 1rem;'>A professional quantitative backtesting platform for stocks trading strategies.</div>", unsafe_allow_html=True)

    # Sidebar – Enhanced Backtest Settings
    st.sidebar.header("Backtest Settings")
    ticker = st.sidebar.text_input("Ticker", value="AAPL")
    start_date = st.sidebar.date_input("Start Date", value=datetime(2020, 1, 1))
    end_date = st.sidebar.date_input("End Date", value=datetime(2021, 1, 1))
    initial_capital = st.sidebar.number_input("Initial Capital", value=100000.0)
    
    # New: Position Sizing Method selection and conditional Shares input
    position_sizing_method = st.sidebar.radio("Position Sizing Method", options=["Fixed", "Dynamic", "Fixed Fractional"])
    if position_sizing_method == "Fixed":
        shares = st.sidebar.number_input("Number of Shares", value=100, step=1)
    else:
        shares = 0  # Not used for dynamic or fixed fractional
    
    risk_fraction = 0.01  # default fraction
    if position_sizing_method == "Fixed Fractional":
        risk_fraction = st.sidebar.slider("Risk Fraction", min_value=0.01, max_value=0.5, value=0.01, step=0.01)

    # New: Optimization toggle
    optimize = st.sidebar.checkbox("Optimize Strategy Parameters")
    if optimize:
        metric_choice = st.sidebar.selectbox("Optimization Metric", ["Total Return", "Sharpe Ratio", "Annualized Return"])

    portfolio = None
    try:
        fetcher = DataFetcher(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        data = fetcher.fetch()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return

    # ------------------------------
    # Run Selected Strategy (with optional optimization)
    # ------------------------------
    if selected_strategy := st.sidebar.selectbox("Select Strategy", ["SMA Crossover", "RSI Trading", "Bollinger Bands",
                                                                      "Custom Profit/Stop", "Second Derivative MA"]):
        # If not optimizing, let the user set parameters manually:
        if not optimize:
            if selected_strategy in ["SMA Crossover", "Custom Profit/Stop"]:
                st.sidebar.subheader("SMA Parameters")
                sma_short = st.sidebar.slider("Short Window", min_value=5, max_value=100, value=50)
                sma_long = st.sidebar.slider("Long Window", min_value=20, max_value=300, value=200)
            if selected_strategy == "RSI Trading":
                st.sidebar.subheader("RSI Parameters")
                rsi_period = st.sidebar.slider("RSI Period", min_value=5, max_value=30, value=14)
                oversold = st.sidebar.slider("Oversold Threshold", min_value=10, max_value=50, value=30)
                overbought = st.sidebar.slider("Overbought Threshold", min_value=50, max_value=90, value=70)
            if selected_strategy == "Bollinger Bands":
                st.sidebar.subheader("Bollinger Bands Parameters")
                bb_window = st.sidebar.slider("Window", min_value=10, max_value=100, value=20)
                bb_std_multiplier = st.sidebar.slider("Std Dev Multiplier", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
            if selected_strategy == "Second Derivative MA":
                st.sidebar.subheader("Second Derivative MA Parameters")
                sd_ma_window = st.sidebar.slider("MA Window", min_value=5, max_value=100, value=50)
                sd_threshold = st.sidebar.slider("Threshold", min_value=0.0, max_value=10.0, value=0.1, step=0.1)
            if selected_strategy == "Custom Profit/Stop":
                st.sidebar.subheader("Profit-taking / Stop-loss Settings")
                profit_target = st.sidebar.slider("Profit Target (%)", min_value=1, max_value=50, value=10) / 100.0
                stop_loss = st.sidebar.slider("Stop Loss (%)", min_value=1, max_value=50, value=5) / 100.0

        if st.sidebar.button("Run Backtest"):
            with st.spinner("Running backtest..."):
                # Optimization branch
                if optimize:
                    best_params, best_metric, portfolio = optimize_strategy(
                        selected_strategy, data, initial_capital, shares,
                        position_sizing_method.lower(), risk_fraction,
                        metric=metric_choice
                    )
                    st.sidebar.markdown(f"**Optimal Parameters Found:** {best_params}")
                    st.sidebar.markdown(f"**Best {metric_choice}:** {best_metric:.4f}")
                else:
                    # Use manual parameters for each strategy
                    if selected_strategy in ["SMA Crossover", "Custom Profit/Stop"]:
                        strategy = SMACrossoverStrategy(short_window=sma_short, long_window=sma_long)
                        signals = strategy.generate_signals(data)
                        backtester = Backtester(data, signals, initial_capital, shares,
                                                  position_sizing=position_sizing_method.lower(),
                                                  risk_fraction=risk_fraction)
                        if selected_strategy == "Custom Profit/Stop":
                            portfolio = backtester.run_backtest_custom(profit_target, stop_loss)
                        else:
                            portfolio = backtester.run_backtest()
                    elif selected_strategy == "RSI Trading":
                        strategy = RSITradingStrategy(period=rsi_period, oversold=oversold, overbought=overbought)
                        signals = strategy.generate_signals(data)
                        backtester = Backtester(data, signals, initial_capital, shares,
                                                  position_sizing=position_sizing_method.lower(),
                                                  risk_fraction=risk_fraction)
                        portfolio = backtester.run_backtest()
                    elif selected_strategy == "Bollinger Bands":
                        strategy = BollingerBandsStrategy(window=bb_window, std_multiplier=bb_std_multiplier)
                        signals = strategy.generate_signals(data)
                        backtester = Backtester(data, signals, initial_capital, shares,
                                                  position_sizing=position_sizing_method.lower(),
                                                  risk_fraction=risk_fraction)
                        portfolio = backtester.run_backtest()
                    elif selected_strategy == "Second Derivative MA":
                        strategy = SecondDerivativeMAStrategy(ma_window=sd_ma_window, threshold=sd_threshold)
                        signals = strategy.generate_signals(data)
                        backtester = Backtester(data, signals, initial_capital, shares,
                                                  position_sizing=position_sizing_method.lower(),
                                                  risk_fraction=risk_fraction)
                        portfolio = backtester.run_backtest()

    # ------------------------------
    # Results & Visualization
    # ------------------------------
    if portfolio is not None:
        st.subheader("Performance Summary")
        total_return = (portfolio['total'].iloc[-1] - portfolio['total'].iloc[0]) / portfolio['total'].iloc[0]
        days = (portfolio.index[-1] - portfolio.index[0]).days
        annual_return = (1 + total_return) ** (365.0 / days) - 1 if days > 0 else np.nan
        sharpe = compute_sharpe_ratio(portfolio['returns'])

        # Display metric cards in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="metric-card"><h3>Total Return</h3><p>{total_return * 100:.2f}%</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><h3>Annualized Return</h3><p>{annual_return * 100:.2f}%</p></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><h3>Sharpe Ratio</h3><p>{sharpe:.2f}</p></div>', unsafe_allow_html=True)

        max_dd, avg_dd, rec_time = compute_drawdown_metrics(portfolio)
        sortino = compute_sortino_ratio(portfolio['returns'])
        calmar = compute_calmar_ratio(annual_return, max_dd)

        st.markdown("---")
       
                st.subheader("Strategy vs. Buy & Hold Comparison")
        fig2 = plot_buy_hold_comparison(portfolio, data, initial_capital)
        st.pyplot(fig2)

        # ------------------------------
        # Advanced Analytics Dashboard
        # ------------------------------
        st.markdown("### Advanced Analytics Dashboard")
        st.markdown("Explore detailed analytics in the tabs below:")

        market_data = get_market_data("SPY", start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        market_returns = market_data['Close'].pct_change().fillna(0.0)
        beta = compute_beta(portfolio['returns'], market_returns)

        tabs = st.tabs([
            "Performance Overview", 
            "Beta Analysis", 
            "QQ Plot", 
            "Monte Carlo Simulation", 
            "Statistical Edge", 
            "Hedge Optimization", 
            "Options Greeks", 
            "Export Report"
        ])

        with tabs[0]:
            st.markdown("#### Performance Overview")
            st.write(f"**Annualized Return (%):** {annual_return * 100:.2f}%")
            st.write(f"**Sharpe Ratio:** {sharpe:.2f}")
            st.write(f"**Sortino Ratio:** {sortino:.2f}")
            st.write(f"**Calmar Ratio:** {calmar:.2f}")
            st.write(f"**Max Drawdown (%):** {max_dd * 100:.2f}%")
            st.write(f"**Average Drawdown (%):** {avg_dd * 100:.2f}%")
            st.write(f"**Average Recovery Time (days):** {rec_time:.1f}")
            st.write(f"**Portfolio Beta (vs. SPY):** {beta:.2f}")

        with tabs[1]:
            st.markdown("#### Beta Analysis")
            beta_fig = plot_beta_comparison(portfolio['returns'], market_returns)
            st.plotly_chart(beta_fig, use_container_width=True)

        with tabs[2]:
            st.markdown("#### QQ Plot")
            qq_fig = plot_qq(portfolio['returns'].dropna())
            st.pyplot(qq_fig)

        with tabs[3]:
            st.markdown("#### Monte Carlo Simulation & Risk Metrics")
            num_simulations = 1000
            horizon = 252
            simulated_vals = monte_carlo_simulation(portfolio['returns'], portfolio['total'].iloc[-1], num_simulations, horizon)
            fig_mc = plot_monte_carlo(simulated_vals)
            st.pyplot(fig_mc)
            VaR, CVaR = compute_VaR_CVaR(simulated_vals, confidence_level=0.95)
            st.write(f"Value at Risk (95%): {VaR:.2f}")
            st.write(f"Conditional VaR (95%): {CVaR:.2f}")
            blowup_prob = np.mean(simulated_vals < initial_capital) * 100
            st.write(f"Blowup Probability (% final < initial): {blowup_prob:.2f}%")

        with tabs[4]:
            st.markdown("#### Statistical Edge & Market Comparison")
            t_stat, p_value, adf_result, autocorr = perform_statistical_tests(portfolio['returns'], market_returns)
            st.write(f"t-test Statistic: {t_stat:.2f}, p-value: {p_value:.4f}")
            st.write("ADF Test Result:")
            st.write(adf_result)
            st.write("Autocorrelation (first 10 lags):")
            st.write(autocorr[:10])

        with tabs[5]:
            st.markdown("#### Hedge Optimization")
            st.write(f"Portfolio Beta (vs. SPY): {beta:.2f}")
            if beta > 1:
                st.write("Suggestion: The portfolio is more volatile than SPY. Consider hedging with SPY put options or other risk mitigation strategies.")
            elif beta < 1:
                st.write("Suggestion: The portfolio is less volatile than SPY.")
            else:
                st.write("Suggestion: The portfolio beta is close to 1, matching the market.")
        
        with tabs[6]:
            st.markdown("#### Options Greeks Tracking")
            st.markdown("Enter parameters below to compute options Greeks:")
            S = st.number_input("Underlying Price (S)", value=float(data['Close'].iloc[-1]))
            K = st.number_input("Strike Price (K)", value=float(data['Close'].iloc[-1]))
            T = st.number_input("Time to Expiration (years)", value=0.25, step=0.01)
            r = st.number_input("Risk-Free Rate (annual)", value=0.02, step=0.001)
            sigma = st.number_input("Volatility (annual)", value=0.2, step=0.01)
            option_type = st.selectbox("Option Type", ["call", "put"])
            greeks = compute_greeks(S, K, T, r, sigma, option_type)
            st.write("Computed Options Greeks:")
            st.write(greeks)
        
        with tabs[7]:
            st.markdown("#### Export Summary Report")
            report_df = generate_report(portfolio, market_data, annual_return, max_dd, avg_dd, rec_time, sharpe, compute_sortino_ratio(portfolio['returns']), calmar, beta)
            st.dataframe(report_df)
            csv = report_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Report as CSV",
                data=csv,
                file_name='quantbacktest_report.csv',
                mime='text/csv'
            )

if __name__ == "__main__":
    main()
