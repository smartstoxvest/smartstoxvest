import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
import io
from fpdf import FPDF
import pickle
import os
import requests
from textblob import TextBlob
import requests
import re
import shap
import warnings

# Enhancements for Smart Investment Dashboard
# Options 4 (Backtesting), 6 (Crypto-specific volatility), 8 (Explainability for LSTM)

# Import additional required libraries
import seaborn as sns

# Title
st.title("📈 Smart Investment Decision")

# Sidebar for user inputs
st.sidebar.header("Select Investment Preferences")
asset_type = st.sidebar.selectbox("Select Asset Type", ["Stock", "ETF", "Crypto"])
exchange = st.sidebar.selectbox("Select Exchange", ["LSE", "NASDAQ", "NYSE", "NSE", "Crypto"])
stocks_input = st.sidebar.text_area("Enter Stock Symbols (comma-separated)")

# Parse stock symbols
stocks = [stock.strip().upper() for stock in stocks_input.split(",") if stock.strip()]

# Exchange suffix mapping
exchange_suffix = {
    "LSE": ".L",
    "NASDAQ": "",
    "NYSE": "",
    "NSE": ".NS",
    "Crypto": "-USD"
}

# Time range selection
time_range = st.sidebar.selectbox("Select Historical Data Range", ["1y", "2y", "5y", "10y"])

# Risk tolerance selection
risk_tolerance = st.sidebar.slider("Select Risk Tolerance (Low → High)", 0.1, 2.0, 1.0, 0.1)

#User-defined Monte Carlo simulation runs
num_simulations = st.sidebar.slider("Monte Carlo Simulations", min_value=100, max_value=2000, value=500, step=100)


# Telegram Bot Credentials
TELEGRAM_BOT_TOKEN = "8045664062:AAEa70O6QgRa61xEHkPO8ON7y6zL5XrDm3g"
TELEGRAM_CHAT_ID = "7953882980"

# Function to Send Telegram Alert
def send_telegram_alert(stock, current_price, stop_loss, take_profit):
    message = f"""
    📢 **Stock Alert: {stock}**
    🔹 **Current Price:** {current_price:.2f}
    🛑 **Stop-Loss:** {stop_loss:.2f}
    🎯 **Take-Profit:** {take_profit:.2f}

    ⚠️ Action Required: Adjust your position accordingly!
    """

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    params = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        print(f"✅ Alert sent for {stock}")
    else:
        print(f"❌ Failed to send alert for {stock}")


@st.cache_data(ttl=3600)
def fetch_stock_data(stock, period, exchange):
    stock_with_suffix = stock + exchange_suffix.get(exchange, "")
    data = yf.download(stock_with_suffix, period=period)

    if data.empty or 'Close' not in data.columns:
        st.warning(f"⚠️ No data found for {stock}. Please check the symbol and exchange.")
        return None  # Return None instead of empty data
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]  # Flatten MultiIndex
    return data

# Calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(window=window, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=window, min_periods=1).mean()
    
    # Fix: Handle division by zero correctly
    avg_loss = np.where(avg_loss == 0, np.nan, avg_loss)  
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Fill NaN values
    data['RSI'] = pd.Series(rsi).fillna(method='bfill')
    return data

# Monte Carlo Simulation Function
def monte_carlo_simulation(data, days=252, simulations=1000):
    returns = data['Close'].pct_change().dropna()
    mean_return, std_dev = returns.mean(), returns.std()

    last_price = data['Close'].iloc[-1]
    random_returns = np.random.normal(mean_return, std_dev, (days, simulations))

    price_paths = np.zeros((days, simulations))
    price_paths[0] = last_price

    for t in range(1, days):
        price_paths[t] = price_paths[t - 1] * (1 + random_returns[t])

    # Calculate Value at Risk (5% worst-case scenario)
    var_5_percentile = np.percentile(price_paths[-1, :], 5)

    return price_paths, var_5_percentile

from sklearn.preprocessing import MinMaxScaler

# LSTM Model for Medium-Term Prediction
def prepare_lstm_data(data, sequence_length=50):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    x, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        x.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    
    return np.array(x), np.array(y), scaler

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(input_shape[1], 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def build_and_train_lstm_model(data, stock_name):
    model_file = f"lstm_model_{stock_name}.h5"

    if os.path.exists(model_file):
        model = load_model(model_file)
        return model, None  # No need to preprocess again

    # Only preprocess when training a new model
    x, y, scaler = prepare_lstm_data(data)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))

    model = build_lstm_model(x.shape)
    model.fit(x, y, epochs=5, batch_size=16, verbose=0)

    model.save(model_file)
    return model, scaler

# Calculate Average True Range (ATR)
def calculate_atr(data, window=14):
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())

    true_range = pd.DataFrame({'HL': high_low, 'HC': high_close, 'LC': low_close}).max(axis=1)
    atr = true_range.rolling(window=window).mean()
    
    data['ATR'] = atr
    return data

def calculate_sl_tp(data, risk_tolerance):
    data = calculate_atr(data)  # Compute ATR first

    current_price = data['Close'].iloc[-1]
    atr_value = data['ATR'].iloc[-1]

    # Define Stop-Loss & Take-Profit Multipliers
    sl_factor = 1.5 * (2 - risk_tolerance)  # Lower risk = tighter SL
    tp_factor = 2.5 * risk_tolerance  # Higher risk = wider TP

    stop_loss = current_price - (atr_value * sl_factor)
    take_profit = current_price + (atr_value * tp_factor)

    return stop_loss, take_profit



# --- BACKTESTING FUNCTIONALITY ---
def backtest_strategy(data, strategy_func, initial_capital=10000):
    cash = initial_capital
    shares = 0
    portfolio_value = []
    for i in range(1, len(data)):
        decision = strategy_func(data.iloc[:i])
        price = data['Close'].iloc[i]

        if decision == 'Buy' and cash >= price:
            shares = cash // price
            cash -= shares * price
        elif decision == 'Sell' and shares > 0:
            cash += shares * price
            shares = 0

        total_value = cash + shares * price
        portfolio_value.append(total_value)
    return portfolio_value

# Example basic strategy: buy when RSI < 30, sell when RSI > 70
def rsi_strategy(data):
    if 'RSI' not in data.columns:
        data = calculate_rsi(data)
    rsi = data['RSI'].iloc[-1]
    if rsi < 30:
        return 'Buy'
    elif rsi > 70:
        return 'Sell'
    else:
        return 'Hold'

# Add a backtesting button in Streamlit
def run_backtest_ui(stock, data):
    with st.expander("📈 Run Backtest for RSI Strategy"):
        if st.button(f"Start Backtest for {stock}"):
            portfolio = backtest_strategy(data, rsi_strategy)
            if portfolio:
                st.line_chart(portfolio)
                st.success(f"📊 Final Portfolio Value: {portfolio[-1]:.2f}")

# --- CRYPTO VOLATILITY HANDLING ---
def calculate_crypto_volatility(data):
    log_returns = np.log(data['Close'] / data['Close'].shift(1))
    volatility = log_returns.rolling(window=14).std() * np.sqrt(365)  # daily volatility annualized
    data['CryptoVolatility'] = volatility
    return data

# Example usage:
# if asset_type == 'Crypto':
#     data = calculate_crypto_volatility(data)

# --- LSTM EXPLAINABILITY ---
def lstm_feature_importance(data):
    st.subheader("🧠 LSTM Explainability")
    # Plot correlation heatmap
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)
    st.markdown("""
    🔍 **Interpretation:** Features most correlated with closing price can help guide the LSTM model.
    This chart shows how features like Volume, SMA, or RSI are influencing price trends.
    """)

# Fix NoneType bug + Add conflict debug

def fetch_news(stock):
    url = f"https://newsapi.org/v2/everything?q={stock}&sortBy=publishedAt&apiKey=3ddbc93a99894e9c82f5c4e4a497ce8f"
    response = requests.get(url)
    articles = response.json().get("articles", [])
    return articles[:5]

def analyze_sentiment(article_text):
    blob = TextBlob(article_text)
    return blob.sentiment.polarity

def get_news_decision(stock):
    articles = fetch_news(stock)

    # DEBUG: Log missing fields
    for article in articles:
        title = article.get('title') or ''
        description = article.get('description') or ''
        if not title or not description:
            print(f"⚠️ Missing data for {stock}: {article}")

    sentiment_scores = [
        analyze_sentiment((article.get('title') or '') + " " + (article.get('description') or ''))
        for article in articles
    ]

    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

    if avg_sentiment > 0.15:
        return "🟢 Positive News - Consider Buying", avg_sentiment
    elif avg_sentiment < -0.15:
        return "🔴 Negative News - Consider Selling", avg_sentiment
    else:
        return "🟡 Neutral News - Hold", avg_sentiment
        
# Function to clean emoji/text formatting for matching
def clean_decision_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^\w\s()\-]", "", text)  # remove emojis and symbols
    text = text.replace("\n", " ").strip()  # remove newlines, trim spaces
    return text

# Initialize session state for missing attributes
if 'previous_stocks' not in st.session_state or st.session_state.previous_stocks != stocks:
    st.session_state.portfolio_data = {}
    st.session_state.short_term_predictions = {}
    st.session_state.medium_term_predictions = {}
    st.session_state.long_term_predictions = {}
    st.session_state.portfolio_holdings = {}
    st.session_state.short_term_run = False
    st.session_state.medium_term_run = False
    st.session_state.long_term_run = False
    st.session_state.selected_stock = None
    st.session_state.previous_stocks = stocks

# Ensure all session state variables exist
for key in ['portfolio_data', 'short_term_predictions', 'medium_term_predictions', 'long_term_predictions',
            'short_term_run', 'medium_term_run', 'long_term_run']:
    if key not in st.session_state:
        st.session_state[key] = {}

# Create tabs for different investment strategies
tab1, tab2, tab3, tab4 = st.tabs(["📊 Short-Term", "🔮 Medium-Term", "📉 Long-Term", "📜 Important Info"])


if stocks:
    if not st.session_state.portfolio_data:
        st.session_state.portfolio_data = {
            stock: fetch_stock_data(stock, time_range, exchange)
            for stock in stocks
            if fetch_stock_data(stock, time_range, exchange) is not None
        }
        if asset_type == 'Crypto':
            for stock in st.session_state.portfolio_data:
                st.session_state.portfolio_data[stock] = calculate_crypto_volatility(
                    st.session_state.portfolio_data[stock]
                )

    
# Short-Term Analysis (Technical Indicators)

with tab1:
    st.subheader("📊 Short-Term Analysis")


    # ✅ Add toggle to enable/disable downgrade rule
    apply_mixed_signal_rule = st.sidebar.checkbox("Apply Mixed Signal Downgrade Rule", value=True)

    if st.button("Run Short-Term Analysis for All Stocks"):
        st.session_state.short_term_run = True
        for stock, data in st.session_state.portfolio_data.items():
            if data.empty or 'Close' not in data.columns:
                st.warning(f"No data found for {stock}")
                continue

            data['SMA50'] = data['Close'].rolling(window=50).mean()
            data['SMA200'] = data['Close'].rolling(window=200).mean()
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            data['Volatility'] = data['Close'].pct_change().rolling(14).std()

            if not data.empty and 'Close' in data.columns and not data['Close'].isna().all():
                st.session_state.short_term_predictions[stock] = data['Close'].dropna().iloc[-1] * 1.02

    if st.session_state.short_term_run:
        st.subheader("📋 Short-Term Stock Summary")
        st.info("ℹ️ **Mixed Signal**: Occurs when RSI is 'Overbought' but news is positive. This may indicate hype-driven price action. If enabled, the rule will downgrade the investment rating to 'Review Further'.")
        short_term_data = []
        for stock in stocks:
            if stock in st.session_state.portfolio_data and not st.session_state.portfolio_data[stock].empty:
                data = st.session_state.portfolio_data[stock]
                current_price = data['Close'].dropna().iloc[-1] if not data['Close'].isna().all() else None
                predicted_price = st.session_state.short_term_predictions.get(stock, None)
                rsi = data['RSI'].dropna().iloc[-1] if 'RSI' in data and not data['RSI'].isna().all() else None
                volatility = data['Volatility'].dropna().iloc[-1] if 'Volatility' in data and not data['Volatility'].isna().all() else None
                stop_loss, take_profit = calculate_sl_tp(data, risk_tolerance)

                news_decision, sentiment = get_news_decision(stock)

                if current_price >= stop_loss:
                    send_telegram_alert(stock, current_price, stop_loss, take_profit)
                    decision = "🔴 Stop-Loss Hit (Sell)"
                elif current_price >= take_profit:
                    send_telegram_alert(stock, current_price, stop_loss, take_profit)
                    decision = "🟢 Take-Profit Hit (Sell)"
                else:
                    decision = "✅ Hold"

                if rsi > 70:
                    rsi_status = "🔥 Overbought"
                elif rsi < 30:
                    rsi_status = "❄️ Oversold"
                else:
                    rsi_status = "Neutral"

                if predicted_price and predicted_price > current_price:
                    if rsi > 70:
                        decision = "⚠️ Hold (Overbought)"
                    elif rsi < 30:
                        decision = "✅ Invest (Buy Opportunity)"
                    else:
                        decision = "✅ Invest"
                else:
                    decision = "❌ Avoid"

                tech_weight = {
                    "Invest": 7,
                    "Invest (Buy Opportunity)": 9,
                    "Hold (Overbought)": 4,
                    "Hold": 3,
                    "Avoid": 1
                }

                news_weight = {
                    "Positive News - Consider Buying": 8,
                    "Neutral News - Hold": 5,
                    "Negative News - Consider Selling": 2
                }

                cleaned_decision = clean_decision_text(decision)
                cleaned_news_decision = clean_decision_text(news_decision)

                tech_score = tech_weight.get(cleaned_decision, 0)
                news_score = news_weight.get(cleaned_news_decision, 0)
                score = tech_score + news_score

                if score >= 14:
                    final_decision = "Invest Strongly"
                elif score >= 11:
                    final_decision = "Invest"
                elif score >= 8:
                    final_decision = "Review Further"
                else:
                    final_decision = "Hold or Avoid"

                # 🔁 New: Downgrade if conflict detected AND toggle is ON
                if rsi_status == "🔥 Overbought" and "Positive News" in news_decision:
                    signal_conflict = "⚠️ Mixed Signal"
                    if apply_mixed_signal_rule:
                        print(f"⚠️ Mixed Signal detected for {stock} — Downgrading decision to 'Review Further'")
                        final_decision = "Review Further"
                else:
                    signal_conflict = "✅ No Conflict"

                # 🧠 Extra logic: downgrade only if apply_mixed_signal_rule is enabled
                if apply_mixed_signal_rule:
                    if rsi_status == "🔥 Overbought" and cleaned_news_decision == "neutral news hold" and final_decision in ["Invest", "Invest Strongly"]:
                        print(f"📉 Downgrading {stock} due to RSI overbought and only neutral news.")
                        final_decision = "Review Further"

                short_term_data.append([stock, current_price, predicted_price, rsi, rsi_status, volatility, stop_loss, take_profit, decision, news_decision, final_decision, signal_conflict])

        short_term_df = pd.DataFrame(short_term_data, columns=["Stock", "Current Price", "Predicted Price", "RSI", "RSI Status", "Volatility", "Stop-Loss", "Take-Profit", "Decision", "News Decision", "Final Decision", "Signal Conflict"])
        # 🖌️ Highlight rows with Mixed Signals using Styler
        def highlight_conflicts(row):
            color = 'background-color: #fff3cd' if row['Signal Conflict'] == '⚠️ Mixed Signal' else ''
            return [color] * len(row)

        styled_df = short_term_df.style.apply(highlight_conflicts, axis=1)
        st.dataframe(styled_df, use_container_width=True)

        st.session_state.selected_short_term_stock = st.selectbox("Select a stock to view details", stocks, key="short_term_selectbox")

        if st.session_state.selected_short_term_stock:
            selected_stock = st.session_state.selected_short_term_stock
            st.subheader(f"Short-Term Analysis for {selected_stock}")
            data = st.session_state.portfolio_data[selected_stock]

            st.subheader("📈 Candlestick Chart")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'],
                                         low=data['Low'], close=data['Close'], name="Candlesticks"))
            st.plotly_chart(fig)

            if 'SMA50' in data.columns and 'SMA200' in data.columns:
                st.subheader("📊 Moving Averages (SMA50 & SMA200)")
                st.line_chart(data[['Close', 'SMA50', 'SMA200']])
            if 'RSI' in data.columns:
                st.subheader("📉 Relative Strength Index (RSI)")
                st.line_chart(data['RSI'])

             # 👉 ADD BACKTESTING UI RIGHT HERE
            #run_backtest_ui(selected_stock, data)
    selected_backtest_stock = st.selectbox(
            "Select a stock to backtest RSI strategy", stocks, key="backtest_selectbox"
    )
    if selected_backtest_stock:
            run_backtest_ui(
                selected_backtest_stock,
                st.session_state.portfolio_data[selected_backtest_stock]
            )    
    # Medium-Term Analysis (LSTM Predictions)

# Medium-Term Tab Rearranged
# Medium-Term Tab Final Update
# Medium-Term Tab Final Update
# Medium-Term Tab Final Update
# Medium-Term Tab Final Update
with tab2:
    st.subheader("🔮 Medium-Term Analysis")
    if st.button("Run Medium-Term Analysis for All Stocks"):
        st.session_state.medium_term_run = True
        medium_term_data = []

        for stock, data in st.session_state.portfolio_data.items():
            if not data.empty and 'Close' in data.columns:
                x, y, scaler = prepare_lstm_data(data)
                x = np.reshape(x, (x.shape[0], x.shape[1], 1))

                model = build_lstm_model(x.shape)
                model.fit(x, y, epochs=5, batch_size=16, verbose=0)

                last_sequence = x[-1].reshape(1, x.shape[1], 1)
                predicted_price_scaled = model.predict(last_sequence)[0][0]
                predicted_price = scaler.inverse_transform([[predicted_price_scaled]])[0][0]

                current_price = data['Close'].iloc[-1]
                decision = "✅ Buy" if predicted_price > current_price else "⚠️ Hold"
                medium_term_data.append([stock, current_price, predicted_price, decision])

        st.session_state.medium_term_df = pd.DataFrame(
            medium_term_data, columns=["Stock", "Current Price", "Predicted Price", "Decision"]
        )

    if "medium_term_df" in st.session_state and not st.session_state.medium_term_df.empty:
        st.subheader("📋 Medium-Term Stock Summary")
        st.dataframe(st.session_state.medium_term_df)

    st.subheader("📈 Individual Stock Price Prediction")
    selected_analysis_stock = st.selectbox(
        "Select a stock for price prediction analysis", stocks, key="lstm_analysis_stock"
    )
    if selected_analysis_stock:
        data = st.session_state.portfolio_data[selected_analysis_stock]
        if not data.empty and 'Close' in data.columns:
            x, y, scaler = prepare_lstm_data(data)
            x = np.reshape(x, (x.shape[0], x.shape[1], 1))

            model = build_lstm_model(x.shape)
            model.fit(x, y, epochs=5, batch_size=16, verbose=0)

            last_sequence = x[-1].reshape(1, x.shape[1], 1)
            predicted_price_scaled = model.predict(last_sequence)[0][0]
            predicted_price = scaler.inverse_transform([[predicted_price_scaled]])[0][0]

            current_price = data['Close'].iloc[-1]
            st.markdown(f"**Current Price:** ${current_price:.2f}")
            st.markdown(f"**Predicted Price:** ${predicted_price:.2f}")

            fig, ax = plt.subplots()
            ax.plot(data['Close'].values, label="Actual Price", linestyle='solid')
            ax.axhline(predicted_price, color='red', linestyle='dashed', label="Predicted Price")
            ax.set_title(f"{selected_analysis_stock} Price Prediction")
            ax.set_xlabel("Days")
            ax.set_ylabel("Price")
            ax.legend()
            st.pyplot(fig)

    st.subheader("🧠 LSTM Explainability")
    selected_explain_stock = st.selectbox(
        "Select a stock for LSTM explainability", stocks, key="explain_selectbox"
    )
    if selected_explain_stock:
        data = st.session_state.portfolio_data[selected_explain_stock]
        x, y, _ = prepare_lstm_data(data)
        x = np.reshape(x, (x.shape[0], x.shape[1], 1))

        model = build_lstm_model(x.shape)
        model.fit(x, y, epochs=5, batch_size=16, verbose=0)

        try:
            import shap
            import warnings
            background = x[:100].reshape((100, x.shape[1]))
            test_sample = x[-1:].reshape((1, x.shape[1]))
            explainer = shap.Explainer(model.predict, background)
            shap_values = explainer(test_sample)

            st.markdown("#### 🔍 SHAP Explainability")
            shap_fig = shap.plots.bar(shap_values[0], show=False)
            plt.tight_layout()
            st.pyplot(plt.gcf())
        except Exception as e:
            warnings.warn(str(e))
            st.warning(f"⚠️ SHAP explanation failed: {e}")

        st.markdown("#### 📊 Correlation Heatmap")
        import seaborn as sns
        corr = data.corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)
        st.markdown("""
        🔍 **Interpretation:**
        - SHAP shows which time steps influenced the prediction most.
        - Correlation heatmap shows which features are linearly related to the price.
        """)




    
# Long-Term Analysis (Monte Carlo Simulation)
with tab3:
    st.subheader("📉 Long-Term Risk Analysis")
    if st.button("Run Long-Term Analysis for All Stocks"):
        st.session_state.long_term_run = True
        for stock, data in st.session_state.portfolio_data.items():
            if data.empty or 'Close' not in data.columns:
                st.warning(f"No data found for {stock}")
                continue

            data['SMA200'] = data['Close'].rolling(window=200).mean()
            data['Volatility'] = data['Close'].pct_change().rolling(60).std()

            current_price = data['Close'].dropna().iloc[-1]
            predicted_price = current_price * 1.10  # Assuming 10% increase

            # Monte Carlo Simulation for risk assessment
            simulated_prices, var_5_percentile = monte_carlo_simulation(data, simulations=num_simulations)
            worst_case = var_5_percentile
            best_case = np.percentile(simulated_prices[-1, :], 95)

            if worst_case > current_price * 0.9:
                decision = "✅ Buy"
            elif worst_case > current_price * 0.75:
                decision = "⚠️ Hold"
            else:
                decision = "❌ Sell"

            st.session_state.long_term_predictions[stock] = {
                "Current Price": current_price,
                "Predicted Price": predicted_price,
                "SMA200": data['SMA200'].iloc[-1],
                "Volatility": data['Volatility'].iloc[-1],
                "Worst Case": worst_case,
                "Best Case": best_case,
                "Decision": decision
            }

    if st.session_state.long_term_run:
        st.subheader("📋 Long-Term Stock Summary")
        long_term_data = [[stock, data["Current Price"], data["Predicted Price"], data["SMA200"],
                           data["Volatility"], data["Worst Case"], data["Best Case"], data["Decision"]]
                          for stock, data in st.session_state.long_term_predictions.items()]
        long_term_df = pd.DataFrame(long_term_data,
                                    columns=["Stock", "Current Price", "Predicted Price", "SMA200", "Volatility", "Worst Case", "Best Case", "Decision"])
        st.dataframe(long_term_df)

# Investment Analysis Summary (Tab 4)
with tab4:
    st.subheader("📜 Investment Summary")
    st.markdown("""
    **Short-Term Analysis (Technical Indicators)**
    - Computes 50-day (SMA50) and 200-day (SMA200) moving averages.
    - Calculates RSI and volatility (standard deviation of % price change).
    - Predicts next-day price with a simple +2% assumption.
    - Categorizes stocks as “Invest”, “Hold”, or “Avoid” based on RSI and trend.

     - >>Typical Timeframe: Days to 6 months)
     - >>Who Uses It? Day traders, swing traders, active investors
     - >>RSI & Volatility: If RSI is above 70 (overbought), the stock might drop; if below 30 (oversold), it might rebound.
     - >>MA50 vs. SMA200:  If SMA50 > SMA200, stock is in an uptrend (bullish signal).
     - >>High Volatility:  High short-term risks and opportunities.

    **Medium-Term Analysis (LSTM-Based Prediction)**
    - Uses LSTM (Long Short-Term Memory) neural networks.
    - Estimates price changes over the next few months with a +5% assumption.
    - Determines stock recommendation: Buy, Hold, or Sell.

     - >>Typical Timeframe: 6 months to 3 years
     - >>Who Uses It? Retail investors, growth investors, fund managers
     - >>LSTM Model Predictions: Uses historical data to predict next stock movement.
     - >>SMA50 & SMA200 Trend Confirmation: If a stock is above its long-term moving averages, it's a strong buy.
     - >>Earnings Growth & Market Sentiment: A company with increasing profits is a good medium-term bet.

    **Long-Term Analysis (Monte Carlo Simulation)**
    - Simulates future stock prices using a Monte Carlo model.
    - Assumes stock price follows a normal distribution of returns.
    - Generates thousands of simulated paths to estimate worst-case (5th percentile) and best-case (95th percentile) scenarios.
    - Provides long-term buy/hold/sell decisions based on downside risk.
     - >>3+ years (usually 5-10 years)
     - >>Who Uses It? Long-term investors, pension funds, Warren Buffett-style investors
     - >>Monte Carlo Simulation: Predicts worst-case and best-case price movements over 5-10 years.
     - >>Company’s Financial Health: Revenue, debt, and profitability over time.
     - >>Macroeconomic Trends: Interest rates, inflation, industry growth.
     - >>Dividend & Value Investing: Stable companies with consistent dividends.
    """)
