

# https://cryptobotapp.streamlit.app/




import streamlit as st
import yfinance as yf
import requests
from datetime import datetime, timedelta
from newsapi import NewsApiClient
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
from dotenv import load_dotenv
import os
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Try to import tensorflow, fallback to linear regression if not available
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    st.warning("TensorFlow not available. Using Ensemble methods for predictions.")

# Set your API key
newsapi = NewsApiClient(api_key = os.getenv('NEWSAPI_KEY'))


# ---------------- Page Setup ----------------
st.set_page_config(page_title="Crypto Bot", page_icon="💰", layout="centered")
st.title("💰 Crypto Bot")
st.markdown("Real-time price, historical charts, market summary, and news of your favorite crypto tokens.")

# ---------------- Input ----------------
symbol = st.text_input("🔎 Enter Crypto Symbol (e.g., BTC-USD, ETH-USD)", value="BTC-USD")
coin_id_map = {   
    "BTC": {"id": "bitcoin", "type": "crypto"},
    "ETH": {"id": "ethereum", "type": "crypto"},
    "SOL": {"id": "solana", "type": "crypto"},
    "BNB": {"id": "binancecoin", "type": "crypto"},
    "DOGE": {"id": "dogecoin", "type": "crypto"},
    "ADA": {"id": "cardano", "type": "crypto"},
    "TSLA": {"id": "TSLA", "type": "stock"},
    "AAPL": {"id": "AAPL", "type": "stock"},
    "GOOGL": {"id": "GOOGL", "type": "stock"},
    "MSFT": {"id": "MSFT", "type": "stock"},
    "AMZN": {"id": "AMZN", "type": "stock"},
    "NFLX": {"id": "NFLX", "type": "stock"},
    "AMD": {"id": "AMD", "type": "stock"},
    "XRP": {"id": "ripple", "type": "crypto"},
    "USDT": {"id": "tether", "type": "crypto"},
}

# Extract coin ID for CoinGecko
def get_coin_id(symbol):
    base = symbol.split("-")[0].upper()
    return coin_id_map.get(base, None)

# ---------------- Function: Get Crypto & Stock News ----------------
def get_news_articles(query):
    try:
        response = newsapi.get_everything(
            q=query,
            language='en',
            sort_by='publishedAt',
            page_size=5,
        )
        return response.get("articles", [])
    except Exception as e:
        return [{"title": "Error fetching news", "description": str(e), "url": "", "publishedAt": ""}]
# ..........................................................

# ---------------- Function: Real-Time Price ----------------
def get_realtime_price(sym):
    try:
        ticker = yf.Ticker(sym)
        hist = ticker.history(period="2d")
        if hist.empty:
            return {"error": "Invalid symbol or no data available."}
        latest_close = hist['Close'].iloc[-1]
        previous_close = hist['Close'].iloc[-2]
        percent_change = ((latest_close - previous_close) / previous_close) * 100
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return {
            "price": round(latest_close, 2),
            "change": round(percent_change, 2),
            "updated": timestamp
        }
    except Exception as e:
        return {"error": str(e)}
# ..........................................................

# ---------------- Function: Historical Price ----------------
def get_historical_data(sym, days):
    try:
        ticker = yf.Ticker(sym)
        hist = ticker.history(period=f"{days}d")
        if hist.empty:
            return {"error": "No historical data available."}
        hist = hist.reset_index()
        return {"data": hist}
    except Exception as e:
        return {"error": str(e)}

# ..........................................................

# ---------------- Function: Market Summary (CoinGecko) ----------------
def get_market_summary(symbol):
    symbol_base = symbol.split("-")[0].upper()
    coin_info = coin_id_map.get(symbol_base)

    if not coin_info:
        return {"error": "Asset not supported."}

    if coin_info["type"] == "crypto":
        # Basic summary via yfinance
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d")
            if hist.empty:
                return {"error": "Crypto data unavailable"}
            price = hist['Close'].iloc[-1]
            prev_price = hist['Close'].iloc[-2]
            change = ((price - prev_price) / prev_price) * 100
            return {
                "type": "crypto",
                "symbol": symbol,
                "name": coin_info["id"].capitalize(),
                "change_24h": round(change, 2),
                "homepage": f"https://www.coingecko.com/en/coins/{coin_info['id']}",
                "news": get_news_articles(symbol_base)
            }
        except Exception as e:
            return {"error": str(e)}

    else:
        # Stock summary
        try:
            ticker = yf.Ticker(coin_info["id"])
            hist = ticker.history(period="2d")
            info = ticker.info
            if hist.empty:
                return {"error": "Stock data unavailable"}
            price = hist['Close'].iloc[-1]
            prev_price = hist['Close'].iloc[-2]
            change = ((price - prev_price) / prev_price) * 100
            return {
                "type": "stock",
                "symbol": coin_info["id"],
                "name": info.get("shortName", symbol),
                "market_cap": info.get("marketCap", "N/A"),
                "volume": info.get("volume", "N/A"),
                "change_24h": round(change, 2),
                "homepage": f"https://finance.yahoo.com/quote/{coin_info['id']}",
                "news": get_news_articles(coin_info["id"])
            }
        except Exception as e:
            return {"error": str(e)}

# ..........................................................

# ---------------- Function: News Summary (CoinGecko) ----------------
def get_latest_crypto_news(coin_id):
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/status_updates"
        response = requests.get(url)
        if response.status_code != 200:
            return []

        data = response.json()
        updates = data.get("status_updates", [])

        # ✅ Filter for last 7 days
        seven_days_ago = datetime.utcnow() - timedelta(days=7)
        recent_news = []
        for item in updates:
            try:
                created_at = datetime.strptime(item['created_at'], "%Y-%m-%dT%H:%M:%S.%fZ")
                if created_at >= seven_days_ago:
                    recent_news.append(item)
            except Exception:
                continue

        return recent_news[:5]  # Return top 5
    except Exception as e:
        return []

# ..........................................................

# ---------------- Enhanced Technical Indicators ----------------
def calculate_technical_indicators(data):
    """Calculate various technical indicators"""
    df = data.copy()
    
    # Moving Averages
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / df['BB_Width']
    
    # Volume indicators
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    
    # Price change indicators
    df['Price_Change'] = df['Close'].pct_change()
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Close_Open_Ratio'] = df['Close'] / df['Open']
    
    return df

def prepare_enhanced_data(data, lookback_days=60):
    """Prepare enhanced data with technical indicators"""
    if len(data) < lookback_days + 30:
        return None, None, None, None
    
    # Calculate technical indicators
    enhanced_data = calculate_technical_indicators(data)
    
    # Select features
    feature_columns = [
        'Close', 'Volume', 'High', 'Low', 'Open',
        'SMA_5', 'SMA_10', 'SMA_20', 'EMA_12', 'EMA_26',
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
        'BB_Position', 'BB_Width', 'Volume_Ratio',
        'Price_Change', 'High_Low_Ratio', 'Close_Open_Ratio'
    ]
    
    # Remove NaN values
    enhanced_data = enhanced_data.dropna()
    
    if len(enhanced_data) < lookback_days + 10:
        return None, None, None, None
    
    # Prepare features and target
    features = enhanced_data[feature_columns].values
    target = enhanced_data['Close'].values
    
    # Scale the data
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    
    scaled_features = feature_scaler.fit_transform(features)
    scaled_target = target_scaler.fit_transform(target.reshape(-1, 1))
    
    # Create sequences
    X, y = [], []
    for i in range(lookback_days, len(scaled_features)):
        X.append(scaled_features[i-lookback_days:i])
        y.append(scaled_target[i, 0])
    
    return np.array(X), np.array(y), feature_scaler, target_scaler

# ..........................................................

# ---------------- Enhanced LSTM Model ----------------
def create_enhanced_lstm_model(lookback_days=60, n_features=20):
    """Create enhanced LSTM model with more features"""
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(lookback_days, n_features)),
        Dropout(0.3),
        LSTM(80, return_sequences=True),
        Dropout(0.3),
        GRU(60, return_sequences=False),
        Dropout(0.3),
        Dense(50, activation='relu'),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

# ..........................................................

# ---------------- Ensemble Prediction System ----------------
def ensemble_prediction(symbol, days_ahead=7, lookback_days=60):
    """Enhanced prediction using ensemble of multiple models"""
    try:
        # Get historical data
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="2y")
        
        if len(hist) < lookback_days + 50:
            return {"error": "Insufficient historical data for ensemble prediction"}
        
        # Prepare enhanced data
        X, y, feature_scaler, target_scaler = prepare_enhanced_data(hist, lookback_days)
        if X is None:
            return {"error": "Could not prepare enhanced prediction data"}
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        predictions_ensemble = []
        model_performances = []
        
        # Model 1: Enhanced LSTM (if TensorFlow available)
        if TENSORFLOW_AVAILABLE:
            try:
                lstm_model = create_enhanced_lstm_model(lookback_days, X.shape[2])
                lstm_model.fit(X_train, y_train, epochs=30, batch_size=32, 
                             validation_split=0.1, verbose=0)
                
                # Test performance
                lstm_pred_test = lstm_model.predict(X_test, verbose=0)
                lstm_mae = mean_absolute_error(y_test, lstm_pred_test)
                model_performances.append(('LSTM', 1.0 / (1.0 + lstm_mae)))
                
                # Make predictions
                last_sequence = X[-1].reshape(1, lookback_days, X.shape[2])
                lstm_predictions = []
                
                for _ in range(days_ahead):
                    pred = lstm_model.predict(last_sequence, verbose=0)
                    lstm_predictions.append(pred[0, 0])
                    
                    # Update sequence (simplified)
                    new_features = last_sequence[0, -1].copy()
                    new_features[0] = pred[0, 0]  # Update close price
                    last_sequence = np.roll(last_sequence, -1, axis=1)
                    last_sequence[0, -1] = new_features
                
                predictions_ensemble.append(('LSTM', lstm_predictions))
                
            except Exception as e:
                st.warning(f"LSTM model failed: {str(e)}")
        
        # Model 2: Random Forest
        try:
            # Flatten features for Random Forest
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train_flat, y_train)
            
            # Test performance
            rf_pred_test = rf_model.predict(X_test_flat)
            rf_mae = mean_absolute_error(y_test, rf_pred_test)
            model_performances.append(('Random Forest', 1.0 / (1.0 + rf_mae)))
            
            # Make predictions
            last_sequence_flat = X[-1].reshape(1, -1)
            rf_predictions = []
            
            for _ in range(days_ahead):
                pred = rf_model.predict(last_sequence_flat)
                rf_predictions.append(pred[0])
                
                # Simple update for next prediction
                last_sequence_flat = np.roll(last_sequence_flat, -X.shape[2])
                last_sequence_flat[0, -1] = pred[0]
            
            predictions_ensemble.append(('Random Forest', rf_predictions))
            
        except Exception as e:
            st.warning(f"Random Forest model failed: {str(e)}")
        
        # Model 3: Linear Regression with features
        try:
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            
            lr_model = LinearRegression()
            lr_model.fit(X_train_flat, y_train)
            
            # Test performance
            lr_pred_test = lr_model.predict(X_test_flat)
            lr_mae = mean_absolute_error(y_test, lr_pred_test)
            model_performances.append(('Linear Regression', 1.0 / (1.0 + lr_mae)))
            
            # Make predictions
            last_sequence_flat = X[-1].reshape(1, -1)
            lr_predictions = []
            
            for _ in range(days_ahead):
                pred = lr_model.predict(last_sequence_flat)
                lr_predictions.append(pred[0])
                
                last_sequence_flat = np.roll(last_sequence_flat, -X.shape[2])
                last_sequence_flat[0, -1] = pred[0]
            
            predictions_ensemble.append(('Linear Regression', lr_predictions))
            
        except Exception as e:
            st.warning(f"Linear Regression model failed: {str(e)}")
        
        if not predictions_ensemble:
            return {"error": "All models failed to generate predictions"}
        
        # Calculate weighted ensemble predictions
        total_weight = sum([perf[1] for perf in model_performances])
        weights = {perf[0]: perf[1]/total_weight for perf in model_performances}
        
        final_predictions = []
        for day in range(days_ahead):
            weighted_pred = 0
            for model_name, preds in predictions_ensemble:
                if model_name in weights:
                    weighted_pred += weights[model_name] * preds[day]
            final_predictions.append(weighted_pred)
        
        # Inverse transform predictions
        final_predictions = np.array(final_predictions).reshape(-1, 1)
        final_predictions = target_scaler.inverse_transform(final_predictions)
        
        # Calculate confidence intervals
        prediction_std = np.std([preds for _, preds in predictions_ensemble], axis=0)
        confidence_intervals = []
        for i, pred in enumerate(final_predictions.flatten()):
            std = prediction_std[i] if i < len(prediction_std) else prediction_std[-1]
            ci_lower = pred - (1.96 * std * target_scaler.scale_[0])
            ci_upper = pred + (1.96 * std * target_scaler.scale_[0])
            confidence_intervals.append((ci_lower, ci_upper))
        
        # Create future dates
        last_date = hist.index[-1]
        future_dates = [last_date + timedelta(days=i+1) for i in range(days_ahead)]
        
        return {
            "predictions": final_predictions.flatten().tolist(),
            "confidence_intervals": confidence_intervals,
            "dates": [date.strftime("%Y-%m-%d") for date in future_dates],
            "current_price": hist['Close'].iloc[-1],
            "model_performances": model_performances,
            "ensemble_weights": weights,
            "model_type": "Enhanced Ensemble"
        }
        
    except Exception as e:
        return {"error": f"Ensemble prediction failed: {str(e)}"}

# ..........................................................

# ---------------- Real-Time Price Section ----------------
st.subheader("📈 Real-Time Price")
if st.button("Check Real-Time Price"):
    with st.spinner("Fetching price..."):
        result = get_realtime_price(symbol.upper())
        if "error" in result:
            st.error(result["error"])
        else:
            st.success(f"💲 {symbol.upper()} Price: ${result['price']}")
            st.metric(label="24h Change", value=f"{result['change']}%")
            st.caption(f"Last updated: {result['updated']}")

# ..........................................................


# ---------------- Historical Chart Section ----------------
st.subheader("📊 Historical Price Chart")
days = st.selectbox("Select number of days", [1, 7, 14, 30], index=1, key="historical_days")
if st.button("Show Historical Data"):
    with st.spinner("Loading historical data..."):
        result = get_historical_data(symbol.upper(), days)
        if "error" in result:
            st.error(result["error"])
        else:
            df = result["data"]
            st.dataframe(df[['Date', 'Open', 'High', 'Low', 'Close']].set_index('Date'))
            st.line_chart(df.set_index('Date')['Close'])

# ..........................................................


# ---------------- Price Prediction Section ----------------
st.subheader("🔮 Price Prediction")
st.markdown("Advanced ML prediction using ensemble methods, technical indicators, and confidence intervals")

col1, col2 = st.columns(2)
with col1:
    prediction_days = st.selectbox("Select prediction period", [3, 7, 14, 30], index=1, key="enhanced_prediction_days")
with col2:
    show_confidence = st.checkbox("Show Confidence Intervals", value=True)

if st.button("🚀 Generate Enhanced Predictions"):
    symbol_base = symbol.split("-")[0].upper()
    
    if symbol_base in coin_id_map:
        with st.spinner(f"Running enhanced prediction for {symbol.upper()}..."):
            prediction_result = ensemble_prediction(symbol.upper(), prediction_days)
            
            if "error" in prediction_result:
                st.error(prediction_result["error"])
            else:
                st.success(f"✅ Enhanced prediction completed using {prediction_result['model_type']}")
                
                # Display current price
                current_price = prediction_result["current_price"]
                st.metric("Current Price", f"${current_price:.2f}")
                
                # Model performance display
                if "model_performances" in prediction_result:
                    st.subheader("🎯 Model Performance & Weights")
                    perf_df = pd.DataFrame(prediction_result["model_performances"], 
                                         columns=["Model", "Performance Score"])
                    st.dataframe(perf_df)
                    
                    weights_df = pd.DataFrame(list(prediction_result["ensemble_weights"].items()),
                                            columns=["Model", "Weight"])
                    st.bar_chart(weights_df.set_index("Model"))
                
                # Create enhanced prediction dataframe
                pred_data = {
                    'Date': prediction_result["dates"],
                    'Predicted Price': prediction_result["predictions"]
                }
                
                if show_confidence and "confidence_intervals" in prediction_result:
                    ci_lower = [ci[0] for ci in prediction_result["confidence_intervals"]]
                    ci_upper = [ci[1] for ci in prediction_result["confidence_intervals"]]
                    pred_data['Lower Bound (95%)'] = ci_lower
                    pred_data['Upper Bound (95%)'] = ci_upper
                
                pred_df = pd.DataFrame(pred_data)
                
                # Display predictions table
                st.subheader("📊 Enhanced Prediction Results")
                if show_confidence:
                    st.dataframe(pred_df.style.format({
                        'Predicted Price': '${:.2f}',
                        'Lower Bound (95%)': '${:.2f}',
                        'Upper Bound (95%)': '${:.2f}'
                    }))
                else:
                    st.dataframe(pred_df.style.format({'Predicted Price': '${:.2f}'}))
                
                # Enhanced insights
                predictions = prediction_result["predictions"]
                avg_predicted = np.mean(predictions)
                price_change = ((avg_predicted - current_price) / current_price) * 100
                
                # Volatility analysis
                volatility = np.std(predictions) / np.mean(predictions) * 100
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Average Predicted", f"${avg_predicted:.2f}")
                with col2:
                    st.metric("Expected Change", f"{price_change:+.2f}%")
                with col3:
                    trend = "📈 Bullish" if price_change > 2 else "📉 Bearish" if price_change < -2 else "➡️ Neutral"
                    st.metric("Trend Signal", trend)
                with col4:
                    vol_level = "High" if volatility > 5 else "Medium" if volatility > 2 else "Low"
                    st.metric("Volatility", f"{vol_level} ({volatility:.1f}%)")
                
                # Enhanced visualization
                st.subheader("📈 Enhanced Prediction Chart")
                
                # Get recent historical data
                ticker = yf.Ticker(symbol.upper())
                recent_hist = ticker.history(period="30d")
                
                # Create comprehensive chart data
                hist_dates = recent_hist.index.strftime('%Y-%m-%d').tolist()
                hist_prices = recent_hist['Close'].tolist()
                
                # Combine data
                all_dates = hist_dates + prediction_result["dates"]
                historical_prices = hist_prices + [None] * len(predictions)
                predicted_prices = [None] * len(hist_prices) + predictions
                
                chart_data = {
                    'Date': pd.to_datetime(all_dates),
                    'Historical Price': historical_prices,
                    'Predicted Price': predicted_prices
                }
                
                if show_confidence and "confidence_intervals" in prediction_result:
                    ci_lower_chart = [None] * len(hist_prices) + [ci[0] for ci in prediction_result["confidence_intervals"]]
                    ci_upper_chart = [None] * len(hist_prices) + [ci[1] for ci in prediction_result["confidence_intervals"]]
                    chart_data['Lower Bound'] = ci_lower_chart
                    chart_data['Upper Bound'] = ci_upper_chart
                
                chart_df = pd.DataFrame(chart_data).set_index('Date')
                st.line_chart(chart_df)
                
                # Risk assessment
                st.subheader("⚠️ Risk Assessment")

                # Initialize default values
                volatility = 0
                price_change = 0

                # Check if prediction results exist
                if 'predictions' in prediction_result and len(prediction_result['predictions']) > 0:
                    predictions = prediction_result["predictions"]
                    avg_predicted = np.mean(predictions)
                    price_change = ((avg_predicted - current_price) / current_price) * 100
                    volatility = np.std(predictions) / np.mean(predictions) * 100

                risk_factors = []
                if volatility > 5:
                    risk_factors.append("High volatility detected")
                if abs(price_change) > 10:
                    risk_factors.append("Large price movement predicted")
                if len(prediction_result.get("model_performances", [])) < 2:
                    risk_factors.append("Limited model consensus")

                if risk_factors:
                    for factor in risk_factors:
                        st.warning(f"⚠️ {factor}")
                else:
                    st.success("✅ Moderate risk level detected")
                
                # Enhanced disclaimer
                st.error("🚨 **Enhanced Disclaimer**: These predictions use advanced ML techniques but are still based on historical data. Market conditions can change rapidly due to news, regulations, or unexpected events. Use this as ONE factor in your analysis, not the sole basis for investment decisions. Always do your own research and consider your risk tolerance.")
    else:
        st.error("This asset is not supported for prediction. Please select from the supported assets.")

# ..........................................................


# ---------------- Multi-Asset Dashboard Section ----------------
st.subheader("📊 Enhanced Asset Analysis")
st.markdown("Detailed chart and prediction analysis for your selected asset")

def get_single_asset_enhanced_data(symbol):
    """Get enhanced data for a single asset based on user input"""
    try:
        # Extract base symbol (e.g., BTC from BTC-USD)
        base_symbol = symbol.split("-")[0].upper()
        
        # Check if asset is supported
        if base_symbol not in coin_id_map:
            return {"error": f"Asset {base_symbol} not supported"}
        
        asset_info = coin_id_map[base_symbol]
        
        # Determine correct ticker symbol
        if asset_info["type"] == "crypto":
            ticker_symbol = f"{base_symbol}-USD"
        else:  # stock
            ticker_symbol = asset_info["id"]
        
        # Get current price
        price_data = get_realtime_price(ticker_symbol)
        if "error" in price_data:
            return {"error": f"Could not fetch price data for {base_symbol}"}
        
        # Get historical data (60 days for better analysis)
        hist_data = get_historical_data(ticker_symbol, 60)
        if "error" in hist_data:
            return {"error": f"Could not fetch historical data for {base_symbol}"}
        
        return {
            "symbol": ticker_symbol,
            "base_symbol": base_symbol,
            "type": asset_info["type"],
            "current_price": price_data["price"],
            "change_24h": price_data["change"],
            "historical_data": hist_data["data"],
            "last_updated": price_data["updated"]
        }
        
    except Exception as e:
        return {"error": str(e)}

def generate_enhanced_single_prediction(asset_data, prediction_days=7):
    """Generate enhanced prediction for single asset"""
    try:
        # Generate prediction using ensemble method
        prediction_result = ensemble_prediction(asset_data["symbol"], prediction_days)
        
        if "error" in prediction_result:
            return {"error": prediction_result["error"]}
        
        return {
            "predictions": prediction_result["predictions"],
            "model_type": prediction_result["model_type"],
            "confidence_intervals": prediction_result.get("confidence_intervals", []),
            "model_performances": prediction_result.get("model_performances", []),
            "current_price": asset_data["current_price"],
            "symbol": asset_data["symbol"],
            "base_symbol": asset_data["base_symbol"],
            "type": asset_data["type"]
        }
        
    except Exception as e:
        return {"error": str(e)}

# Enhanced Analysis UI
col1, col2, col3 = st.columns(3)
with col1:
    analysis_type = st.selectbox(
        "Analysis Type", 
        ["Chart Analysis", "Prediction Analysis", "Complete Analysis"],
        key="enhanced_analysis_type"
    )
with col2:
    chart_period = st.selectbox(
        "Chart Period (Days)", 
        [30, 60, 90, 180], 
        index=1,
        key="enhanced_chart_period"
    )
with col3:
    enhanced_prediction_days = st.selectbox(
        "Prediction Period", 
        [3, 7, 14, 30], 
        index=1,
        key="enhanced_prediction_period"
    )

if st.button("🚀 Generate Enhanced Analysis"):
    # Use the symbol from the main input
    if not symbol:
        st.error("Please enter a symbol first")
    else:
        with st.spinner(f"Loading enhanced analysis for {symbol.upper()}..."):
            # Get enhanced data for the specific asset
            asset_data = get_single_asset_enhanced_data(symbol)
            
            if "error" in asset_data:
                st.error(asset_data["error"])
            else:
                st.success(f"✅ Enhanced analysis loaded for {asset_data['base_symbol']} ({asset_data['type'].title()})")
                
                # Display current metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        label="Current Price",
                        value=f"${asset_data['current_price']}",
                        delta=f"{asset_data['change_24h']:+.2f}%"
                    )
                with col2:
                    asset_type_emoji = "🪙" if asset_data['type'] == 'crypto' else "📈"
                    st.metric(label="Asset Type", value=f"{asset_type_emoji} {asset_data['type'].title()}")
                with col3:
                    st.metric(label="Symbol", value=asset_data['base_symbol'])
                with col4:
                    st.metric(label="Last Updated", value=asset_data['last_updated'].split(' ')[1][:5])
                
                # Chart Analysis
                if analysis_type in ["Chart Analysis", "Complete Analysis"]:
                    st.subheader("📊 Advanced Chart Analysis")
                    
                    # Get historical data for selected period
                    extended_hist = get_historical_data(asset_data['symbol'], chart_period)
                    if "error" not in extended_hist:
                        hist_df = extended_hist['data']
                        
                        # Create multiple chart views
                        chart_tab1, chart_tab2, chart_tab3 = st.tabs(["📈 Price Chart", "📊 Volume Analysis", "🔍 Technical Indicators"])
                        
                        with chart_tab1:
                            # OHLC Chart
                            st.markdown("**OHLC Price Movement**")
                            chart_data = hist_df.set_index('Date')[['Open', 'High', 'Low', 'Close']]
                            st.line_chart(chart_data)
                            
                            # Price statistics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Highest", f"${hist_df['High'].max():.2f}")
                            with col2:
                                st.metric("Lowest", f"${hist_df['Low'].min():.2f}")
                            with col3:
                                price_range = hist_df['High'].max() - hist_df['Low'].min()
                                st.metric("Range", f"${price_range:.2f}")
                            with col4:
                                avg_volume = hist_df['Volume'].mean()
                                st.metric("Avg Volume", f"{avg_volume:,.0f}")
                        
                        with chart_tab2:
                            # Volume Analysis
                            st.markdown("**Volume Trends**")
                            volume_data = hist_df.set_index('Date')['Volume']
                            st.bar_chart(volume_data)
                            
                            # Volume vs Price correlation
                            st.markdown("**Price vs Volume Correlation**")
                            corr_data = hist_df.set_index('Date')[['Close', 'Volume']]
                            st.line_chart(corr_data)
                        
                        with chart_tab3:
                            # Technical Indicators (simplified)
                            st.markdown("**Moving Averages**")
                            hist_df['MA_7'] = hist_df['Close'].rolling(window=7).mean()
                            hist_df['MA_21'] = hist_df['Close'].rolling(window=21).mean()
                            
                            ma_data = hist_df.set_index('Date')[['Close', 'MA_7', 'MA_21']]
                            st.line_chart(ma_data)
                            
                            # Current MA positions
                            current_price = asset_data['current_price']
                            current_ma7 = hist_df['MA_7'].iloc[-1]
                            current_ma21 = hist_df['MA_21'].iloc[-1]
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                ma7_signal = "🟢 Above" if current_price > current_ma7 else "🔴 Below"
                                st.metric("vs MA(7)", ma7_signal)
                            with col2:
                                ma21_signal = "🟢 Above" if current_price > current_ma21 else "🔴 Below"
                                st.metric("vs MA(21)", ma21_signal)
                            with col3:
                                trend = "📈 Bullish" if current_ma7 > current_ma21 else "📉 Bearish"
                                st.metric("Trend", trend)
                
                # Prediction Analysis
                if analysis_type in ["Prediction Analysis", "Complete Analysis"]:
                    st.subheader(f"🔮 {enhanced_prediction_days}-Day Prediction Analysis")
                    
                    with st.spinner("Generating advanced predictions..."):
                        prediction_data = generate_enhanced_single_prediction(asset_data, enhanced_prediction_days)
                        
                        if "error" in prediction_data:
                            st.error(prediction_data["error"])
                        else:
                            # Prediction Summary
                            predictions = prediction_data["predictions"]
                            current_price = prediction_data["current_price"]
                            avg_predicted = np.mean(predictions)
                            price_change = ((avg_predicted - current_price) / current_price) * 100
                            
                            # Key metrics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Current Price", f"${current_price:.2f}")
                            with col2:
                                st.metric("Predicted Price", f"${avg_predicted:.2f}")
                            with col3:
                                st.metric("Expected Change", f"{price_change:+.2f}%")
                            with col4:
                                trend_signal = "📈 Bullish" if price_change > 2 else "📉 Bearish" if price_change < -2 else "➡️ Neutral"
                                st.metric("Signal", trend_signal)
                            
                            # Prediction Chart
                            st.markdown("**Historical + Predicted Price Chart**")
                            
                            # Prepare combined chart data
                            hist_data = asset_data["historical_data"]
                            last_30_days = hist_data.tail(30)  # Show last 30 days + predictions
                            
                            chart_data = {
                                'Date': list(last_30_days['Date']) + [last_30_days['Date'].iloc[-1] + timedelta(days=i+1) for i in range(enhanced_prediction_days)],
                                'Historical Price': list(last_30_days['Close']) + [None] * enhanced_prediction_days,
                                'Predicted Price': [None] * len(last_30_days) + predictions
                            }
                            
                            # Add confidence intervals if available
                            if prediction_data.get("confidence_intervals"):
                                ci_lower, ci_upper = zip(*prediction_data["confidence_intervals"])
                                chart_data['Lower Bound'] = [None] * len(last_30_days) + list(ci_lower)
                                chart_data['Upper Bound'] = [None] * len(last_30_days) + list(ci_upper)
                            
                            chart_df = pd.DataFrame(chart_data).set_index('Date')
                            st.line_chart(chart_df)
                            
                            # Prediction Details Table
                            st.markdown("**Daily Predictions**")
                            pred_dates = [asset_data["historical_data"]['Date'].iloc[-1] + timedelta(days=i+1) for i in range(enhanced_prediction_days)]
                            
                            pred_table_data = {
                                'Date': [d.strftime('%Y-%m-%d') for d in pred_dates],
                                'Predicted Price': [f"${p:.2f}" for p in predictions],
                                'Change from Current': [f"{((p - current_price) / current_price * 100):+.2f}%" for p in predictions]
                            }
                            
                            if prediction_data.get("confidence_intervals"):
                                ci_lower, ci_upper = zip(*prediction_data["confidence_intervals"])
                                pred_table_data['Lower Bound'] = [f"${l:.2f}" for l in ci_lower]
                                pred_table_data['Upper Bound'] = [f"${u:.2f}" for u in ci_upper]
                            
                            pred_df = pd.DataFrame(pred_table_data)
                            st.dataframe(pred_df, use_container_width=True)
                            
                            # Model Performance
                            if prediction_data.get("model_performances"):
                                st.markdown("**Model Performance**")
                                perf_data = []
                                for model_name, score in prediction_data["model_performances"]:
                                    perf_data.append({
                                        'Model': model_name,
                                        'Performance Score': f"{score:.3f}",
                                        'Weight': f"{(score / sum([s[1] for s in prediction_data['model_performances']]) * 100):.1f}%"
                                    })
                                
                                perf_df = pd.DataFrame(perf_data)
                                st.dataframe(perf_df, use_container_width=True)
                            
                            # Risk Assessment for single asset
                            st.markdown("**Risk Assessment**")
                            volatility = np.std(predictions) / np.mean(predictions) * 100
                            
                            risk_factors = []
                            if volatility > 5:
                                risk_factors.append(f"High volatility detected ({volatility:.1f}%)")
                            if abs(price_change) > 10:
                                risk_factors.append(f"Large price movement predicted ({price_change:+.1f}%)")
                            if len(prediction_data.get("model_performances", [])) < 2:
                                risk_factors.append("Limited model consensus")
                            
                            if risk_factors:
                                for factor in risk_factors:
                                    st.warning(f"⚠️ {factor}")
                            else:
                                st.success("✅ Moderate risk level detected")
                            
                            # Enhanced disclaimer
                            st.info(f"📊 **Analysis Summary for {asset_data['base_symbol']}**: This prediction uses {prediction_data['model_type']} model with {enhanced_prediction_days}-day forecast. Current trend shows {trend_signal.split(' ')[1].lower()} sentiment with {volatility:.1f}% volatility.")

# ..........................................................

# ---------------- Market Summary + News Section ----------------
st.subheader("🧾 Market Summary & 📰 Latest News")
coin_id = get_coin_id(symbol)

symbol_base = symbol.split("-")[0].upper()

if symbol_base in coin_id_map:
    if st.button("Get Market Summary"):
        with st.spinner("Getting market data..."):
            summary = get_market_summary(symbol)
            if "error" in summary:
                st.error(summary["error"])
            else:
                st.markdown(f"### {summary['name']} ({summary['symbol']})")
                st.metric("24h Price Change", f"{summary['change_24h']}%")
                if summary["type"] == "stock":
                    if "market_cap" in summary and summary["market_cap"] != "N/A":
                        st.metric("Market Cap (USD)", f"${summary['market_cap']:,}")
                    if "volume" in summary and summary["volume"] != "N/A":
                        st.metric("24h Volume (USD)", f"${summary['volume']:,}")
                
                elif summary["type"] == "crypto":
                    if "market_cap" in summary and summary["market_cap"] != "N/A":
                        st.metric("Market Cap (USD)", f"${summary['market_cap']:,}")
                    if "volume" in summary and summary["volume"] != "N/A":
                        st.metric("24h Volume (USD)", f"${summary['volume']:,}")


                if summary["news"]:
                    st.markdown("### 📰 Latest News")
                    for article in summary["news"]:
                        st.markdown(f"**[{article['title']}]({article['url']})**")
                        st.caption(article.get("publishedAt", ""))
                        st.write(article.get("description") or "_No description provided._")
                        st.markdown("---")
                else:
                    st.info("No recent news found.")

else:
    st.warning("This asset is not yet supported.")

# ..........................................................