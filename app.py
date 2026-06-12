import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="AI Stock Analyzer", page_icon="📈")

st.title("📈 AI Stock Analyzer")
st.write("Enter a stock or ETF ticker and get a basic 20-day forecast.")

ticker = st.text_input("Enter ticker", value="VTI").upper()

if st.button("Analyze"):
    with st.spinner(f"Analyzing {ticker}..."):
        data = yf.download(ticker, period="10y", auto_adjust=True)

        if data.empty:
            st.error("No data found. Check the ticker.")
        else:
            data["Return_1d"] = data["Close"].pct_change()
            data["Return_5d"] = data["Close"].pct_change(5)
            data["Return_20d"] = data["Close"].pct_change(20)
            data["MA_20"] = data["Close"].rolling(20).mean()
            data["MA_50"] = data["Close"].rolling(50).mean()

            data["Target"] = (data["Close"].shift(-20) > data["Close"]).astype(int)
            data = data.dropna()

            features = ["Return_1d", "Return_5d", "Return_20d", "MA_20", "MA_50"]

            X = data[features]
            y = data["Target"]

            split = int(len(data) * 0.8)

            X_train = X.iloc[:split]
            X_test = X.iloc[split:]
            y_train = y.iloc[:split]
            y_test = y.iloc[split:]

            model = RandomForestClassifier(n_estimators=200, random_state=42)
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)

            latest = X.iloc[-1:]
            probability = model.predict_proba(latest)[0][1]

            st.subheader(f"{ticker} Forecast")

            col1, col2 = st.columns(2)

            col1.metric("Model accuracy", f"{accuracy:.1%}")
            col2.metric("Chance higher in 20 trading days", f"{probability:.1%}")

            st.progress(float(probability))

            if probability >= 0.65 and accuracy >= 0.55:
                rating = "Potential Buy"
            elif probability >= 0.50:
                rating = "Hold / Watch"
            else:
                rating = "Avoid / Wait"

            st.subheader(f"Rating: {rating}")

            st.line_chart(data["Close"].tail(252))

            st.caption(
                "Educational model only, not financial advice. "
                "The market remains annoyingly allergic to certainty."
            
            )
