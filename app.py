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

            usable_days = len(data)

            required_days = 100
            days_needed = max(0, required_days - usable_days)

            if usable_days < required_days:
                from datetime import datetime, timedelta

            estimated_date = datetime.today() + timedelta(days=days_needed * 1.4)

        st.warning(
        f"{ticker} currently has {usable_days} usable trading days.\n\n"
        f"At least {required_days} usable trading days are recommended.\n\n"
        f"Estimated trading days still needed: {days_needed}\n\n"
        f"Approximate forecast availability: {estimated_date.strftime('%B %d, %Y')}"
        )

        st.stop()

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

            price_value = data["Close"].iloc[-1]

            if hasattr(price_value, "iloc"):
                current_price = float(price_value.iloc[0])
            else:
                current_price = float(price_value)

            recent_returns = data["Close"].pct_change().dropna().tail(60)

            if hasattr(recent_returns, "columns"):
                recent_returns = recent_returns.iloc[:, 0]

            volatility = recent_returns.std()
            risk_score = min(10, max(1, round(float(volatility) * 200)))

            st.subheader(f"{ticker} Forecast")

            expected_20d_move = volatility * (20 ** 0.5)

            direction_adjustment = (probability - 0.5) * expected_20d_move

            estimated_20d_price = current_price * (1 + direction_adjustment)

            lower_price = current_price * (1 - expected_20d_move)
            upper_price = current_price * (1 + expected_20d_move)

            with st.expander("ℹ️ What do these metrics mean?"):
                st.write("""
                **Current Price**  
                Latest stock price pulled from Yahoo Finance.

                **Model Accuracy**  
                How often the model correctly predicted whether the stock was higher 20 trading days later during historical testing.

                **20-Day Probability**  
                The model’s estimated chance that the stock will be higher 20 trading days from now. This is not a guarantee, because sadly the market refuses to obey apps built on iPads.

                **Risk Score**  
                A volatility score from 1 to 10 based on recent price movement.  
                1 = steadier.  
                10 = very volatile.

                **Investment Score**  
                A combined score from 0 to 100 using probability, model accuracy, and risk. Higher is better, but it still depends on the model quality.

                **Estimated 20-Day Value**  
                A rough estimated price 20 trading days from now using the model probability and recent volatility.

                **Likely Range**  
                A volatility-based price range showing where the stock could reasonably move over the next 20 trading days.
                """)

            row1col1, row1col2, row1col3, row1col4 = st.columns(4)

            row2col1, row2col2, row2col3, row2col4 = st.columns(4)

            investment_score = round(
            (probability * 50) +
            (accuracy * 30) +
            ((10 - risk_score) / 10 * 20))

            investment_score = min(100, max(0, investment_score))

            confidence_score = round(
            accuracy * 100 * (1 - ((risk_score - 1) / 10))
            )

            confidence_score = min(100, max(0, confidence_score))

            row1col1.metric("Current Price", f"${current_price:,.2f}")
            row1col2.metric("Model Accuracy", f"{accuracy:.1%}")
            row1col3.metric("20-Day Probability", f"{probability:.1%}")
            row1col4.metric("Risk Score", f"{risk_score}/10")

            row2col1.metric("Investment Score", f"{investment_score}/100")
            row2col2.metric("Est. 20-Day Value", f"${estimated_20d_price:,.2f}")
            row2col3.markdown("**Likely Range**")
            row2col3.markdown(
                f"<h2>${lower_price:,.0f} - ${upper_price:,.0f}</h2>",
                unsafe_allow_html=True
)
            if confidence_score >= 75:
                confidence_label = "High"
            elif confidence_score >= 50:
                confidence_label = "Medium"
            else:
                confidence_label = "Low"
            
            row2col4.metric("Confidence", confidence_label)
    
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
