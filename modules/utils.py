import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


def plot_data(df,ticker):
    st.subheader(f"{ticker} Closing Price Over Time")
    plt.figure(figsize=(10, 5))
    plt.plot(df['Date'], df['Close'], label='Close Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'{ticker} Closing Price Over Time')
    plt.legend()
    # plt.show()# this doesn't work in streamlit only works in jupyter notebook
    st.pyplot(plt)  # Display the plot in Streamlit app
    
    # st.line_chart(df['Close'])

def plot_predictions(Y_true, Y_pred, scaler):
    Y_true_inv = scaler.inverse_transform(Y_true)
    Y_pred_inv = scaler.inverse_transform(Y_pred)

    plt.figure(figsize=(10, 4))
    plt.plot(Y_true_inv, label='Actual')
    plt.plot(Y_pred_inv, label='Predicted')
    plt.legend()
    plt.title("Actual vs. Predicted Prices")
    st.pyplot(plt.gcf())

def show_metrics(Y_true, Y_pred, scaler):
    Y_true_inv = scaler.inverse_transform(Y_true)
    Y_pred_inv = scaler.inverse_transform(Y_pred)
    rmse = np.sqrt(mean_squared_error(Y_true_inv, Y_pred_inv))
    r2 = r2_score(Y_true_inv, Y_pred_inv)
    st.markdown(f"**RMSE:** {rmse:.2f}")
    st.markdown(f"**R² Score:** {r2:.2f}")


def plot_forecast(future_forecast, scaler):
    forecast_inv = scaler.inverse_transform(future_forecast)
    # Find max and min prices
    max_price = forecast_inv.max()
    min_price = forecast_inv.min()
    plt.figure(figsize=(12, 6))
    plt.plot(forecast_inv, marker='o', label='Forecast')
    # Add dashed lines for max and min
    plt.axhline(y=max_price, color='green', linestyle='--', label=f'Max: {max_price:.2f}')
    plt.axhline(y=min_price, color='red', linestyle='--', label=f'Min: {min_price:.2f}')
    plt.xlabel('Days Ahead')
    plt.ylabel('Price')
    plt.xticks(np.arange(0, len(forecast_inv), step=5))
    plt.title("Future Price Forecast")
    plt.legend()
    st.pyplot(plt.gcf())


def download_forecast(forecast, ticker, scaler):
    # Inverse scale
    forecast_inv = scaler.inverse_transform(forecast.reshape(-1, 1))
    df_forecast = pd.DataFrame(forecast_inv, columns=["Forecasted Price"])
    csv = df_forecast.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="📥 Download Forecast as CSV",
        data=csv,
        file_name=f'{ticker}_forecast.csv',
        mime='text/csv',
        key='download_forecast_btn'  # Add a stable key to prevent reloads
    )