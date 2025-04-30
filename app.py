import pandas as pd
import streamlit as st
from modules import data_loader, preprocess, utils, model
import time

# Load ticker CSV (can be from local or web source)
@st.cache_data
def load_ticker_csv():
    url = "https://raw.githubusercontent.com/datasets/nasdaq-listings/master/data/nasdaq-listed-symbols.csv"
    df = pd.read_csv(url)
    df = df[['Symbol', 'Company Name']]
    df.dropna(inplace=True)
    return df

ticker_df = load_ticker_csv()
# Build dropdown options
options = ticker_df.apply(lambda row: f"{row['Symbol']}-{row['Company Name']}", axis=1).tolist()
# Display dropdown in Streamlit app
selected = st.selectbox("Search for a Stock", options, index=1859)
# Extract the symbol
ticker = selected.split("-")[0]
Company_Name = selected.split("-")[1]

# --- Settings ---
st.header("1. Set Date Range and Model Parameters")
start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("today"))
time_step = st.slider("Select Time Step (for LSTM sequence)", 10, 100, 60)
predict_days = st.slider("Days to Predict Ahead", 1, 30, 7)

if st.button("Load Historical Data"):
    with st.spinner('Loading historical data...'):
        time.sleep(1)  # Add a 1-second delay
        df, data = data_loader.load_data(ticker, start_date, end_date)
        st.session_state.df = df
        st.session_state.data = data
        st.session_state.data_loaded = True
    st.success(f"Data loaded for {ticker} from {start_date} to {end_date}")

# --- Show Data if Loaded ---
if st.session_state.get("data_loaded", False):
    st.write(st.session_state.df.sort_values(by='Date', ascending=False))
    utils.plot_data(st.session_state.data, ticker)

    # --- Predict Future Prices ---
    st.header("2. Predict Future Prices")
    if st.button("Predict using LSTM"):
        # Show loading animation for model building
        with st.spinner('Building LSTM Model...'):
            time.sleep(1)
            lstm_model = model.build_model(time_step)
        st.success("Model built successfully!")

        # Show loading animation for model compilation
        with st.spinner('Compiling the Model...'):
            time.sleep(1)
            lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        st.success("Model compiled successfully!")

        # Show loading animation for model training
        with st.spinner('Training the model...'):
            time.sleep(1)
            clean_df = preprocess.clean_data(st.session_state.data)
            scaled_data, scaler = preprocess.scale_data(clean_df)

            # ✅ ADDITION: Check if enough data for selected time_step
            if len(scaled_data) <= time_step:
                st.error("Not enough data for the selected time step. Please lower the time step or select a longer date range.")
                st.stop()

            X, Y = preprocess.create_sequences(scaled_data, time_step)
            X_train, X_test, Y_train, Y_test = preprocess.split_data(X, Y)
            history = model.train_model(lstm_model, X_train, Y_train)
        st.success("Model trained successfully!")

        # Show loading animation for predictions
        with st.spinner('Making predictions...'):
            time.sleep(1)
            Y_pred = model.predict(lstm_model, X_test)
        st.success("Model Predicted successfully!")

        st.subheader("Predictions vs Actual Prices")
        utils.plot_predictions(Y_test, Y_pred, scaler)
        utils.show_metrics(Y_test, Y_pred, scaler)

        future_forecast = model.forecast_future(lstm_model, scaled_data, predict_days, time_step)
        utils.plot_forecast(future_forecast, scaler)
        utils.download_forecast(future_forecast, ticker, scaler)

else:
    st.warning("Please load the historical data first by clicking the 'Load Historical Data' button.")
