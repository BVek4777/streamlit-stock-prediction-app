import pandas as pd
import streamlit as st
from modules import data_loader, preprocess, utils, model
import time
from tensorflow.keras.models import load_model

# Load ticker CSV
@st.cache_data
def load_ticker_csv():
    url = "https://raw.githubusercontent.com/datasets/nasdaq-listings/master/data/nasdaq-listed-symbols.csv"
    df = pd.read_csv(url)
    df = df[['Symbol', 'Company Name']]
    df.dropna(inplace=True)
    return df

ticker_df = load_ticker_csv()
options = ticker_df.apply(lambda row: f"{row['Symbol']}-{row['Company Name']}", axis=1).tolist()
selected = st.selectbox("Search for a Stock", options, index=1859)
ticker = selected.split("-")[0]
Company_Name = selected.split("-")[1]

# --- Settings ---
st.header("1. Set Date Range and Model Parameters")
start_date = st.date_input("Start Date", value=pd.to_datetime("2024-04-30"))
end_date = st.date_input("End Date", value=pd.to_datetime("today"))
time_step = st.slider("Select Time Step (for LSTM sequence)", 10, 100, 60)
predict_days = st.slider("Days to Predict Ahead", 1, 60, 60)

if st.button("Load Historical Data"):
    with st.spinner('Loading historical data...'):
        time.sleep(1)
        df, data = data_loader.load_data(ticker, start_date, end_date)
        st.session_state.df = df
        st.session_state.data = data
        st.session_state.data_loaded = True
        st.session_state.loaded_start_date = start_date
        st.session_state.loaded_end_date = end_date
        st.session_state.loaded_ticker = ticker
    st.success(f"Data loaded for {ticker} from {start_date} to {end_date}")

# --- Show Data if Loaded ---
if st.session_state.get("data_loaded", False):

    # Check if ticker or date has changed
    ticker_changed = ticker != st.session_state.get("loaded_ticker")
    date_changed = (start_date != st.session_state.get("loaded_start_date") or
                    end_date != st.session_state.get("loaded_end_date"))

    # Show a single toast if either ticker or date changed
    if ticker_changed or date_changed:
        st.toast("You've changed either stock or dates range. Click 'Load Historical Data' again to update the dataset.", icon="⚠️")

    # Show the data in descending order of date
    st.write(st.session_state.df.sort_values(by='Date', ascending=False))

    # Plotting the closing price of the stock
    utils.plot_data(st.session_state.data, ticker)

    # Predict Future Prices
    st.header("2. Predict Future Prices")
    if st.button("Predict using LSTM"):

        data = st.session_state.get("data", None)
        if data is None:
            st.error("Data not found in session state. Please load data again.")
            st.stop()

        clean_df = preprocess.clean_data(data)
        scaled_data, scaler = preprocess.scale_data(clean_df)

        with st.spinner(f'Checking number of data points for time step ({time_step} days)...'):
            time.sleep(1)
            not_enough_data = len(scaled_data) <= time_step

        if not_enough_data:
            st.error(f"Not enough data for the selected time step ({time_step} days). Please lower the time step or select a longer date range.")
            st.stop()

        # Model building
        with st.spinner('Building LSTM Model...'):
            time.sleep(1)
            lstm_model = model.build_model(time_step)
        st.success("Model built successfully!")

        with st.spinner('Compiling the Model...'):
            time.sleep(1)
            lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        st.success("Model compiled successfully!")

        with st.spinner('Training the model...'):
            time.sleep(1)
            X, Y = preprocess.create_sequences(scaled_data, time_step)
            X_train, X_test, Y_train, Y_test = preprocess.split_data(X, Y)
            history = model.train_model(lstm_model, X_train, Y_train)
        st.success("Model trained successfully!")

        with st.spinner('Making predictions...'):
            time.sleep(1)
            Y_pred = model.predict(lstm_model, X_test)
        st.success("Model predicted successfully!")

        st.subheader("Predictions vs Actual Prices on Test data")
        utils.plot_predictions(Y_test, Y_pred, scaler)
        utils.show_metrics(Y_test, Y_pred, scaler)

        st.subheader("Forecast Future Prices")
        with st.spinner('Generating future forecast and plot...'):
            future_forecast = model.forecast_future(lstm_model, scaled_data, predict_days, time_step)
            utils.plot_forecast(future_forecast, scaler)

            future_forecast_actual = scaler.inverse_transform(future_forecast)
            day = range(1, predict_days + 1)
            future_forecast_df = pd.DataFrame({
                "Days": day,
                "Forecasted Price": future_forecast_actual.flatten()
            })
            future_forecast_df = future_forecast_df.reset_index(drop=True)

            st.write(future_forecast_df)
        # utils.download_forecast(future_forecast, ticker, scaler)

else:
    st.warning("Please load the historical data first by clicking the 'Load Historical Data' button.")
