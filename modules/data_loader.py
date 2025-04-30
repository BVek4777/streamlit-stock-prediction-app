import yfinance as yf
import pandas as pd

#Download data
def load_data(ticker,start_date,end_date):
    # Download historical data from Yahoo Finance
    df=yf.download(ticker,start=start_date,end=end_date)
    df.reset_index(inplace=True)#make date a column instead of index
    df['Date'] = pd.to_datetime(df['Date']) #convert date to datetime format
    date_close_df=df[['Date','Close']] #keep only date and close columns
    return df,date_close_df #return both the original and date-close dataframes
