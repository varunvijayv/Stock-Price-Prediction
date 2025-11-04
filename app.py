# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

# custom function to fetch data (modern yfinance API)
def get_data_yahoo(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data.reset_index(inplace=True)
    return data

# date range
start = "2009-01-01"
end = "2023-01-01"

st.title('📈 Stock Closing Price Prediction')

# user input
user_input = st.text_input('Enter Stock Ticker (e.g., GOOGL, TSLA, RELIANCE.NS)', 'GOOGL')
df = get_data_yahoo(user_input, start, end)

st.subheader('Dated from 1st Jan, 2009 to 1st Jan, 2023')
st.write(df.describe())

# first plot
st.subheader('Closing Price Vs Time Chart')
fig1 = plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Close'], label="Closing Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig1)

# moving averages
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()

# second plot
st.subheader('Closing Price Vs Time Chart with 100-day Moving Average')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Close'], 'r', label="Per Day Closing")
plt.plot(df['Date'], ma100, 'g', label="100-Day MA")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig2)

# third plot
st.subheader('Closing Price Vs Time Chart with 100 & 200-day Moving Averages')
fig3 = plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Close'], 'r', label="Closing")
plt.plot(df['Date'], ma100, 'g', label="100-Day MA")
plt.plot(df['Date'], ma200, 'b', label="200-Day MA")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig3)

# prepare data for prediction
train_df = pd.DataFrame(df['Close'][0: int(len(df)*0.85)])
test_df = pd.DataFrame(df['Close'][int(len(df)*0.85):])

scaler = MinMaxScaler(feature_range=(0, 1))
train_df_arr = scaler.fit_transform(train_df)

# load model
model = load_model('keras_model.h5')

# prepare test data
past_100_days = train_df.tail(100)
final_df = pd.concat([past_100_days, test_df], ignore_index=True)

input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)

# predictions
y_pred = model.predict(x_test)

scale = scaler.scale_[0]
scale_factor = 1 / scale
y_pred = y_pred * scale_factor
y_test = y_test * scale_factor

# final plot
st.subheader('Predicted Vs Original Prices')
fig4 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'g', label="Original Price")
plt.plot(y_pred, 'r', label="Predicted Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig4)
