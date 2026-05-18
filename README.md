# Stock Price Prediction Using LSTM

## Table of Contents

* [Overview](#overview)
* [Project Structure](#project-structure)
* [Technologies Used](#technologies-used)
* [Installation](#installation)
* [Usage](#usage)
* [Model Information](#model-information)
* [Deployment](#deployment)

---

## Overview

This project predicts stock closing prices using **Long Short-Term Memory (LSTM)** neural networks. The application fetches stock market data using the **yfinance** library, processes the data, and predicts future stock prices using a trained deep learning model built with **Keras** and **TensorFlow**.

The project also includes a **Streamlit** web application that provides an interactive user interface for visualizing stock data and predictions.

### Features

* Fetch real-time stock market data from Yahoo Finance
* Data preprocessing and visualization
* LSTM-based stock price prediction
* Interactive Streamlit web application
* Pre-trained Keras model included

---

## Project Structure

```bash
├── image/
├── LSTM Model.ipynb      # Jupyter Notebook for data preprocessing, training, and testing
├── app.py                # Streamlit application
├── keras_model.h5        # Pre-trained LSTM model
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* yfinance
* TensorFlow
* Keras
* Streamlit

---

## Installation

Install the required libraries using pip:

```bash
pip install pandas
pip install numpy
pip install matplotlib
pip install pandas_datareader
pip install yfinance
pip install streamlit
pip install tensorflow
pip install keras
```

---

## Usage

### Import Required Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
from pandas_datareader import data as pdr
import yfinance as yf
```

---

### Fetch Stock Market Data

```python
yf.pdr_override()

df = pdr.get_data_yahoo(ticker, start_date, end_date)
```

---

### Run the Streamlit App

```bash
streamlit run app.py
```

After running the command, Streamlit will provide a local URL in the terminal to access the application in your browser.

---

## Model Information

* The project uses an **LSTM (Long Short-Term Memory)** neural network.
* LSTM is a specialized type of **Recurrent Neural Network (RNN)** designed for sequence prediction tasks.
* The trained model is saved as:

```bash
keras_model.h5
```

---

## Deployment

### Deploying on Streamlit Cloud

1. Upload your project files to a GitHub repository.
2. Open Streamlit Cloud.
3. Connect your GitHub account.
4. Select the repository and choose:

```bash
app.py
```

5. Click **Deploy**.

Your application will be hosted online and accessible through a public URL.

---

## Future Improvements

* Add multiple stock comparison
* Improve prediction accuracy
* Add technical indicators
* Deploy with custom UI styling
* Add live prediction updates

---

## Author

Developed using Python, TensorFlow, Keras, and Streamlit.
