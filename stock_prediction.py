import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import datetime

# Takes any ticker
t = ""
while True:
    t = input("Enter a ticker:\n") # take input
    ticker = yf.Ticker(t)

    try:
        info = ticker.info
    except:
        print(f"Cannot get info of {t}, it probably does not exist")
        continue

    break

ticker_info = ticker.info

# Download historical data as dataframe
end_date = datetime.datetime.now().date() - datetime.timedelta(days=365)
start_date = end_date - datetime.timedelta(days=365*11)
df = yf.download(t, start=start_date, end=end_date)

test_data = yf.download(t,start=datetime.datetime.now().date()-datetime.timedelta(days=365),end=datetime.datetime.now())

# Calculate the moving average
df['Moving_Avg'] = df['Close'].rolling(window=5).mean()

# Drop the NaN values
df = df.dropna()

# Plotting the closing prices and moving averages
plt.figure(figsize=(14, 7))
plt.plot(df['Close'])
plt.plot(df['Moving_Avg'])
name = ticker_info['longName']
plt.title(f'Closing Prices vs Moving Average of {name}')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend(['Close', 'Moving_Avg'], loc='upper left')
plt.show()

# Data Preprocessing for Full Model Training
X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Close']

# Model Training on entire data
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

date_to_predict = "9999-99-99"  # Default value
make_more = True
# Date Validation
today = pd.Timestamp.today().normalize()  # Get today's date and normalize time to midnight
one_year_ago = today - pd.DateOffset(years=1)  # Date one year ago

while make_more:
    while True:  # Keep running the loop until a valid date is entered
        date_input = input("Give day from the past year to perform prediction (YYYY-MM-DD):\n")
        date_to_predict = pd.to_datetime(date_input, errors='coerce').normalize()

        if date_to_predict is pd.NaT:
            print("Invalid date format. Please try again.")
            continue

        if date_to_predict < one_year_ago or date_to_predict > today:
            print("Please enter a date within the last year.")
            continue

        # Create feature vector for the given date
        try:
            features = test_data.loc[date_to_predict][['Open', 'High', 'Low', 'Volume']]
        except KeyError:
            print(f"Data for {date_to_predict} not found in the dataset. Please try again.")
            continue

        features = np.array(features).reshape(1, -1)  # Reshape for single prediction

        # Predict the close price for that day
        predicted_value = model.predict(features)

        # Print the results
        print(f"Predicted close price on {date_to_predict}: {predicted_value[0]}")

        break  # Exit the loop since a valid date was found and processed

    predict_another = input("Make another prediction? (Y/N):\n")
    if predict_another == "N":
        break