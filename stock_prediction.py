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
end_date = datetime.datetime.now().date()
start_date = end_date - datetime.timedelta(days=365*10)
df = yf.download(t, start=start_date, end=end_date)

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

# Data Preprocessing
X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Close']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")

# Visualization
plt.figure(figsize=(14, 7))
plt.plot(y_test.index, y_test.values, label='Actual')
plt.plot(y_test.index, y_pred, label='Predicted')
plt.title(f"{name} Stock Price Prediction")
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

# print(type(y_test.iloc))

date_to_predict = "9999-99-99" # Default value
make_more = True

while make_more:
    while True:  # Keep running the loop until a valid date is entered
        date_to_predict = input("Give date to perform prediction (YYYY-MM-DD):\n")

        if date_to_predict in y_test.index:  # Check if date exists
            index_of_date = y_test.index.get_loc(date_to_predict)  # This should not fail now

            # Retrieve the actual and predicted value for that date
            actual_value = y_test.iloc[index_of_date]
            predicted_value = y_pred[index_of_date]

            # Print the results
            print(f"Actual close price on {date_to_predict}: {actual_value}")
            print(f"Predicted close price on {date_to_predict}: {predicted_value}")

            # Calculate the error
            error = abs(actual_value - predicted_value)
            print(f"Absolute error: {error}")
            
            break  # Exit the loop since a valid date was found and processed
        else:
            print(f"Data for {date_to_predict} not found in the dataset. Please try again.")

    predict_another = input("Make another prediction? (Y/N):\n")
    if predict_another == "N":
        break