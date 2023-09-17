import pandas as pd
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
model_path = './model'

t = ""
while True:
    t = input("Enter a ticker:\n")  # take input
    ticker = yf.Ticker(t)

    try:
        info = ticker.info
    except:
        print(f"Cannot get info of {t}, it probably does not exist")
        continue

    break

df = yf.download(t)
df = df[["Close"]]
# print(df.head())

# print(df.index)

# Plotting the data
# plt.plot(df.index, df['Close'])
# plt.xlabel('Date')
# plt.ylabel('Close Price')
# plt.title(f'{t} Closing Prices')
# plt.show()

def df_to_windowed_df(dataframe, first_date, last_date, n=3):

  target_date = first_date
  
  dates = []
  X, Y = [], []

  last_time = False
  while True:
    df_subset = dataframe.loc[:target_date].tail(n+1)
    
    if len(df_subset) != n+1:
      print(f'Error: Window of size {n} is too large for date {target_date}')
      return

    values = df_subset['Close'].to_numpy()
    x, y = values[:-1], values[-1]

    dates.append(target_date)
    X.append(x)
    Y.append(y)

    next_week = dataframe.loc[target_date:target_date+datetime.timedelta(days=7)]
    next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
    next_date_str = next_datetime_str.split('T')[0]
    year_month_day = next_date_str.split('-')
    year, month, day = year_month_day
    next_date = datetime.datetime(day=int(day), month=int(month), year=int(year))
    
    if last_time:
      break
    
    target_date = next_date

    if target_date == last_date:
      last_time = True
    
  ret_df = pd.DataFrame({})
  ret_df['Target Date'] = dates
  
  X = np.array(X)
  for i in range(0, n):
    X[:, i]
    ret_df[f'Target-{n-i}'] = X[:, i]
  
  ret_df['Target'] = Y

  return ret_df

windowed_df = df_to_windowed_df(df,df.index.max()-datetime.timedelta(days=365*2),df.index.max())

print(windowed_df)

def windowed_df_to_date_X_y(windowed_dataframe):
   df_as_np = windowed_dataframe.to_numpy()

   dates = df_as_np[:,0]
   middle_matrix = df_as_np[:,1:-1]

   X = middle_matrix.reshape((len(dates),middle_matrix.shape[1],1))

   Y = df_as_np[:,-1]

   return dates, X.astype(np.float32), Y.astype(np.float32)

dates, X, y = windowed_df_to_date_X_y(windowed_df)

print(dates.shape, X.shape, y.shape)

q_80 = int(len(dates)*0.8)
q_90 = int(len(dates)*0.9)

dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]

dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]
dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]

plt.plot(dates_train,y_train)
plt.plot(dates_val,y_val)
plt.plot(dates_test,y_test)

plt.legend(['Train','Validation','Test'])
plt.show()

if os.path.exists(os.path.join(model_path, 'saved_model.pb')):
    print("Loading existing model.")
    model = load_model(model_path)
else:
    print("Training new model.")
    model = Sequential([layers.Input((3,1)),
                        layers.LSTM(64),
                        layers.Dense(32, activation='relu'),
                        layers.Dense(32, activation='relu'),
                        layers.Dense(1)])
    
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['mean_absolute_error'])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)
    
    # Save the trained model
    model.save(model_path)


train_predictions = model.predict(X_train).flatten()
plt.plot(dates_train, train_predictions)
plt.plot(dates_train, y_train)
plt.legend(['Training Predeictions','Training Observations'])
plt.show()

val_predictions = model.predict(X_val).flatten()
plt.plot(dates_val, val_predictions)
plt.plot(dates_val, y_val)
plt.legend(['Validation Predeictions','Validation Observations'])
plt.show()

test_predictions = model.predict(X_test).flatten()
plt.plot(dates_test, test_predictions)
plt.plot(dates_test, y_test)
plt.legend(['Training Predeictions','Training Observations'])
plt.show()


plt.plot(dates_train,y_train)
plt.plot(dates_train,y_train)
plt.plot(dates_val,val_predictions)
plt.plot(dates_val,y_val)
plt.plot(dates_test, test_predictions)
plt.plot(dates_test, y_test)
plt.legend(['Training Predictions',
            'Training Observations',
            'Validation Predictions', 
            'Validation Observations',
            'Testing Predictions', 
            'Testing Observations'])
plt.show()

