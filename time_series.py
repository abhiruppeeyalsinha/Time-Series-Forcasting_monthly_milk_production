import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential,load_model
from keras.layers import LSTM
from keras.layers import Dense
from sklearn.metrics import mean_squared_error
from math import sqrt

csv_file_path = r"E:\Machine learning\time series dataset\monthly_milk_production.csv"

time_series = pd.read_csv(csv_file_path, index_col="Date", parse_dates=True)
# print(time_series.head())
print(len(time_series))
# plot_dataSet = time_series.plot(figsize=(12,6))

result = seasonal_decompose(time_series["Production"])
# result.plot()
# plt.show()

train = time_series.iloc[:156]
test = time_series.iloc[156:]

scaler = MinMaxScaler()
train_scaler = scaler.fit_transform(train)
test_scaler = scaler.fit_transform(test)

# print(test_scaler)

n_inputs = 12
n_features = 1

generator = TimeseriesGenerator(
    train_scaler, train_scaler, length=n_inputs, batch_size=1)
x, y = generator[0]
# print(f"generator:- {generator[0]}")

# print(f"Given the Array: \n{x.flatten()}")
# print(f"Given the Array: \n{y}")
# print(x.shape)


model = Sequential()
model.add(LSTM(100,activation='relu',input_shape=(n_inputs,n_features)))
model.add(Dense(1))
model.compile(optimizer="adam",loss="mse")
# print(model.summary())


# model.fit(generator,epochs=100)
# result = model.save(r"E:\Machine learning\time series forcasting model\time_series_100_.h5")

# loss_per_epochs = model.history.history['loss']
# plt.plot(range(len(loss_per_epochs)),loss_per_epochs)
# plt.show()

train_model = load_model(r"E:\Machine learning\time series forcasting model\time_series_100_.h5")
# print(loadmodel.summary())

last_train_batch = train_scaler[-12:]
# print(last_train_batch)

last_train_batch = last_train_batch.reshape((1,n_inputs,n_features))
# print(last_train_batch.shape)


result=train_model.predict(last_train_batch)
# print(type(result),result)

test_data = test_scaler[0]
# print(test_data)

test_prediction = []


first_evaluate_batch = train_scaler[-n_inputs:]
# print(first_evaluate_batch)

current_batch = first_evaluate_batch.reshape((1,n_inputs,n_features))
# print(current_batch.shape)

for i in range(len(test)):
    current_prediction = train_model.predict(current_batch)[0]
    test_prediction.append(current_prediction)

    current_batch = np.append(current_batch[:,1:,:],[[current_prediction]],axis=1)


# print(f"test_prediction{test_prediction}")


actual_values = scaler.inverse_transform(test_prediction)
test['Predictions'] = actual_values

test.plot(figsize=(12,6))
# plt.show()

rms = sqrt(mean_squared_error(test["Production"],test["Predictions"]))
print(rms)
