import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create timeseries data
def create_ts_data(data, window_size=5):
    i = 1
    while i < window_size:
        data["co2_{}".format(i)] = data["co2"].shift(-i)
        i += 1
    data["target"] = data["co2"].shift(-i)
    data = data.dropna(axis=0)
    return data


data = pd.read_csv("co2.csv")
data["time"] = pd.to_datetime(data["time"])
data["co2"] = data["co2"].interpolate()

data = create_ts_data(data)

# fig, ax = plt.subplots()
# ax.plot(data["time"], data["co2"])
# ax.set_xlabel("Year")
# ax.set_ylabel("CO2")
# plt.show()

x = data.drop(["time", "target"], axis=1)
y = data["target"]

train_ratio = 0.8
num_samples = len(x)

x_train = x[:int(num_samples * train_ratio)]
y_train = y[:int(num_samples * train_ratio)]
x_test = x[int(num_samples * train_ratio):]
y_test = y[int(num_samples * train_ratio):]

# Train model
model = RandomForestRegressor()
model.fit(x_train, y_train)

# Test model
y_predict = model.predict(x_test)
print("MAE: {}".format(mean_absolute_error(y_test, y_predict)))
print("MSE: {}".format(mean_squared_error(y_test, y_predict)))
print("R2: {}".format(r2_score(y_test, y_predict)))

# Visualization
fig, ax = plt.subplots()
ax.plot(data["time"][:int(num_samples * train_ratio)], data["co2"][:int(num_samples * train_ratio)], label="train")
ax.plot(data["time"][int(num_samples * train_ratio):], data["co2"][int(num_samples * train_ratio):], label="test")
ax.plot(data["time"][int(num_samples * train_ratio):], y_predict, label="prediction")
ax.set_xlabel("Year")
ax.set_ylabel("CO2")
ax.legend()
ax.grid()
plt.show()

# # Test
# current_data = [380.5, 390, 390.2, 390.4, 393]
#
# for i in range(10):
#     print("Input is {}".format(current_data))
#     prediction = model.predict([current_data])[0]
#     print("CO2 in week {} is {}".format(i+1, prediction))
#     current_data = current_data[1:] + [prediction]
#     print("---------------------------")
