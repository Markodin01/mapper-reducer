import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.width = None
pd.set_option('display.max_rows', 100)
pd.set_option('display.min_rows', 20)
from psutil import Process
from IPython.display import display_html 

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

tickers = ["AAPL.US", "AMZN.US", "MSFT.US", "NVDA.US", "TSLA.US", "GOOGL.US", ]
names = ["Apple", "Amazon", "Microsoft", "Nvidia", "Tesla", "Google"]
timeframe = "D1"

ticker = "AAPL.US"
df_AAPL = pd.read_csv(f"stock_data/{timeframe}/{ticker}_{timeframe}.csv")

selected_columns = ["datetime", "open", "high", "low", "close", "volume", "roc_10", "roc_20", "roc_50", "sma_10", "sma_20", "sma_50"]

df_all = {}
for ticker in tickers:
    df_all[ticker] = pd.read_csv(f"stock_data/{timeframe}/{ticker}_{timeframe}.csv")
    df_all[ticker] = df_all[ticker][selected_columns]  # filter only needed columns from 1298

def show_dfs_in_side_by_side(dfs, captions):
    _disp_dfs = []
    for i in range(len(dfs)):
        _df = dfs[i]
        _caption = captions[i]
        _df_styler = _df.style.set_table_attributes("style='display:inline'").set_caption(_caption)
        _disp_dfs.append(_df_styler._repr_html_())
    display_html(_disp_dfs, raw=True)

dict_values = list(df_all.values())
dict_names = list(df_all.keys())

dfs=dict_values[0]

n = 30
dict_values_n = [dict_values[i][:n] for i in range(len(dict_values))]  # limit to n rows

show_dfs_in_side_by_side(dfs=dict_values_n, captions=dict_names)

def reduce_mem_usage(df, verbose=0):
    """function to reduce memory usage for dataframe"""
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        try:
            col_type = df[col].dtype
            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                
                if str(col_type)[:3] == "int":
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float32)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float32)
        except:
            pass
    if verbose:
        print(f"Memory usage of dataframe is {start_mem:.2f} MB")
        end_mem = df.memory_usage().sum() / 1024**2
        print(f"Memory usage after optimization is: {end_mem:.2f} MB")
        decrease = 100 * (start_mem - end_mem) / start_mem
        print(f"Decreased by {decrease:.2f}%")
    return df

def checkMemory():
    """This function defines the memory usage across the kernel. Source - https://stackoverflow.com/questions/61366458/how-to-find-memory-usage-of-kaggle-notebook"""
    pid = os.getpid()
    py = Process(pid)
    memory_use = py.memory_info()[0] / 2. ** 30
    return f"RAM memory GB usage = {memory_use :.4}"
checkMemory()

df_AAPL = reduce_mem_usage(df_AAPL, verbose=1)

num_prices = 365  # how many days to show

plt.figure(figsize=(20, 15))
sns.set_style('darkgrid')
plt.subplots_adjust(top=1.25, bottom=1.2)

for i, ticker in enumerate(df_all, 1):
    plt.subplot(3, 2, i)
    df_all[ticker]['close'][len(df_all[ticker]['close'])-num_prices:].plot()
    plt.ylabel('close')
    plt.xlabel('days')
    plt.title(f"closing price of {names[i - 1]}")

plt.tight_layout()

df_AAPL_filtered = df_AAPL[selected_columns]

data_split_percent = 0.9  # 90% to train and 10% to validate

# create a new dataframe with only the close+roc_10 columns
data = df_AAPL_filtered[['close', 'roc_10']]
# convert the dataframe to a numpy array
dataset = data.values
# get the number of rows to train the model on
training_data_len = int(np.ceil( len(dataset) * data_split_percent ))

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:int(training_data_len), :]

test_data = scaled_data[int(training_data_len):, :]

size_of_train_set = 150

# split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(size_of_train_set, len(train_data)):
    x_train.append(train_data[i-size_of_train_set:i, :])  # by use ":" - we are forming N-dimensional input
    y_train.append(train_data[i, 0])  # output - is just next close scalered by MinMaxScaler
    if i<= size_of_train_set or i==size_of_train_set*10 or i==(size_of_train_set*10+1):
        print(x_train[-1])
        print(y_train[-1])
        print()

x_train, y_train = np.array(x_train), np.array(y_train)

# split the data into x_test and y_test data sets
x_test = []
y_test = []

for i in range(size_of_train_set, len(test_data)):
    x_test.append(test_data[i-size_of_train_set:i, :])  # by use ":" - we are forming N-dimensional input
    y_test.append(test_data[i, 0])  # output - is just next close scalered by MinMaxScaler
    if i<= size_of_train_set or i==size_of_train_set*10 or i==(size_of_train_set*10+1):
        print(x_test[-1])
        print(y_test[-1])
        print()

x_test, y_test = np.array(x_test), np.array(y_test)

data = df_AAPL_filtered[['close', ]][len(df_AAPL_filtered)-3000:]  # let's get only last 3000 days

dataset = data.values

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

def split_data_into_buckets(data, size_of_set):
    xx, yy = [], []
    for i in range(size_of_set, len(data)):
        xx.append(data[i-size_of_set:i, :])  # by use ":" - we are forming N-dimensional input
        yy.append(data[i, 0])  # output - is just next close scalered by MinMaxScaler        
    return xx, yy
        
def split_data_for_nn(data_to_split, size_of_set, train_percent, val_percent, test_percent):
    all_len=len(data_to_split)-1  # -1 to get future price of closing
    train_data_len = int(np.ceil(len(dataset) * train_percent))
    val_data_len = int(np.ceil(len(dataset) * val_percent))
    test_data_len = int(np.ceil(len(dataset) * test_percent))
    if train_data_len+val_data_len+test_data_len > all_len: test_data_len = all_len - train_data_len - val_data_len
    print(train_data_len, val_data_len, test_data_len, "+=", train_data_len+val_data_len+test_data_len, "all_len:", all_len)
    
    train_data = scaled_data[0:train_data_len, :]
    val_data = scaled_data[train_data_len:train_data_len+val_data_len, :]
    test_data = scaled_data[train_data_len+val_data_len:train_data_len+val_data_len+test_data_len, :]
    print(train_data.shape, val_data.shape, test_data.shape)
    
    x_train, y_train = split_data_into_buckets(train_data, size_of_set)
    x_val, y_val = split_data_into_buckets(val_data, size_of_set)
    x_test, y_test = split_data_into_buckets(test_data, size_of_set)
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_val, y_val = np.array(x_val), np.array(y_val)
    x_test, y_test = np.array(x_test), np.array(y_test)
    
    return x_train, y_train, x_val, y_val, x_test, y_test, train_data_len, val_data_len, test_data_len

size_of_set = 60
x_train, y_train, x_val, y_val, x_test, y_test, train_data_len, val_data_len, test_data_len = split_data_for_nn(data_to_split=scaled_data, size_of_set=size_of_set, train_percent=0.7, val_percent=0.2, test_percent=0.1)
print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
print(x_test.shape, y_test.shape)




import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.regularizers import l2
import time

# Model Configuration Parameters
batch_size = 32  # Optimal batch size, subject to experimentation
epochs = 100  # Number of epochs to train, adjust based on early stopping
LR = 0.001  # Optimized learning rate for Adam
LAMBD = 0.01  # L2 regularization factor
DP = 0.2  # Dropout rate for dropout layers

# Assuming x_train, y_train, x_val, y_val, x_test, y_test, and scaler are defined and preprocessed

# Initialize the Sequential model
model = Sequential()

# First LSTM layer with L2 regularization, input shape from training data
model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2]), kernel_regularizer=l2(LAMBD)))
model.add(Dropout(DP))
model.add(BatchNormalization())

# Second LSTM layer
model.add(LSTM(64, return_sequences=False, kernel_regularizer=l2(LAMBD)))
model.add(Dropout(DP))
model.add(BatchNormalization())

# Dense layer with ReLU activation
model.add(Dense(25, activation='relu'))
model.add(Dropout(DP))

# Output layer for regression
model.add(Dense(1, activation='linear'))

# Compile the model with Adam optimizer and MSE loss function
model.compile(optimizer=Adam(learning_rate=LR), loss='mean_squared_error')

# Define callbacks for dynamic learning rate adjustment and early stopping
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


start_train = time.time()
# Train the model with training data, validation split, and callbacks
history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val), callbacks=[reduce_lr, early_stopping])
end_train = time.time()

total_training_time = end_train - start_train
average_time_per_epoch = total_training_time / epochs
print(f'Total training time: {total_training_time} seconds')
print(f'Average time per epoch: {average_time_per_epoch} seconds')

# Function to inverse transform the scaled data back to original scale
def inv_transform(scaler, data, column_name, feature_names):
    dummy_df = pd.DataFrame(np.zeros((len(data), len(feature_names))), columns=feature_names)
    dummy_df[column_name] = data.flatten()
    inverted_data = scaler.inverse_transform(dummy_df)[:, feature_names.index(column_name)]
    return inverted_data

# Predict and inverse transform the predictions for validation and test sets
predictions_val = model.predict(x_val)
predictions_val = inv_transform(scaler, predictions_val, "close", ["close"])

predictions_test = model.predict(x_test)
predictions_test = inv_transform(scaler, predictions_test, "close", ["close"])

# Calculate RMSE for validation and test predictions
rmse_val = np.sqrt(np.mean(np.square(predictions_val - y_val)))
rmse_test = np.sqrt(np.mean(np.square(predictions_test - y_test)))
print(f"Validation RMSE: {rmse_val}, Test RMSE: {rmse_test}")

# Plotting the training and validation losses
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Assuming your dataset is already loaded, preprocessed, and split into features (X) and target (y)

# Reshape or adjust your data as necessary for XGBoost
# Note: XGBoost does not require the input to be 3D like LSTM does, so you might flatten or reshape data if coming from LSTM preparation
# Example: x_train = x_train.reshape((x_train.shape[0], -1))
# This step depends on how your data was prepared for LSTM and might not be necessary

# Convert the datasets to DMatrix, which is a highly efficient XGBoost data structure
dtrain = xgb.DMatrix(x_train, label=y_train)
dval = xgb.DMatrix(x_val, label=y_val)
dtest = xgb.DMatrix(x_test)

# Set XGBoost parameters
# These parameters should be tuned according to your specific dataset and task
params = {
    'max_depth': 3,  # depth of the trees
    'eta': 0.1,  # learning rate
    'objective': 'reg:squarederror',  # loss function for regression
    'eval_metric': 'rmse',  # evaluation metric
}
num_rounds = 100  # Number of training rounds

# Train the model
evals = [(dtrain, 'train'), (dval, 'eval')]
bst = xgb.train(params, dtrain, num_rounds, evals, early_stopping_rounds=10)

# Predictions
predictions_val = bst.predict(dval)
predictions_test = bst.predict(dtest)

# Calculate RMSE for validation and test predictions
rmse_val = np.sqrt(mean_squared_error(y_val, predictions_val))
rmse_test = np.sqrt(mean_squared_error(y_test, predictions_test))
print(f"Validation RMSE: {rmse_val}, Test RMSE: {rmse_test}")

# If you want to save the model
# bst.save_model('xgboost_model.model')

# To load the model
# bst = xgb.Booster()
# bst.load_model('xgboost_model.model')





