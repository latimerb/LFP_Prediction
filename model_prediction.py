import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dropout, LSTM, Dense, Activation
import time
import csv
from sklearn.metrics import mean_squared_error
from scipy import signal
import pdb

np.random.seed(1234)

df = pd.read_csv('./data/model_LFP_and_FR.csv',header=None)
dataset = df.iloc[:,2:].values


# fs = 1000.

# f, Pxx_den = signal.welch(dataset,fs,nperseg=2001)

# plt.subplot(2,1,1)
# plt.plot(dataset)
# plt.xlabel('time(ms)')
# plt.subplot(2,1,2)
# plt.semilogy(Pxx_den)
# plt.xlabel('frequency (hz)')
#plt.xlim(0,100)

from utils import series_to_supervised, fit_lstm, forecast_lstm, make_forecasts

# Series to supervised
num_lookback = 200
num_predict = 10

# n_in is the number of samples for "input" and n_out is the number for "output", or prediction
supervised_dataset = series_to_supervised(dataset,n_in=num_lookback,n_out=num_predict)
sup_ds = supervised_dataset.values

# Scale
row_mean = sup_ds.mean(axis=1,keepdims=True)
row_max = sup_ds.max(axis=1,keepdims=True)
row_min = sup_ds.min(axis=1,keepdims=True)

scl_sup_ds = (sup_ds-row_mean)/(row_max-row_min)

# Train test split
N_train = int(0.8*np.size(scl_sup_ds,0))

scaled_train = scl_sup_ds[0:N_train,:]
scaled_test = scl_sup_ds[N_train:,:]

# train    
model_lstm = fit_lstm(scaled_train, num_predict, 10, 20, n_neurons=1000)
model_lstm.summary()

# Make forecasts on test data
scaled_forecasts = make_forecasts(model_lstm, 1, scaled_test, num_predict)

# Make persistence forecasts
y_pers_test = np.transpose(np.tile(scaled_test[:,-num_predict-1], (num_predict,1)))

# Calculate root mean squared error on forecasts
rmse_lstm_test = np.sqrt(np.mean((scaled_forecasts-scaled_test[:,-num_predict:])**2,axis=0))
print("rmse for (LSTM) testing: ", rmse_lstm_test)

rmse_pers_test = np.sqrt(np.mean((y_pers_test-scaled_test[:,-num_predict:])**2,axis=0))
print("rmse for (persistence) testing: ", rmse_pers_test)

# Plot the persistence and LSTM forecasts
plt.figure()
plt.bar(np.arange(0,num_predict)-0.2,rmse_pers_test,0.3,label='rmse of persistence forecast')
plt.bar(np.arange(0,num_predict)+0.2,rmse_lstm_test,0.3,label='rmse of lstm forecast')
plt.legend()

#look at 6 samples to see how we did
chunk_size = num_lookback + num_predict

plt.figure(figsize=(20,10))
for i in np.arange(1,7):
    sample = np.random.randint(0,scaled_forecasts.shape[0])
    plt.subplot(2,3,i)
    plt.plot(np.linspace(1,chunk_size,num=chunk_size),scl_sup_ds[sample+N_train,:],color='blue',label='actual')
    plt.plot(np.linspace(num_lookback+1,chunk_size,num=num_predict),scaled_forecasts[sample,:],color='red',label='predicted')
    
plt.show()