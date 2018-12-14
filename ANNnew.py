import pandas as pd
import csv
import pdb
import numpy as np
from numpy import split
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dropout, LSTM, Dense, Activation
import time
from sklearn.metrics import mean_squared_error
from numpy.random import seed
from tensorflow import set_random_seed
from scipy.signal import butter, lfilter
from numpy import array

# split a univariate dataset into train/test sets
def split_dataset(data, n_out):
	# split into standard weeks
	n_segs = np.floor((len(data))/n_out)
	n_train = np.floor(0.8 * n_segs)
	n_train = int(np.floor(n_train/n_out)*n_out) # make sure it's a factor of n_out
	
	print("n_segs: ", n_segs)
	print("n_train: ", n_train)
	
	
	train, test = data[0:-n_train], data[-n_train:]
	
	print("train.shape: ", train.shape)
	# restructure into windows of weekly data
	train = array(split(train, len(train)/n_out))
	test = array(split(test, len(test)/n_out))
	return train, test

def baseline_model(lookback,pred):
	# create model
	nnode = 400
	model = Sequential()
	model.add(Dense(nnode, input_dim=lookback, kernel_initializer='normal', activation='relu'))
	model.add(Dense(nnode, kernel_initializer='normal', activation='relu'))
	model.add(Dense(nnode, kernel_initializer='normal', activation='relu'))
	model.add(Dense(int(nnode/2), kernel_initializer='normal', activation='relu'))
	model.add(Dense(int(nnode/4), kernel_initializer='normal', activation='relu'))
	model.add(Dense(pred, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
	return model


print('ANN')

channel = 1
num_sims = 3
rmse_ann = np.zeros((10,num_sims))
for k in np.arange(num_sims):

	#set_random_seed(27365+k)
	print("simulation {} of {}".format(k+1,num_sims))
	
	dataset = read_csv('../LFP_Prediction_WITHDATA/data/sample_LFP_3000to3120.csv')
	# length of dataset must be divisible by n_out
	dataset = dataset.values[0:119000,channel-1:channel]*0.195  # convert to microvolts

	#scaler = MinMaxScaler(feature_range=(-2,2))
	#scaled = scaler.fit_transform(short_seg)


	n_channels = dataset.shape[1]

	n_input = 100 #num_lookback
	n_out = 10 #num_predict


	# split into train and test
	train, test = split_dataset(dataset,n_out)
	
	alltt = np.concatenate((train[:,:,0],test[:,:,0]),0)

	train_ANN = np.zeros((np.size(train,0),n_input+n_out))
	test_ANN = np.zeros((np.size(test,0),n_input+n_out))
	
	for i in np.arange(np.size(train,0)):
		train_ANN[i,:] = alltt[i:i+n_out+1,:].ravel()

	for i in np.arange(np.size(test,0)):
		test_ANN[i,:] = alltt[np.size(train_ANN,0)-n_out+i:np.size(train_ANN,0)+i+1,:].ravel()
	

	train,test = train_ANN[:,:,np.newaxis], test_ANN[:,:,np.newaxis]
	
	
	# make variables for scaled data
	train_scl = np.zeros((train.shape[0],train.shape[1]))
	test_scl = np.zeros((test.shape[0],test.shape[1]))

	train_us = train
	test_us = test

	sclr_train = []
	sclr_test = []

	for i in np.arange(train.shape[0]):
		scl1 = MinMaxScaler(feature_range=(-1,1))
		scaled1 = scl1.fit_transform(train[i,:]).flatten()
		train_scl[i,:] = scaled1
		
		# only append scaler for first variable
		sclr_train.append(scl1)

	for i in np.arange(test.shape[0]):
		scl1 = MinMaxScaler(feature_range=(-1,1))
		scaled1 = scl1.fit_transform(test[i,:]).flatten()
		test_scl[i,:] = scaled1
		
		# only append scaler for first variable
		sclr_test.append(scl1)	

	# evaluate model and get scores

	model = baseline_model(n_input,n_out)

	history = model.fit(train_scl[:,0:n_input],train_scl[:,-n_out:], epochs=50,verbose=1)

	preds = model.predict(test_scl[:,0:n_input])
	
	# unscale the preds
	preds = preds.reshape(preds.shape[0],preds.shape[1])
	preds_us = np.zeros((preds.shape[0],preds.shape[1]))
	for i in np.arange(preds.shape[0]):
		seg = preds[i,:]
		seg = seg.reshape(-1,1)
		inv_scale = sclr_test[i].inverse_transform(seg)
		preds_us[i,:] = inv_scale[:,0]


	# PERSISTENCE FORECAST
	y_pers_test = np.zeros((test_us.shape[0],n_out))

	# first test is from the last sample of train
	y_pers_test[0,:] = np.transpose(np.tile(train_us[-1,-1], n_out))
	for i in np.arange(1,test.shape[0]):
		y_pers_test[i,:] = np.transpose(np.tile(test_us[i-1,-1], (n_out,1)))

	rmse_ann[:,k] = np.sqrt(np.mean((preds_us - test_us[:,-n_out:,0])**2,axis=0))

	rmse_pers = np.sqrt(np.mean((y_pers_test[:,:] - test_us[:,-n_out:,0])**2,axis=0))

	print("ANN RMSE: ", rmse_ann)

	print("PERS RMSE: ",rmse_pers)
	
np.savetxt('./modeloutputdata/ANN/model/ANN_Chan_{}_RMSE.csv'.format(channel),rmse_ann,delimiter=',')
np.savetxt('./modeloutputdata/ANN/pers/ANN_Chan_{}_PERS.csv'.format(channel),rmse_pers,delimiter=',')

np.savetxt('./modeloutputdata/ANN/model/ANN_Chan_{}_preds.csv'.format(channel),preds_us,delimiter=',')
np.savetxt('./modeloutputdata/ANN/model/ANN_Chan_{}_test.csv'.format(channel),test_us[:,-n_out:,0],delimiter=',')
np.savetxt('./modeloutputdata/ANN/pers/ANN_Chan_{}_preds.csv'.format(channel),y_pers_test[:,:],delimiter=',')
