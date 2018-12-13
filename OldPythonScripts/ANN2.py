import pandas as pd
import csv
import pdb
import numpy as np
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

set_random_seed(2765)

def scalewindow(short_seg):
	short_seg = short_seg.reshape(-1,1)
	scaler = MinMaxScaler(feature_range=(0,1))
	scaled = scaler.fit_transform(short_seg)
	scaled = scaled.reshape(len(short_seg),)
	return scaled, scaler

def unscale(X, Y, S):
	for i in range(X.shape[0]):
		seg = X[i,:]
		seg = seg.reshape(-1,1)
		us_X = S[i].inverse_transform(seg)
		us_X1d = us_X.reshape(us_X.shape[0],)
			
		Y[i,:] = us_X1d
	return Y
	
sub4_aligned = pd.read_csv('./data/train_aligned_sub4_raw.csv',header=None)
sub4_al_vals = sub4_aligned.values

gamma_segs = sub4_al_vals[np.where(sub4_al_vals[:,-1]==1)[0],:]
nongamma_segs = sub4_al_vals[np.where(sub4_al_vals[:,-1]==0)[0],:]

# supervised_dataset is either gamma or nongamma

supervised_dataset = np.zeros((gamma_segs.shape[0],50))

supervised_dataset = nongamma_segs[:,-51:-1]

# scale
scl_sup_ds_raw = np.zeros((supervised_dataset.shape[0],supervised_dataset.shape[1]))
sclr_raw = []

for i in np.arange(0,gamma_segs.shape[0]):
		
	scl_sup_ds_raw[i,:], scl_r = scalewindow(supervised_dataset[i,:])  
		
	sclr_raw.append(scl_r)


# train/test
num_predict = 10
num_lookback = 5 
chunk_size = num_predict + num_lookback

N_train = int(0.8*supervised_dataset.shape[0])

X_train = scl_sup_ds_raw[0:N_train,-num_lookback-num_predict:-num_predict]
Y_train = scl_sup_ds_raw[0:N_train,-num_predict:]
X_test = scl_sup_ds_raw[N_train:,-num_lookback-num_predict:-num_predict]
Y_test = scl_sup_ds_raw[N_train:,-num_predict:]



nnode = 400
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(nnode, input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))
	model.add(Dense(nnode, kernel_initializer='normal', activation='relu'))
	model.add(Dense(nnode, kernel_initializer='normal', activation='relu'))
	model.add(Dense(int(nnode/2), kernel_initializer='normal', activation='relu'))
	model.add(Dense(int(nnode/4), kernel_initializer='normal', activation='relu'))
	model.add(Dense(Y_train.shape[1], kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
	return model
			
			
model = baseline_model()
t_start = time.time()
history = model.fit(X_train,Y_train, epochs=50,verbose=1)
print("model took ", time.time()-t_start, "seconds to fit")
# plt.figure()
# plt.plot(history.history['acc'])

###################################################################
############ MAKE PREDICTIONS AND UNDO SCALING ######################
y_pred_test = model.predict(X_test)


	
y_pred_test_us = np.zeros((y_pred_test.shape[0],y_pred_test.shape[1]))
y_pred_test_us = unscale(y_pred_test, y_pred_test_us, sclr_raw[N_train:])

Y_test_us = np.zeros((Y_test.shape[0],Y_test.shape[1]))
Y_test_us = unscale(Y_test, Y_test_us, sclr_raw[N_train:])

rmse_ann_test = np.sqrt(np.mean((y_pred_test_us-Y_test_us)**2,axis=0))
print("rmse for testing: ", rmse_ann_test)

# Y_PERS IS ON RAW if 0, FILT if 1
y_pers_test = np.transpose(np.tile(supervised_dataset[N_train:,-num_predict-1], (num_predict,1)))

rmse_pers_test = np.sqrt(np.mean((y_pers_test-Y_test_us)**2,axis=0))
print("rmse for (persistence) testing: ", rmse_pers_test)


plt.figure()
plt.bar(np.arange(0,num_predict)-0.2,rmse_pers_test*0.1,0.3,label='persistence')
plt.bar(np.arange(0,num_predict)+0.2,rmse_ann_test*0.1,0.3,label='MLP')
plt.ylabel('error in uV')
plt.ylim(0,62)
plt.legend()

plt.figure()
	
for i in np.arange(1,7):

	sample = np.random.randint(0,Y_test.shape[0])
	ax = plt.subplot(2,3,i)
	#ax.set_xticklabels([])
	#ax.set_yticklabels([])
	plt.plot(np.linspace(1,chunk_size,num=chunk_size),supervised_dataset[N_train+sample,-chunk_size:],color='#00FF00')
	plt.plot(np.linspace(num_lookback+1,chunk_size,num=num_predict),Y_test_us[sample,:],color='#FF0000')
	plt.plot(np.linspace(num_lookback+1,chunk_size,num=num_predict),y_pred_test_us[sample,:],color='orange')
		
plt.tight_layout()
plt.show()

