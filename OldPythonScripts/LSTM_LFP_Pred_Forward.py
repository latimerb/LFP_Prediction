# Written by Ben Latimer 2017 #
# Modified by Ziao Chen #
#####
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import pdb
import matplotlib.pyplot as plt

#datafile should contain three columns with raw, causal, non-causal filtered signal.

# fix random seed for reproducibility
seed = 111
np.random.seed(seed)

# % of training courses
ptrain = 0.9
nsamp = 50000	# 50000
xchan = 1		# 0 (raw), 1 (causal filter), 2(non-causal)
ychan = 1


if xchan==0:
	print("Predicting on raw signal")
if xchan==1:
	print("Predicting on causal filtered signal")
if xchan==2:
	print("Predicting on non-causal filtered signal")


# read data
Lseg = pd.read_csv('segment_length.csv',header=None).values
print("number of segments:",Lseg.shape)
data = pd.read_csv('train_forward_pred.csv',header=None).values
print("raw data shape:", data.shape)



length = data.shape[0] #length of signal
nseg = Lseg.shape[1]
ntrain = int(round(nsamp*ptrain))
ntest = nsamp-ntrain

# select training set
prelen = 50
forward = 10


chunk = prelen+forward-1
id = np.arange(length-nseg*chunk)
iseg = np.cumsum(Lseg-chunk).astype(int)
for i in np.arange(nseg-1):
	id[iseg[i]:iseg[i+1]] += chunk*(i+1)
idx = np.random.choice(id,nsamp,replace=False)

x = np.empty((nsamp,prelen))
y = np.empty((nsamp,forward))
for i in np.arange(nsamp):
	x[i,:] = data[idx[i]:idx[i]+prelen,xchan]
	y[i,:] = data[idx[i]+prelen:idx[i]+prelen+forward,ychan]
x_train = x[:ntrain,:]
y_train = y[:ntrain,:]
x_test = x[-ntest:,:]
y_test = y[-ntest:,:]


# Remove the features from the labels
print("x_train.shape:", x_train.shape)
print("y_train.shape:", y_train.shape)
print("x_test.shape:",x_test.shape)
print("y_test.shape:",y_test.shape)


# #------------------------------------------------------------------------------#
# #Build neural network
nnode = 40;
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(nnode, input_dim=prelen, kernel_initializer='normal', activation='relu'))
	model.add(Dense(int(nnode/2), kernel_initializer='normal', activation='relu'))
	model.add(Dense(int(nnode/4), kernel_initializer='normal', activation='relu'))
	model.add(Dense(forward, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model


# Fit the network
model = baseline_model()
model.fit(x_train,y_train, epochs=5)

# Make predictions
y_pred_train = model.predict(x_train)
rmse = np.sqrt(np.mean((y_pred_train-y_train)**2,axis=0))
print ("Training RMSE of %d forward predictions:" % (forward))
np.set_printoptions(precision=3)
print(rmse)

y_pred_test = model.predict(x_test)
rmse = np.sqrt(np.mean((y_pred_test-y_test)**2,axis=0))
print ("Testing RMSE of %d forward predictions:" % (forward))
np.set_printoptions(precision=4)
print(rmse)

# Look at some examples
# prediction is nexamp rows and prelen + 2*forward columns
# x_test + y_test + y_pred_test
nexamp = 10
prediction = np.concatenate((x_test[:nexamp],y_test[:nexamp],y_pred_test[:nexamp]),axis=1)
time = np.arange(1,61)

plt.figure()
for i in range(0,5):
	plt.subplot(5,1,i+1)
	plt.scatter(time[0:51],prediction[i,0:51],color='blue')
	plt.scatter(time[50:60],prediction[i,50:60],color='orange')
	plt.scatter(time[50:60],prediction[i,60:70],color='red')
plt.show()



# Write to file
# df = pd.DataFrame(prediction)
# df.to_csv("forward_pred_sub4.csv",header=None,index=False)

