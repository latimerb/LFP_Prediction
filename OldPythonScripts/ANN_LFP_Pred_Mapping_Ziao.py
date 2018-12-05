# Written by Ben Latimer 2017 #
# Modified by Ziao Chen #

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

"""
from keras.callbacks import History
from keras.utils import plot_model, np_utils
from keras.optimizers import SGD

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.cross_validation import train_test_split
from keras.utils import to_categorical
import seaborn
"""

# fix random seed for reproducibility
seed = 111
np.random.seed(seed)

# % of training courses
ptrain = 0.9
nsamp = 50000	# 50000
xchan = 1		# 0 (raw), 1 (causal filter), 2(non-causal)
ychan = 2

# read data
Lseg = pd.read_csv('segment_length.csv',header=None).values
print(Lseg.shape)
data = pd.read_csv('train_forward_pred.csv',header=None).values
print(data.shape)
length = data.shape[0] #length of signal
nseg = Lseg.shape[1]
ntrain = int(round(nsamp*ptrain))
ntest = nsamp-ntrain

# select training set
prelen = 50
maplen = 20
forward = 0
chunk = prelen+forward-1
id = np.arange(length-nseg*chunk)
iseg = np.cumsum(Lseg-chunk).astype(int)
for i in xrange(nseg-1):
	id[iseg[i]:iseg[i+1]] += chunk*(i+1)
idx = np.random.choice(id,nsamp,replace=False)

#x = np.empty((nsamp,prelen))
x = np.empty((nsamp,prelen*2))
y = np.empty((nsamp,maplen+forward))
for i in xrange(nsamp):
	x[i,:prelen] = data[idx[i]:idx[i]+prelen,xchan]
	x[i,-prelen:] = data[idx[i]:idx[i]+prelen,1-xchan]
	y[i,:] = data[idx[i]+prelen-maplen:idx[i]+prelen+forward,ychan]
x_train = x[:ntrain,:]
y_train = y[:ntrain,:]
x_test = x[-ntest:,:]
y_test = y[-ntest:,:]

# Remove the features from the labels
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# #------------------------------------------------------------------------------#
# #Build neural network
nnode = 40;
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(nnode, input_dim=prelen*2, kernel_initializer='normal', activation='relu'))
	model.add(Dense(nnode/2, kernel_initializer='normal', activation='relu'))
	model.add(Dense(nnode/4, kernel_initializer='normal', activation='relu'))
	model.add(Dense(maplen+forward, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

"""	
estimator = KerasRegressor(build_fn=baseline_model, epochs=200, batch_size=5, verbose=1)
kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
results = cross_val_score(estimator, x, y, cv=kfold)
print("\nResults: %.2f (%.2f) MSE %.2f RMSE" % (results.mean(), results.std(), np.sqrt(results.mean())))

"""

model = baseline_model()
model.fit(x_train,y_train, epochs=200)


y_pred_train = model.predict(x_train)
rmse = np.sqrt(np.mean((y_pred_train-y_train)**2,axis=0))
print ("Training RMSE of %d mapping and %d forward predictions:" % (maplen,forward))
np.set_printoptions(precision=3)
print(rmse)

y_pred_test = model.predict(x_test)
rmse = np.sqrt(np.mean((y_pred_test-y_test)**2,axis=0))
print ("Testing RMSE of %d mapping and %d forward predictions:" % (maplen,forward))
np.set_printoptions(precision=4)
print(rmse)

nexamp = 10
prediction = np.concatenate((idx[:nexamp].reshape((-1,1))+1,y_test[:nexamp],y_pred_test[:nexamp]),axis=1)
df = pd.DataFrame(prediction)
df.to_csv("mapping_pred_sub4.csv",header=None,index=False)
#"""
