from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import pdb
import numpy as np
import matplotlib.pyplot as plt

N = 6
#series = read_csv('./modeloutputdata/ex_LFP.csv',header=None)
#X = series.values[0:11000]
series = read_csv('ANNexamples.csv',header=None)
X = series.values


rmse_arima_test = np.zeros((N,10))
rmse_pers_test = np.zeros((N,10))
YHAT = np.zeros((N,10))
ACTUAL = np.zeros((N,10))
IN_ACTUAL = np.zeros((N,60))
PERS = np.zeros((N,10))

for i in range(N):
	
	#train, test = X[0+100*i:100*i+50], X[100*i+50:100*i+60]
	train,test = X[i,0:20], X[i,20:30]
	# plt.plot(np.arange(1,51),train)
	# plt.plot(np.arange(50,60),test)
	# plt.show()
	
	history = [x for x in train]
	predictions = list()
	print('beginning test %i' % (i))
	for t in range(len(test)):
		model = ARIMA(history, order=(3,1,0))
		model_fit = model.fit(disp=0)
		output = model_fit.forecast()
		yhat = output[0]
		predictions.append(yhat)
		obs = test[t]
		history.append(yhat)
		print('predicted=%f, expected=%f' % (yhat, obs))
	
	# save predictions and test for later
	YHAT[i,:] = predictions
	ACTUAL[i,:] = test
pdb.set_trace()
	#IN_ACTUAL[i,:] = X[0+100*i:100*i+60,0]
	#rmse_arima_test[i,:] = np.sqrt(np.mean((predictions-test)**2,axis=1))
	
	#PERS[i,:]=np.transpose(np.tile(IN_ACTUAL[i,-11], (10,1)))
	#rmse_pers_test[i,:] = np.sqrt(np.mean((PERS[i,:]-test)**2,axis=1))

	

fig1 = plt.figure()	
for i in range(6):
	ax=plt.subplot(2,3,i+1)
	plt.plot(np.arange(1,51),IN_ACTUAL[i,:],color='#00FF00')
	plt.plot(np.arange(51,61), YHAT[i,:], color='orange')
	plt.plot(np.arange(51,61),ACTUAL[i,:],color='#FF0000')
	ax.set_xticklabels([])
	ax.set_yticklabels([])
plt.tight_layout()
#fig1.savefig('ARIMAexamples.png')
	
plt.figure()
plt.subplot(2,1,1)
plt.bar(np.arange(1,11)-0.2,np.mean(rmse_pers_test,axis=0)*0.1,0.3,label='persistence')
plt.bar(np.arange(1,11)+0.2,np.mean(rmse_arima_test,axis=0)*0.1,0.3,label='ARIMA')
plt.ylabel('error in uV')
plt.ylim(0,62)
plt.legend()

plt.subplot(2,1,2)
plt.bar(np.arange(1,11),100*np.mean(rmse_arima_test,axis=0)/np.mean(rmse_pers_test,axis=0))
plt.plot(np.arange(0,12),100*np.ones((12,)),'r--')
plt.xlim(0,11)

plt.show()

print(rmse_arima_test)
