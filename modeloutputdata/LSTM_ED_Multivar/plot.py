import pandas as pd
import pdb
import matplotlib.pyplot as plt
import numpy as np

RMSE = np.zeros((10,1,64))

for i in np.arange(1):
	df = pd.read_csv('./model/LSTM_ED_Multivar_Chan{}_RMSE.csv'.format(i+1),header=None)
	RMSE[:,:,i] = df.values[:,:]

	plt.plot(np.arange(1,11),np.mean(RMSE[:,:,i],1),label='chan {}'.format(i+1))
	plt.fill_between(np.arange(1,11),np.mean(RMSE[:,:,i],1)+np.std(RMSE[:,:,i],1),np.mean(RMSE[:,:,i],1)-np.std(RMSE[:,:,i],1),alpha=0.2)

plt.xlabel('prediction (ms)')
plt.ylabel('RMSE (uV)')
plt.legend()
plt.show()
