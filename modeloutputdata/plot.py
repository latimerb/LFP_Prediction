import pandas as pd
import pdb
import matplotlib.pyplot as plt
import numpy as np

RMSE_ED_Multivar = np.zeros((10,4,64))
RMSE_CNN_ED_Univar = np.zeros((10,3,64))
RMSE_Univar = np.zeros((10,5,64))
RMSE_ED_Univar = np.zeros((10,3,64))
ANN = np.zeros((10,10,64))
PERS = np.zeros((10,1,64))

for i in np.arange(1):
	df = pd.read_csv('./LSTM_ED_Multivar/pers/LSTM_ED_Multivar_Chan{}_pers.csv'.format(i+1),header=None)
	PERS[:,:,i] = df.values[:,:]

	df = pd.read_csv('./LSTM_ED_Multivar/model/LSTM_ED_Multivar_Chan{}_RMSE.csv'.format(i+1),header=None)
	RMSE_ED_Multivar[:,:,i] = df.values[:,:]
	
	df = pd.read_csv('./ANN/model/ANN_Chan_{}_RMSE.csv'.format(i+1),header=None)
	ANN[:,:,i] = df.values[:,:]

	df = pd.read_csv('./LSTM_Univar/model/LSTM_Univar_Channel_{}.csv'.format(i+1),header=None)
	RMSE_Univar[:,:,i] = df.values[:,:]
	
	df = pd.read_csv('./LSTM_CNN/model/LSTM_CNN_Univar_Chan_{}_RMSE.csv'.format(i+1),header=None)
	RMSE_CNN_ED_Univar[:,:,i] = df.values[:,:]
	
	df = pd.read_csv('./LSTM_ED_Univar/model/LSTM_CNN_Univar_Chan_{}_RMSE.csv'.format(i+1),header=None)
	RMSE_ED_Univar[:,:,i] = df.values[:,:]
	
	
	plt.plot(np.arange(1,11),np.mean(PERS[:,:,i],1),label='Pers. chan {}'.format(i+1))
	
	plt.plot(np.arange(1,11),np.mean(ANN[:,:,i],1),label='ANN chan {}'.format(i+1))
	plt.fill_between(np.arange(1,11),np.mean(ANN[:,:,i],1)+np.std(ANN[:,:,i],1),np.mean(ANN[:,:,i],1)-np.std(ANN[:,:,i],1),alpha=0.2)
	
	plt.plot(np.arange(1,11),np.mean(RMSE_ED_Multivar[:,:,i],1),label='multivar chan {}'.format(i+1))
	plt.fill_between(np.arange(1,11),np.mean(RMSE_ED_Multivar[:,:,i],1)+np.std(RMSE_ED_Multivar[:,:,i],1),np.mean(RMSE_ED_Multivar[:,:,i],1)-np.std(RMSE_ED_Multivar[:,:,i],1),alpha=0.2)

	plt.plot(np.arange(1,11),np.mean(RMSE_ED_Univar[:,:,i],1),label='ED univar chan {}'.format(i+1))
	plt.fill_between(np.arange(1,11),np.mean(RMSE_ED_Univar[:,:,i],1)+np.std(RMSE_ED_Univar[:,:,i],1),np.mean(RMSE_ED_Univar[:,:,i],1)-np.std(RMSE_ED_Univar[:,:,i],1),alpha=0.2)
	
	plt.plot(np.arange(1,11),np.mean(RMSE_Univar[:,:,i],1),label='univar chan {}'.format(i+1))
	plt.fill_between(np.arange(1,11),np.mean(RMSE_Univar[:,:,i],1)+np.std(RMSE_Univar[:,:,i],1),np.mean(RMSE_Univar[:,:,i],1)-np.std(RMSE_Univar[:,:,i],1),alpha=0.2)
	
	plt.plot(np.arange(1,11),np.mean(RMSE_CNN_ED_Univar[:,:,i],1),label='LSTM+CNN chan {}'.format(i+1))
	plt.fill_between(np.arange(1,11),np.mean(RMSE_CNN_ED_Univar[:,:,i],1)+np.std(RMSE_CNN_ED_Univar[:,:,i],1),np.mean(RMSE_CNN_ED_Univar[:,:,i],1)-np.std(RMSE_CNN_ED_Univar[:,:,i],1),alpha=0.2)
	
plt.xlabel('prediction (ms)')
plt.ylabel('RMSE (uV)')
plt.legend()
plt.show()
