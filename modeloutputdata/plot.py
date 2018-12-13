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

i = 30 #channel 31
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
	
ax = plt.subplot(1,1,1)
plt.bar(np.arange(1,11)-0.3,np.mean(PERS[:,:,i],1),label='Persistence'.format(i+1),width=0.2,color='#0000FF')
plt.bar(np.arange(1,11)-0.1,np.mean(ANN[:,:,i],1),label='MLP'.format(i+1),width=0.2,color='green')
#plt.fill_between(np.arange(1,11),np.mean(ANN[:,:,i],1)+np.std(ANN[:,:,i],1),np.mean(ANN[:,:,i],1)-np.std(ANN[:,:,i],1),alpha=0.2)
#plt.bar(np.arange(1,11)+0.1,np.mean(RMSE_CNN_ED_Univar[:,:,i],1),label='LSTM+CNN'.format(i+1),width=0.2,color='orange')
plt.bar(np.arange(1,11)+0.1,np.mean(RMSE_Univar[:,:,i],1),label='Univariate LSTM'.format(i+1),width=0.2,color='orange')
plt.bar(np.arange(1,11)+0.3,np.mean(RMSE_ED_Multivar[:,:,i],1),label='Multivariate LSTM'.format(i+1),width=0.2,color='#FF0000')
plt.xticks([1,2,3,4,5,6,7,8,9,10])
#plt.fill_between(np.arange(1,11),np.mean(RMSE_ED_Multivar[:,:,i],1)+np.std(RMSE_ED_Multivar[:,:,i],1),np.mean(RMSE_ED_Multivar[:,:,i],1)-np.std(RMSE_ED_Multivar[:,:,i],1),alpha=0.2)
#plt.plot(np.arange(1,11),np.mean(RMSE_ED_Univar[:,:,i],1),label='ED univar chan {}'.format(i+1))
#plt.fill_between(np.arange(1,11),np.mean(RMSE_ED_Univar[:,:,i],1)+np.std(RMSE_ED_Univar[:,:,i],1),np.mean(RMSE_ED_Univar[:,:,i],1)-np.std(RMSE_ED_Univar[:,:,i],1),alpha=0.2)
#plt.bar(np.arange(1,11),np.mean(RMSE_Univar[:,:,i],1),label='univar chan {}'.format(i+1))
#plt.fill_between(np.arange(1,11),np.mean(RMSE_Univar[:,:,i],1)+np.std(RMSE_Univar[:,:,i],1),np.mean(RMSE_Univar[:,:,i],1)-np.std(RMSE_Univar[:,:,i],1),alpha=0.2)
#plt.fill_between(np.arange(1,11),np.mean(RMSE_CNN_ED_Univar[:,:,i],1)+np.std(RMSE_CNN_ED_Univar[:,:,i],1),np.mean(RMSE_CNN_ED_Univar[:,:,i],1)-np.std(RMSE_CNN_ED_Univar[:,:,i],1),alpha=0.2)
	
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlabel('prediction (ms)')
plt.ylabel('RMSE (uV)')
plt.tight_layout()
plt.legend()

# Plot predictions

df = pd.read_csv('./ANN/model/ANN_Chan_{}_test.csv'.format(i+1),header=None)
ANN_test = df.values[:,:]

df = pd.read_csv('./ANN/pers/ANN_Chan_{}_preds.csv'.format(i+1),header=None)
PERS_preds = df.values[:,:]

df = pd.read_csv('./ANN/model/ANN_Chan_{}_preds.csv'.format(i+1),header=None)
ANN_preds = df.values[:,:]

df = pd.read_csv('./LSTM_CNN/model/LSTM_CNN_Univar_Chan_{}_preds.csv'.format(i+1),header=None)
LSTM_CNN_preds = df.values[:,:]

df = pd.read_csv('./LSTM_ED_Multivar/model/LSTM_ED_Multivar_Chan{}_preds.csv'.format(i+1),header=None)
LSTM_Multivar_preds = df.values[:,:]



plt.figure()
for j in np.arange(4):
	plt.subplot(2,2,j+1) 
	plt.plot(ANN_test.ravel(),label='actual',color='black')
	if j==0:
		plt.plot(PERS_preds.ravel(),label='ANN prediction',color='#0000FF')
	if j==1:
		plt.plot(ANN_preds.ravel(),label='ANN prediction',color='green')
	if j==2:
		plt.plot(LSTM_CNN_preds.ravel(), label = 'LSTM+CNN prediction',color='orange')
	if j==3:
		plt.plot(LSTM_Multivar_preds.ravel(), label = 'LSTM Multivariate prediction',color='#FF0000')
	#plt.axis('off')
	plt.xlim(5000,5500)
	plt.ylim(-180,260)

plt.tight_layout(pad = 0, w_pad=-3, h_pad=-3)	









plt.show()
