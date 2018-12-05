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

set_random_seed(27365)


############### FILTER PROPERTIES #######################
filter_flag = 0 # Set to 0 if using raw to train, 1 if using filtered

def butter_bandpass(lowcut, highcut, fs, order=2):
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	b, a = butter(order, [low, high], btype='band')
	return b, a

def butter_lowpass(cutoff, fs, order=2):
	nyq = 0.5 * fs
	Ncutoff = cutoff / nyq
	b, a = butter(order, Ncutoff, btype='low')
	return b, a
	
def butter_highpass(cutoff, fs, order=2):
	nyq = 0.5 * fs
	Ncutoff = cutoff / nyq
	b, a = butter(order, Ncutoff, btype='high')
	return b, a	

################# LOAD THE DATA #########################

num_lines = 28 # How many lines to load (artifact-free chunks)
skip_rows = 0 # Skip some rows if you want

dataset = np.empty((num_lines,3000000)) #initialize dataset matrix
filt_dataset = np.empty((4,num_lines,3000000)) #initialize dataset matrix

f = open('./data/subject3_seg.csv', newline='')
reader = csv.reader(f)

for i in np.arange(0,skip_rows):
    row = next(reader)

for i in np.arange(0,num_lines):
    row = next(reader)  # gets the first line
    arr = np.array(list(map(float, row)))
    dataset[i,0:len(arr)] = arr

# If you want to filter then filter	
b_filt, a_filt = butter_bandpass(30,100,1000)
b_filt1, a_filt1 = butter_lowpass(5,1000)
b_filt2, a_filt2 = butter_bandpass(5,30,1000)
b_filt3, a_filt3 = butter_highpass(100,1000)

for i in np.arange(0,num_lines):
	nz_seg = dataset[i,dataset[i,:]!=0]
	nz_seg_filt = lfilter(b_filt,a_filt,nz_seg)
	nz_seg_filt1 = lfilter(b_filt1,a_filt1,nz_seg)
	nz_seg_filt2 = lfilter(b_filt2,a_filt2,nz_seg)
	nz_seg_filt3 = lfilter(b_filt3,a_filt3,nz_seg)
	
	filt_dataset[0,i,dataset[i,:]!=0] = nz_seg_filt
	filt_dataset[1,i,dataset[i,:]!=0] = nz_seg_filt1
	filt_dataset[2,i,dataset[i,:]!=0] = nz_seg_filt2
	filt_dataset[3,i,dataset[i,:]!=0] = nz_seg_filt3

##########################################################
# X = np.zeros((21684,5))
# X[:,0] = dataset[0,dataset[0,:]!=0]
# X[:,1] = filt_dataset[0,0,dataset[0,:]!=0]
# X[:,2] = filt_dataset[1,0,dataset[0,:]!=0]
# X[:,3] = filt_dataset[2,0,dataset[0,:]!=0]
# X[:,4] = filt_dataset[3,0,dataset[0,:]!=0]

#np.savetxt('ex_LFP_5chan.csv',X,delimiter=',')
#pdb.set_trace()
# ############ SERIES -> SUPERVISED #######################
n_sims = 1

onems_rmse = np.zeros((n_sims,n_sims))
fivems_rmse = np.zeros((n_sims,n_sims))
tenms_rmse = np.zeros((n_sims,n_sims))

for k in np.arange(1,n_sims+1):
	num_lookback = k*5 # Size of window
	#for m in np.arange(1,n_sims+1):
	nnode = 400#m*2
	print("nnode: ", nnode)
	num_predict = 10 # Number of ms to predict
	num_segs = 5000 # How many rows in the supervised dataset
		
	print("num_lookback:", num_lookback)
	chunk_size = num_predict + num_lookback

	if chunk_size*num_segs > dataset[dataset!=0].shape[0]:
		print ('num_segs too large. only {} chunks possible'.format(int(dataset[dataset!=0].shape[0]/chunk_size)))
			
	else:
		supervised_dataset = np.zeros((2,num_segs,num_lookback+num_predict))

		t_start = time.time()
		ds_row = 0 #Start at row zero of dataset and make chunks until it's exhausted
		ind = 0
		for i in np.arange(0,num_segs):
			# if we have arrived at the end of the row, go to the next row
			if (ind+1)*chunk_size > dataset[ds_row,dataset[ds_row,:]!=0].shape[0]:
				ds_row = ds_row + 1
				ind = 0
			# save it to the supervised dataset
			supervised_dataset[0,i,:] = dataset[ds_row,0+ind*chunk_size:(ind+1)*chunk_size]
			supervised_dataset[1,i,:] = filt_dataset[0,ds_row,0+ind*chunk_size:(ind+1)*chunk_size]
			ind = ind + 1
			
		print("dataset took ", time.time() - t_start, "seconds to generate")
		print(supervised_dataset.shape)
	
	###########################################################


	######### SCALE THE SUPERVISED DATASET ####################
	## Unscaled (original) is supervised_dataset
	## Scaled (0 to 1) is scl_sup_ds

	# Scaled supervised dataset
	scl_sup_ds_raw = np.zeros((num_segs,num_lookback+num_predict))
	scl_sup_ds_filt = np.zeros((num_segs,num_lookback+num_predict))
	sclr_raw = []
	sclr_filt = []

	def scalewindow(short_seg):
		short_seg = short_seg.reshape(-1,1)
		scaler = MinMaxScaler(feature_range=(0,1))
		scaled = scaler.fit_transform(short_seg)
		scaled = scaled.reshape(len(short_seg),)
		return scaled, scaler

	for i in np.arange(0,supervised_dataset.shape[1]):
		
		scl_sup_ds_raw[i,:], scl_r = scalewindow(supervised_dataset[0,i,:])  
		
		scl_sup_ds_filt[i,:], scl_f = scalewindow(supervised_dataset[1,i,:])  
		
		sclr_raw.append(scl_r)
		sclr_filt.append(scl_f)
			
	# Plot one segment to show scaling
	# plt.figure()
	# plt.subplot(2,1,1)
	# plt.plot(scl_sup_ds[10,:])
	# plt.subplot(2,1,2)
	# plt.plot(supervised_dataset[10,:])
	#################################################################
	######## SPLIT DATA INTO TRAIN/TEST and INPUT/OUTPUT ############
	N_train = int(0.8*num_segs)

	X_train = scl_sup_ds_raw[0:N_train,0:-num_predict]
	Y_train = scl_sup_ds_raw[0:N_train,-num_predict:]
	X_test = scl_sup_ds_raw[N_train:,0:-num_predict]
	Y_test = scl_sup_ds_raw[N_train:,-num_predict:]

	print(X_train.shape)
	##################################################################

	############## BUILD AND FIT MODEL ###############################
		
		
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
	history = model.fit(X_train,Y_train, epochs=20,verbose=1)
	print("model took ", time.time()-t_start, "seconds to fit")
	# plt.figure()
	# plt.plot(history.history['acc'])

	###################################################################

	############ MAKE PREDICTIONS AND UNDO SCALING ######################
	y_pred_test = model.predict(X_test)

	def unscale(X, Y, S):
		for i in range(X.shape[0]):
			seg = X[i,:]
			seg = seg.reshape(-1,1)
			us_X = S[i].inverse_transform(seg)
			us_X1d = us_X.reshape(us_X.shape[0],)
			
			Y[i,:] = us_X1d
		return Y
			
	y_pred_test_us = np.zeros((y_pred_test.shape[0],y_pred_test.shape[1]))
	y_pred_test_us = unscale(y_pred_test, y_pred_test_us, sclr_raw[N_train:])

	Y_test_us = np.zeros((Y_test.shape[0],Y_test.shape[1]))
	Y_test_us = unscale(Y_test, Y_test_us, sclr_raw[N_train:])

	rmse_ann_test = np.sqrt(np.mean((y_pred_test_us-Y_test_us)**2,axis=0))
	print("rmse for testing: ", rmse_ann_test)

	# Y_PERS IS ON RAW if 0, FILT if 1
	y_pers_test = np.transpose(np.tile(supervised_dataset[0,N_train:,-num_predict-1], (num_predict,1)))

	rmse_pers_test = np.sqrt(np.mean((y_pers_test-Y_test_us)**2,axis=0))
	rmse_pers_test_std = np.sqrt(np.std((y_pers_test-Y_test_us)**2,axis=0))
	print("rmse for (persistence) testing: ", rmse_pers_test)
		
	#onems_rmse[0,k-1] = rmse_ann_test[0]/rmse_pers_test[0]
	#fivems_rmse[0,k-1] = rmse_ann_test[4]/rmse_pers_test[4]
	#tenms_rmse[0,k-1] = rmse_ann_test[9]/rmse_pers_test[9]
		
	
		
	############## PLOT SOME EXAMPLES #######################
	examples = np.zeros((6,chunk_size))
	ex_preds = np.zeros((6,num_predict))
	plt.figure()
	
	for i in np.arange(1,7):
		sample = np.random.randint(0,Y_test.shape[0])
		ax = plt.subplot(2,3,i)
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		plt.plot(np.linspace(1,chunk_size,num=chunk_size),scl_sup_ds_raw[N_train+sample,0:chunk_size],color='#00FF00')
		#plt.plot(np.linspace(1,chunk_size,num=chunk_size),supervised_dataset[1,N_train+sample,0:chunk_size],color='#00FF00')
		plt.plot(np.linspace(num_lookback+1,chunk_size,num=num_predict),Y_test[sample,:],color='#FF0000')
		plt.plot(np.linspace(num_lookback+1,chunk_size,num=num_predict),y_pred_test[sample,:],color='orange')
		
		examples[i-1,:] = supervised_dataset[0,N_train+sample,0:chunk_size]
		ex_preds[i-1,:] = y_pred_test_us[sample,:]
		
	plt.tight_layout()

	

fig2 = plt.figure()
plt.subplot(2,1,1)
plt.bar(np.arange(0,num_predict)-0.2,rmse_pers_test*0.1,0.3,label='persistence')
plt.errorbar(np.arange(0,num_predict)-0.2,rmse_pers_test*0.1,yerr = rmse_pers_test_std*0.1)
plt.bar(np.arange(0,num_predict)+0.2,rmse_ann_test*0.1,0.3,label='MLP')
plt.ylabel('error in uV')
plt.legend()
	
plt.subplot(2,1,2)
plt.bar(np.arange(1,num_predict+1),100*rmse_ann_test/rmse_pers_test)
plt.plot(np.arange(0,12),100*np.ones((12,)),'r--')
plt.xlim(0,11)
#fig2.savefig('filt_rmse.png')


# Analyze segments that performed well to find out why
Y_error = np.sqrt(np.mean((y_pred_test_us-Y_test_us)**2,axis=1))
low_err = np.where(Y_error<70)[0]


low_err_chunks = supervised_dataset[0,N_train+low_err,:]
low_err_preds = y_pred_test_us[low_err,:]

plt.figure()

#num_to_plot = low_err_chunks.shape[0]
num_to_plot = 36

for i in range(num_to_plot):
	num_sp = np.ceil(np.sqrt(num_to_plot))
	plt.subplot(num_sp,num_sp,i+1)
	plt.plot(low_err_chunks[i,:])
	plt.plot(np.arange(low_err_chunks.shape[1]-10,low_err_chunks.shape[1]),low_err_preds[i,:],'r')



plt.show()

#np.savetxt("onems_rmse.csv", onems_rmse, delimiter=",")
#np.savetxt("fivems_rmse.csv", fivems_rmse, delimiter=",")
#np.savetxt("tenms_rmse.csv", tenms_rmse, delimiter=",")

#plt.subplot(1,3,1)
#plt.imshow(onems_rmse, vmin=0, vmax=1, cmap='jet', aspect='auto')
#plt.subplot(1,3,2)
#plt.imshow(fivems_rmse, vmin=0, vmax=1, cmap='jet', aspect='auto')
#plt.subplot(1,3,3)
#plt.imshow(tenms_rmse, vmin=0, vmax=1, cmap='jet', aspect='auto')
#plt.colorbar()
#plt.show()