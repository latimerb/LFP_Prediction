# Make figures

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
from scipy.signal import butter, lfilter
from scipy import signal
from scipy import fftpack
import pdb

#### LOAD DATA ####
examples = pd.read_csv('ANNexamples.csv',header=None)
examples_values = examples.values

ANN_preds = pd.read_csv('ANNexamplespreds.csv',header=None)
ANN_preds_values = ANN_preds.values

ARIMA_preds = pd.read_csv('ARIMApreds.csv',header=None)
ARIMA_preds_values = ARIMA_preds.values


# tenms = pd.read_csv('./modeloutputdata/tenms_rmse.csv',header=None)
# tenms_values = tenms.values

# fivems = pd.read_csv('./modeloutputdata/fivems_rmse.csv',header=None)
# fivems_values = fivems.values

# onems = pd.read_csv('./modeloutputdata/onems_rmse.csv',header=None)
# onems_values = onems.values


# sub4_aligned = pd.read_csv('./data/train_aligned_sub4.csv',header=None)
# sub4_al_vals = sub4_aligned.values

# gamma_segs = sub4_al_vals[np.where(sub4_al_vals[:,-1]==1)[0],:]
# nongamma_segs = sub4_al_vals[np.where(sub4_al_vals[:,-1]==0)[0],:]


# ###########################################################
# def auto_corr(wave, lag=1):
    # n = len(wave)
    # y1 = wave[lag:]
    # y2 = wave[:n-lag]
    # corr = np.corrcoef(y1, y2, ddof=0)[0, 1]
    # return corr
# ############################################################
# filter_flag = 0

# def butter_bandpass(lowcut, highcut, fs, order=5):
	# nyq = 0.5 * fs
	# low = lowcut / nyq
	# high = highcut / nyq
	# b, a = butter(order, [low, high], btype='band')
	# return b, a
	
# b_filt, a_filt = butter_bandpass(50,100,1000)

# num_lines = 1 # How many lines to load (artifact-free chunks)
# skip_rows = 0 # Skip some rows if you want
# ex_timeseries = np.empty((num_lines,900000)) #initialize dataset matrix
# f = open('./data/subject3_seg.csv', newline='')
# reader = csv.reader(f)

# for i in np.arange(0,skip_rows):
    # row = next(reader)

# for i in np.arange(0,num_lines):
    # row = next(reader)  # gets the first line
    # arr = np.array(list(map(float, row)))
    # ex_timeseries[i,0:len(arr)] = arr

# # If you want to filter then filter	
# if filter_flag == 1:
	# for i in np.arange(0,num_lines):
		# nz_seg = ex_timeseries[i,ex_timeseries[i,:]!=0]
		# nz_seg_filt = lfilter(b_filt,a_filt,nz_seg)
		# ex_timeseries[i,ex_timeseries[i,:]!=0] = nz_seg_filt

################################################################
		
#### PLOT THINGS ####

# Spike triggered average
# for i in np.arange(1,8):
	# plt.plot(np.mean(gamma_segs[:,50*(i-1):50*i],axis=0))

# plt.show()

# ex_ts_nz = ex_timeseries[0,ex_timeseries[0,:]!=0]
# fig, ax = plt.subplots()
# plt.plot(ex_ts_nz[0:330],color='black')

# for i in np.arange(1,4):
	# x = np.arange(0+110*(i-1),110*(i-1)+100,1)
	# y1 = (np.max(ex_ts_nz[0:330])+100)*np.ones((1,100,1))[0]
	# y2 = (np.min(ex_ts_nz[0:330])-100)*np.ones((1,100,1))[0]
	# y1 = y1.reshape(100,)
	# y2 = y2.reshape(100,)
	# ax.fill_between(x,y1,y2,alpha=0.3,color='#00FF00')
	# x = np.arange(100+110*(i-1),100+110*(i-1)+10,1)
	# y1 = (np.max(ex_ts_nz[0:330])+100)*np.ones((1,10,1))[0]
	# y2 = (np.min(ex_ts_nz[0:330])-100)*np.ones((1,10,1))[0]
	# y1 = y1.reshape(10,)
	# y2 = y2.reshape(10,)
	# ax.fill_between(x,y1,y2,alpha=0.3,color='#FF0000')

# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)

#AUTOCORRELATION
# X = ex_ts_nz
# X_auto = np.zeros((100,))
# for i in np.arange(0,100):
	# X_auto[i] = auto_corr(X, lag=i)
	
# plt.figure()
# plt.bar(np.arange(0,100),X_auto)
# plt.xlabel('time(ms)')
# plt.ylabel('correlation coefficient')


# #FFT
# Fs = 1000

# X = fftpack.fft(ex_ts_nz)
# freqs = fftpack.fftfreq(len(ex_ts_nz))


# plt.figure()
# ax = plt.subplot(1,1,1)
# plt.scatter(freqs, X.real, freqs, X.imag)
# ax.set_yscale('log')
# ax.set_xscale('log')
# #plt.xlim([0, Fs/2])
# plt.xlabel('frequency(Hz)')


# Proposal figure
y_pers_test = np.transpose(np.tile(examples_values[:,-10-1], (10,1)))

plt.figure(figsize=(5,2))
x = 1
for i in [5,1]:
	ax = plt.subplot(1,2,x)
	plt.plot(np.arange(1,31),examples_values[i,:],'g',label='raw LFP')
	plt.plot(np.arange(21,31),y_pers_test[i,:],'blue',label='Persis.')
	plt.plot(np.arange(21,31),ARIMA_preds_values[i,:],'red',label='ARIMA')
	plt.plot(np.arange(21,31),ANN_preds_values[i,:],'orange',label='ANN')
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	if x==1:
		plt.legend()
	x = x+1


plt.tight_layout()
plt.show()

# HEATMAP OF ERRORS
# fig, ax = plt.subplots()
# im = plt.imshow(tenms_values, vmin=0, vmax=1, cmap='jet', aspect='auto', extent = [5,100,20,400])
# cbar = plt.colorbar()
# cbar.set_label('RMSE ANN/RMSE Persistence')
# plt.xlabel('window size (ms)')
# plt.ylabel('# nodes')


# LINEPLOT OF ERRORS FOR RAW ANN

# plt.figure()
# plt.plot(np.arange(1,51),onems_values[0,:])
# plt.plot(np.arange(1,51),fivems_values[0,:])
# plt.plot(np.arange(1,51),tenms_values[0,:])
# plt.ylabel('RMSE ANN/RMSE Persistence')
# plt.xlabel('window size (ms)')
# plt.show()

