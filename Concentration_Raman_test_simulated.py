# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 09:35:28 2019

@author: admin
"""
from tensorflow import keras
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.pyplot.switch_backend('agg')
import spectra_process.subpys as subpys
import scipy.optimize as optimize
import os
import time

# Set default decvice: GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '3' # if set '-1', python runs on CPU, '0' uses 1060 6GB
# In[] CNN preprocessing: 2-2
source_data_path = './spectra_data/binary/'
save_train_model_path = './RamanNet/2021_01/'

# load CNN model
X_mean = np.load(save_train_model_path+'X_scale_mean.npy')
X_std = np.load(save_train_model_path+'X_scale_std.npy')
model = keras.models.load_model(save_train_model_path+'regression_model.h5')
print('1. Finish loading CNN binary-model!')
# load simulated dataset
X = np.load(source_data_path+'X_train.npy')
X = X[:100, :]
Y = np.load(source_data_path+'Y_train.npy')
Y = Y[:100, :]
print('2. Finish loading test dataset!') 
X = (X - X_mean)/X_std
YPredict = model.predict(np.reshape(X, (X.shape[0], X.shape[1], 1)))
print('3. Finish predicting!') 

rmse = Y[:, :2] - YPredict[:, :2]
r1 = np.sqrt(np.mean(np.power(rmse[:, 0], 2)))
r2 = np.sqrt(np.mean(np.power(rmse[:, 1], 2)))

fontsize_val = 20
plt.figure(figsize=(12, 6))
    
plt.subplot(121)
plt.plot(Y[:, 0], YPredict[:, 0], '.b',
np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), 'g',markersize=13)
#plt.xlabel('True concentrations', fontsize=fontsize_val)
plt.ylabel('Concentrations by CNN', fontsize=fontsize_val)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid()
plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
plt.text(0.05, 0.85, 'RMSE = %f'%r1, fontsize=fontsize_val)

plt.setp(plt.gca().get_lines(), linewidth=4)
plt.title('CL', fontsize=fontsize_val)

plt.subplot(122)
plt.plot(Y[:, 1], YPredict[:, 1], '.b', 
np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), 'g',markersize=13)
plt.xlabel('True concentrations', fontsize=fontsize_val)
#plt.ylabel('Concentrations by CNN', fontsize=fontsize_val)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid()
plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
plt.text(0.05, 0.85, 'RMSE = %f'%r2, fontsize=fontsize_val)

plt.setp(plt.gca().get_lines(), linewidth=4)
plt.title('CytoC', fontsize=fontsize_val)

plt.savefig('./output/prediction/2-2.png', dpi=300)
plt.show()
# In[] CNN preprocessing: 2-3
source_data_path = './spectra_data/ternary/'
save_train_model_path = './RamanNet/2021_01/'

# load CNN model
X_mean = np.load(save_train_model_path+'X_scale_mean.npy')
X_std = np.load(save_train_model_path+'X_scale_std.npy')
model = keras.models.load_model(save_train_model_path+'regression_model.h5')
print('1. Finish loading CNN binary-model!')
# load simulated dataset
X = np.load(source_data_path+'X_train.npy')
X = X[:100, :]
Y = np.load(source_data_path+'Y_train.npy')
Y = Y[:100, :]
print('2. Finish loading test dataset!') 
X = (X - X_mean)/X_std
YPredict = model.predict(np.reshape(X, (X.shape[0], X.shape[1], 1)))
print('3. Finish predicting!') 

rmse = Y[:, :2] - YPredict[:, :2]
r1 = np.sqrt(np.mean(np.power(rmse[:, 0], 2)))
r2 = np.sqrt(np.mean(np.power(rmse[:, 1], 2)))

# YPredict = YPredict
fontsize_val = 24
plt.figure(figsize=(12, 6))
    
plt.subplot(121)
plt.plot(Y[:, 0], YPredict[:, 0], '.b',
np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), 'g',markersize=13)
plt.xlabel('True concentrations', fontsize=fontsize_val)
plt.ylabel('Concentrations by CNN', fontsize=fontsize_val)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid()
plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
plt.text(0.05, 0.85, 'RMSE = %f'%r1, fontsize=fontsize_val)

plt.setp(plt.gca().get_lines(), linewidth=4)
plt.title('CL', fontsize=fontsize_val)

plt.subplot(122)
plt.plot(Y[:, 1], YPredict[:, 1], '.b',
np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), 'g',markersize=13)
#plt.xlabel('True concentrations', fontsize=fontsize_val)
#plt.ylabel('Concentrations by CNN', fontsize=fontsize_val)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid()
plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
plt.text(0.05, 0.85, 'RMSE = %f'%r2, fontsize=fontsize_val)

plt.setp(plt.gca().get_lines(), linewidth=4)
plt.title('CytoC', fontsize=fontsize_val)

plt.savefig('./output/prediction/2-3.png', dpi=300)
plt.show()
# In[] CNN preprocessing: 3-2
source_data_path = './spectra_data/binary/'
save_train_model_path = './RamanNet/2021_02/'

# load CNN model
X_mean = np.load(save_train_model_path+'X_scale_mean.npy')
X_std = np.load(save_train_model_path+'X_scale_std.npy')
model = keras.models.load_model(save_train_model_path+'regression_model.h5')
print('1. Finish loading CNN ternary-model!')
# load simulated dataset
X = np.load(source_data_path+'X_train.npy')
X = X[:100, :]
Y = np.load(source_data_path+'Y_train.npy')
Y = Y[:100, :]
print('2. Finish loading test dataset!') 
X = (X - X_mean)/X_std
YPredict = model.predict(np.reshape(X, (X.shape[0], X.shape[1], 1)))
print('3. Finish predicting!') 

rmse = Y[:, :2] - YPredict[:, :2]
r1 = np.sqrt(np.mean(np.power(rmse[:, 0], 2)))
r2 = np.sqrt(np.mean(np.power(rmse[:, 1], 2)))

fontsize_val = 20
plt.figure(figsize=(12, 6))
    
plt.subplot(121)
plt.plot(Y[:, 0], YPredict[:, 0], '.b', 
np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), 'g',markersize=13)
#plt.xlabel('True concentrations', fontsize=fontsize_val)
#plt.ylabel('Concentrations by CNN', fontsize=fontsize_val)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid()
plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
plt.text(0.05, 0.85, 'RMSE = %f'%r1, fontsize=fontsize_val)
plt.setp(plt.gca().get_lines(), linewidth=4)
#plt.title('CL', fontsize=fontsize_val)

plt.subplot(122)
plt.plot(Y[:, 1], YPredict[:, 1], '.b', 
np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), 'g',markersize=13)
#plt.xlabel('True concentrations', fontsize=fontsize_val)
# plt.ylabel('Concentrations by CNN', fontsize=fontsize_val)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid()
plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
plt.text(0.05, 0.85, 'RMSE = %f'%r2, fontsize=fontsize_val)
plt.setp(plt.gca().get_lines(), linewidth=4)
#plt.title('CytoC', fontsize=fontsize_val)

plt.savefig('./output/prediction/3-2.png', dpi=300)
plt.show()

# In[] CNN preprocessing: 3-3
source_data_path = './spectra_data/ternary/'
save_train_model_path = './RamanNet/2021_02/'

# load CNN model
X_mean = np.load(save_train_model_path+'X_scale_mean.npy')
X_std = np.load(save_train_model_path+'X_scale_std.npy')
model = keras.models.load_model(save_train_model_path+'regression_model.h5')
print('1. Finish loading CNN ternary-model!')
# load simulated dataset
X = np.load(source_data_path+'X_train.npy')
X = X[:100, :]
Y = np.load(source_data_path+'Y_train.npy')
Y = Y[:100, :]
print('2. Finish loading test dataset!') 
X = (X - X_mean)/X_std
YPredict = model.predict(np.reshape(X, (X.shape[0], X.shape[1], 1)))
print('3. Finish predicting!') 

rmse = Y[:, :3] - YPredict[:, :3]
r1 = np.sqrt(np.mean(np.power(rmse[:, 0], 2)))
r2 = np.sqrt(np.mean(np.power(rmse[:, 1], 2)))
r3 = np.sqrt(np.mean(np.power(rmse[:, 2], 2)))

fontsize_val = 20
plt.figure(figsize=(12, 6))
    
plt.subplot(221)
plt.plot(Y[:, 0], YPredict[:, 0], '.b', 
np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), 'g',markersize=13)
#plt.xlabel('True concentrations', fontsize=fontsize_val)
#plt.ylabel('Concentrations by CNN', fontsize=fontsize_val)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid()
plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
plt.text(0.05, 0.85, 'RMSE = %f'%r1, fontsize=fontsize_val)
plt.setp(plt.gca().get_lines(), linewidth=4)
plt.title('CL', fontsize=fontsize_val)

plt.subplot(222)
plt.plot(Y[:, 1], YPredict[:, 1], '.b',
np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), 'g',markersize=13)
#plt.xlabel('True concentrations', fontsize=fontsize_val)
# plt.ylabel('Concentrations by CNN', fontsize=fontsize_val)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid()
plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
plt.text(0.05, 0.85, 'RMSE = %f'%r2, fontsize=fontsize_val)
plt.setp(plt.gca().get_lines(), linewidth=4)
plt.title('CytoC', fontsize=fontsize_val)


plt.subplot(223)
plt.plot(Y[:, 2], YPredict[:, 2], '.b',
np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), 'g',markersize=13)
#plt.xlabel('True concentrations', fontsize=fontsize_val)
# plt.ylabel('Concentrations by CNN', fontsize=fontsize_val)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid()
plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
plt.text(0.05, 0.85, 'RMSE = %f'%r3, fontsize=fontsize_val)
plt.setp(plt.gca().get_lines(), linewidth=4)
plt.title('DNA', fontsize=fontsize_val)


plt.savefig('./output/prediction/3-3.png', dpi=300)
plt.show()
# In[] 
CSV_Path = './output/train_status/ternary_train/'
train_acc = pd.read_csv(CSV_Path+'run-train-tag-epoch_accuracy.csv').values
train_loss = pd.read_csv(CSV_Path+'run-train-tag-epoch_loss.csv').values
lr = pd.read_csv(CSV_Path+'run-train-tag-epoch_lr.csv').values

valid_acc = pd.read_csv(CSV_Path+'run-validation-tag-epoch_accuracy.csv').values
valid_loss = pd.read_csv(CSV_Path+'run-validation-tag-epoch_loss.csv').values



fontsize_val = 20
plt.figure(figsize=(12, 12))
plt.subplot(211)
plt.plot(train_acc[:, 0], train_acc[:, 1], '-k')
plt.plot(valid_acc[:, 0], valid_acc[:, 1], '--b')
plt.xlabel('Epoch', fontsize=fontsize_val)
plt.ylabel('Accuracy', fontsize=fontsize_val)
# plt.xlim(0, 1)
# plt.ylim(0, 1)
plt.grid()
plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
plt.setp(plt.gca().get_lines(), linewidth=4)

plt.legend(['training', 'validation'], loc=0, fontsize=fontsize_val)

plt.subplot(212)
plt.plot(train_loss[:, 0], train_loss[:, 1], '-k')
plt.plot(valid_loss[:, 0], valid_loss[:, 1], '--b')
plt.xlabel('Epoch', fontsize=fontsize_val)
plt.ylabel('Loss', fontsize=fontsize_val)
# plt.xlim(0, 1)
# plt.ylim(0, 1)
plt.grid()
plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
plt.setp(plt.gca().get_lines(), linewidth=4)

plt.legend(['training', 'validation'], loc=0, fontsize=fontsize_val)

plt.savefig('./output/prediction/acc.png', dpi=300)
plt.show()


plt.figure(figsize=(13, 7))
plt.plot(lr[:, 0], lr[:, 1], '-k')
plt.xlabel('Epoch', fontsize=fontsize_val)
plt.ylabel('Accuracy', fontsize=fontsize_val)
# plt.xlim(0, 1)
# plt.ylim(0, 1)
plt.grid()
plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
plt.setp(plt.gca().get_lines(), linewidth=4)

plt.legend(['learning rate'], loc=0, fontsize=fontsize_val)

plt.savefig('./output/prediction/lr.png', dpi=300)
plt.show()







