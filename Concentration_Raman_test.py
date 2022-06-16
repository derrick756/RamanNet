# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 09:35:28 2021

@author: admin
"""
from tensorflow import keras
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import spectra_process.subpys as subpys
import scipy.optimize as optimize
import os
import time

# Set default decvice: GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
# In[] read pure_spectra and mixture, then smoothing
ternary_or_quaternary = 0 # non-zero is ternary, zero is quaternary

is_smoother = 0 # non-zero is yes
#source_data_path = './spectra_data/ternary/'
#save_train_model_path = './RamanNet/2021_01/'

source_data_path = 'H:/spectra_data/quaternary/'
save_train_model_path = 'H:/RamanNet/2021_02/'

data = np.load('H:/spectra_data/my_measurement_Lipo11.npz')
CH = data['CH_An']
Cur = data['Cur_An']
PC = data['PC_An']
DPPC = data['DPPC_An']
qtz = data['qtz_ss']
wn = data['wn']
names = data['pure_names']
Lipo_An= data['Lipo_An']
#Lipo_An_ternary=data['Lipo_An_ternary']
exp = data['exposure']


#real_ternary_relative_coeffs = np.array([[0.4153, 0.0889, 0.1017, 0.1152, 0.5007, 0.0631, 0.1889, 0.1768, 0.0605, 0.2024, 0.0905, 0.1921, 0.6783, 0.1594, 0.3531,	0.5479, 0.0683, 0.2705,	0.1226, 0.1708, 0.4578,	0.1813, 0.0916, 0.1396,	0.1207,	0.1332,	0.1184,	0.1557, 0.0200, 0.3464],  
#                                        [ 0.0400,	0.0209,	0.0105,	0.0113,	0.2301,	0.0196,	0.0154,	0.0353,	0.0386,	0.0154,	-0.0032,	0.0428,	-0.0046,	0.0173,	0.0744,	0.1695,	0.0134,	-0.0078,	0.0223,	0.0400,	0.0343,	0.0326,	0.0009,	0.0069,	0.0462,	0.0124,	0.0084,	-0.0017,	0.0103,	0.0168], 
#                                        [0.2052, 0.2432,	0.0694,	0.1254,	0.5597,	0.3216,	0.0917,	0.1221,	0.0379,	0.1428,	0.0411,	0.1598,	0.5481,	0.1169,	0.2076,	0.5972,	0.1481,	0.1426,	0.0981,	0.1550,	0.3496,	0.1905,	0.1185,	0.0971,	0.0046,	0.1665,	0.1229,	0.1662,	0.2471,	0.2798]])

real_quaternary_relative_coeffs = np.array([[0.4153, 0.0889, 0.1017, 0.1152, 0.5007, 0.0631, 0.1889, 0.1768, 0.0605, 0.2024, 0.0905, 0.1921, 0.6783, 0.1594, 0.3531,	0.5479, 0.0683, 0.2705,	0.1226, 0.1708, 0.4578,	0.1813, 0.0916, 0.1396,	0.1207,	0.1332,	0.1184,	0.1557, 0.0200, 0.3464 ],  
                                         [ 0.0400,	0.0209,	0.0105,	0.0113,	0.2301,	0.0196,	0.0154,	0.0353,	0.0386,	0.0154,	-0.0032,	0.0428,	-0.0046,	0.0173,	0.0744,	0.1695,	0.0134,	-0.0078,	0.0223,	0.0400,	0.0343,	0.0326,	0.0009,	0.0069,	0.0462,	0.0124,	0.0084,	-0.0017,	0.0103,	0.0168], 
                                         [ 0.2052, 0.2432,	0.0694,	0.1254,	0.5597,	0.3216,	0.0917,	0.1221,	0.0379,	0.1428,	0.0411,	0.1598,	0.5481,	0.1169,	0.2076,	0.5972,	0.1481,	0.1426,	0.0981,	0.1550,	0.3496,	0.1905,	0.1185,	0.0971,	0.0046,	0.1665,	0.1229,	0.1662,	0.2471,	0.2798],
                                         [ 0.2381, 0.1481,	0.0694,	0.1194,	0.1571,	0.1717,	0.1315,	0.1076,	0.0565,	0.1798,	0.0515,	0.1063,	0.3690,	0.1377,	0.1520,	0.3669,	0.1029,	0.1266,	0.0539,	0.0655,	0.2546,	0.1189,	0.0864,	0.1009,	0.2568,	0.02354,	0.0631,	0.0519,	0.1386,	0.1070]])



#  make pure array
pures = np.reshape(np.concatenate((CH, Cur, PC,
                                  DPPC )), (4, wn.shape[0])) 
pures = pures[[0, 1, 2,3], :]
# pures = pures[[0, 1, 2], :]
# In[] CNN preprocessing
X_mean = np.load(save_train_model_path+'X_scale_mean.npy')
X_std = np.load(save_train_model_path+'X_scale_std.npy')
model = keras.models.load_model(save_train_model_path+'regression_model.h5')
print('Finish loading CNN model!')
################################################################################

mixture = Lipo_An
        
real_coeffs = np.transpose(real_quaternary_relative_coeffs)

# In[] smoothing and display test spectra
current_time = time.time()
mss = np.transpose(subpys.whittaker_smooth(spectra=np.transpose(mixture), lmbda=0.5, d=2))
asls_smooth = time.time() - current_time
show_choice = -2
fontsize_val = 16
plt.figure(figsize=(10, 6))
plt.plot(np.transpose(wn), np.transpose(mixture[show_choice, :]),
         np.transpose(wn), np.transpose(mss[show_choice, :]))

plt.title('smoothed spectra')
plt.xlabel('wavenumber (cm-1)')
plt.ylabel('intensity (a.u.)')
plt.legend(["Measured","Fitting"], loc=0, fontsize=fontsize_val)
plt.xlim((np.min(wn), np.max(wn)))
plt.setp(plt.gca().get_lines(), linewidth=3)
plt.show()

# In[]  normalize
current_time = time.time()
if is_smoother != 0:
    mixture = mss

mixture =  mixture - np.transpose(np.tile(np.min(mixture, axis=1), (mixture.shape[1], 1))) 
X = (mixture - X_mean)/X_std
cnn_normalize = time.time() - current_time
      
# In[] load trained model and predict
current_time = time.time()
YPredict = model.predict(np.reshape(X, (X.shape[0], X.shape[1], 1)))
cnn_predict = time.time() - current_time
print('Finish predicting!')
# In[] LS to remove backgrounds
current_time = time.time()
no_bg_mss = np.zeros(mss.shape)
ls_coeffs = np.zeros((YPredict.shape[0], pures.shape[0]))
poly_bg = np.concatenate((qtz, subpys.myploy(4, pures.shape[1])))

#poly_pures = pures
for ij in range(ls_coeffs.shape[0]):
    tmpCoeff = subpys.asls(poly_bg, np.reshape(mss[ij, :], (1, len(wn))), 0.01)
    no_bg_mss[ij, :] = mss[ij, :] - np.matmul(tmpCoeff, poly_bg)
asls_bg_cor = time.time() - current_time

# In[] LS fitting to get concentrations
current_time = time.time()
ls_coeffs = np.zeros((YPredict.shape[0], pures.shape[0]))
poly_pures = np.concatenate((pures, subpys.myploy(4, pures.shape[1])))
#poly_pures = pures
for ij in range(ls_coeffs.shape[0]):
    tmpCoeff = subpys.asls(poly_pures, np.reshape(no_bg_mss[ij, :], (1, len(wn))), 0.5)

    ls_coeffs[ij, :] = tmpCoeff[:pures.shape[0]]
asls_predict = time.time() - current_time

# In[] plot pre-processing
recovered = np.matmul(YPredict[:, :4], pures[:4, :])
ls_recovered = np.matmul(ls_coeffs[:, :4], pures[:4, :])

# recovered = np.matmul(YPredict[:, :3], pures[:3, :])
# ls_recovered = np.matmul(ls_coeffs[:, :3], pures[:3, :])

cnn_mse = np.mean((recovered-no_bg_mss)**2, axis=1)
ls_mse = np.mean((ls_recovered-no_bg_mss)**2, axis=1)

sub_YPredict = YPredict[:, :3]
sub_ls_coeffs = ls_coeffs[:, :3]
sub_real_coeffs = real_coeffs[:, :3]  

mixture_norm = no_bg_mss/np.max(no_bg_mss)

recovered_norm = recovered/np.max(recovered)
YPredict_norm = sub_YPredict/np.transpose(np.tile(np.sum(sub_YPredict, axis=1), (sub_YPredict.shape[1], 1)))

ls_recovered_norm = ls_recovered/np.max(ls_recovered)
ls_coeffs_norm = sub_ls_coeffs/np.transpose(np.tile(np.sum(sub_ls_coeffs, axis=1), (sub_ls_coeffs.shape[1], 1)))

sub_real_coeffs = sub_real_coeffs/np.transpose(np.tile(np.sum(sub_real_coeffs, axis=1), (sub_real_coeffs.shape[1], 1)))
sub_real_coeffs_norm = np.zeros(YPredict_norm.shape)
for ij in range(sub_real_coeffs_norm.shape[1]):
    sub_real_coeffs_norm[:, ij] = np.squeeze(np.reshape(np.transpose(np.tile(sub_real_coeffs[:, ij], (np.int32(sub_YPredict.shape[0]/sub_real_coeffs.shape[0]), 1))), (sub_real_coeffs_norm.shape[0], 1)))

cnn_coeff_mse = np.reshape(np.mean((YPredict_norm - sub_real_coeffs_norm)**2, axis=1), (YPredict_norm.shape[0], 1))
ls_coeff_mse = np.reshape(np.mean((ls_coeffs_norm - sub_real_coeffs_norm)**2, axis=1), (YPredict_norm.shape[0], 1))




# In[] plot result of CNN


fontsize_val = 32
plt.figure(figsize=(30, 15))
plt.subplot(221)

plt.plot(np.transpose(np.arange(sub_YPredict.shape[0]))+1, sub_YPredict)
plt.title('concentrations predicted by CNN', fontsize=fontsize_val)
plt.xlabel('measurements', fontsize=fontsize_val)
plt.ylabel('Coefficients', fontsize=fontsize_val)
plt.legend(names, loc=1, fontsize=fontsize_val)
plt.grid()
plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)


plt.setp(plt.gca().get_lines(), linewidth=5)
plt.xlim((1, YPredict.shape[0]))
#plt.ylim((0, 1))

plt.subplot(223)
plt.plot(np.transpose(np.arange(YPredict_norm.shape[0]))+1, YPredict_norm[:, :])
# plt.title('relative concentrations by CNN', fontsize=fontsize_val)
plt.xlabel('Measurements', fontsize=fontsize_val)
plt.ylabel('Normalized coefficients', fontsize=fontsize_val)
plt.xlim((1, YPredict_norm.shape[0]))
plt.legend(names, loc=1, fontsize=fontsize_val)
plt.grid()
plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)


plt.setp(plt.gca().get_lines(), linewidth=5)
plt.ylim((0, 1))
plt.subplot(222)
plt.plot(wn, np.transpose(recovered),
         wn, -np.transpose(no_bg_mss), linewidth=0.5)
plt.title('fitted and measured spectra', fontsize=fontsize_val)
plt.xlabel('Raman shift (cm-1)', fontsize=fontsize_val)
plt.ylabel('Intensity (a.u.)', fontsize=fontsize_val)
plt.xlim((np.min(wn), np.max(wn)))
plt.grid()
plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
plt.text(600, 0.5*np.max([recovered, no_bg_mss]), 'Fitted', fontsize=fontsize_val)
plt.text(600, -0.5*np.max([recovered, no_bg_mss]),'Measured', fontsize=fontsize_val)
plt.setp(plt.gca().get_lines(), linewidth=2)
plt.ylim((-np.max([recovered, no_bg_mss]), np.max([recovered, no_bg_mss])))

plt.subplot(224)
plt.plot(wn, np.transpose(recovered_norm),
         wn, -np.transpose(mixture_norm), linewidth=0.5)
# plt.title('fitted and raw spectra', fontsize=fontsize_val)
plt.xlabel('Raman shift ($cm^{-1}$)', fontsize=fontsize_val)
plt.ylabel('Normalized intensity (a.u.)', fontsize=fontsize_val)
plt.xlim((np.min(wn), np.max(wn)))
plt.grid()
plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
plt.text(600, 0.5, 'Fitted', fontsize=fontsize_val)
plt.text(600, -0.5,'Measured', fontsize=fontsize_val)
plt.setp(plt.gca().get_lines(), linewidth=2)


plt.savefig('H:/spectra_data/1_output_cnn.png', dpi=300)
plt.show()
# In[] plot result of LS
fontsize_val = 32
plt.figure(figsize=(30, 15))
plt.subplot(221)
plt.plot(np.transpose(np.arange(sub_ls_coeffs.shape[0]))+1, sub_ls_coeffs)
plt.title('concentrations predicted by AsLS ', fontsize=fontsize_val)
plt.xlabel('Measurements', fontsize=fontsize_val)
plt.ylabel('Coefficients', fontsize=fontsize_val)
plt.xlim((1, ls_coeffs.shape[0]))
plt.legend(names, loc=1, fontsize=fontsize_val)
plt.grid()
plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
plt.setp(plt.gca().get_lines(), linewidth=5)

plt.subplot(223)
plt.plot(np.transpose(np.arange(ls_coeffs_norm.shape[0]))+1, ls_coeffs_norm[:, :])
# plt.title('relative concentrations by AsLS', fontsize=fontsize_val)
plt.xlabel('measurements', fontsize=fontsize_val)
plt.ylabel('Normalized coefficients', fontsize=fontsize_val)
plt.xlim((1, ls_coeffs_norm.shape[0]))
plt.legend(names, loc=1, fontsize=fontsize_val)
plt.grid()
plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
plt.setp(plt.gca().get_lines(), linewidth=5)
plt.ylim((0, 1))

plt.subplot(222)
plt.plot(wn, np.transpose(ls_recovered),
         wn, -np.transpose(no_bg_mss), linewidth=0.5)
plt.title('Fitted and measured spectra', fontsize=fontsize_val)
plt.xlabel('Raman shift ($cm^{-1}$)', fontsize=fontsize_val)
plt.ylabel('Intensity (a.u.)', fontsize=fontsize_val)
plt.xlim((np.min(wn), np.max(wn)))
plt.ylim((-np.max([ls_recovered, no_bg_mss]), np.max([ls_recovered, no_bg_mss])))
plt.grid()
plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
plt.text(600, 0.5*np.max([recovered, no_bg_mss]), 'Fitted', fontsize=fontsize_val)
plt.text(600, -0.5*np.max([recovered, no_bg_mss]),'Measured', fontsize=fontsize_val)
plt.setp(plt.gca().get_lines(), linewidth=2)

plt.subplot(224)
plt.plot(wn, np.transpose(ls_recovered_norm),
         wn, -np.transpose(mixture_norm), linewidth=0.5)
#plt.title('Fitted and measured spectra', fontsize=fontsize_val)
plt.xlabel('Raman shift ($cm^{-1}$)', fontsize=fontsize_val)
plt.ylabel('Normalized intensity (a.u.)', fontsize=fontsize_val)
plt.xlim((np.min(wn), np.max(wn)))
plt.grid()
plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
plt.text(600, 0.5, 'Fitted', fontsize=fontsize_val)
plt.text(600, -0.5,'Measured', fontsize=fontsize_val)
plt.setp(plt.gca().get_lines(), linewidth=2)

plt.savefig('H:/spectra_data/2_output_ls.png', dpi=300)
plt.show()

fontsize_val = 25
if recovered.shape[0] <= 30:
    div_cols = 3
    div_rows = np.int32(np.ceil(recovered.shape[0]/div_cols))
    plt.figure(figsize=(div_cols*12, div_rows*7))
    count = 1
    for r in range(div_rows):
        for c in range(div_cols):
            if count <= recovered.shape[0]:
                plt.subplot(div_rows, div_cols, count)
                plt.plot(wn, np.transpose(no_bg_mss[count-1, :]), 'g', 
                         wn, np.transpose(recovered[count-1, :]), 'r--',
                         wn, -np.transpose(no_bg_mss[count-1, :]), 'g',
                         wn, -np.transpose(ls_recovered[count-1, :]), 'r--')        
#                plt.title('group '+str(count), fontsize=12)
#                plt.text(290, np.max([recovered[count-1, :], no_bg_mss[count-1, :]])/9*8, 
#                        'group '+str(count), fontsize=fontsize_val, color='b')
                plt.xlabel('Raman shift ($cm^{-1}$)', fontsize=fontsize_val)
                plt.ylabel('Intensity (a.u.)', fontsize=fontsize_val)
                plt.legend(['recovered spectra', 'reference spectra'], fontsize=fontsize_val, loc=1)
              
                plt.text(600, np.max([recovered[count-1, :], no_bg_mss[count-1, :]])/3, 'CNN', fontsize=fontsize_val)
                plt.text(600, -np.max([recovered[count-1, :], no_bg_mss[count-1, :]])/3,'AsLS', fontsize=fontsize_val)
                plt.xlim((np.min(wn), np.max(wn)))
                plt.grid()
                plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
                plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
                plt.setp(plt.gca().get_lines(), linewidth=2)
                count += 1
            
    plt.savefig('H:/spectra_data/3_compare.png', dpi=300)
    plt.show()
    
    plt.figure(figsize=(12, 12))
    
    plt.subplot(221)
    plt.plot(sub_real_coeffs_norm[:, 0], YPredict_norm[:, 0], 'ob',
             np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), 'g')
    plt.xlabel('True concentrations', fontsize=fontsize_val)
    plt.ylabel('Concentrations by CNN', fontsize=fontsize_val)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid()
    plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
    plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
    plt.setp(plt.gca().get_lines(), linewidth=3)
    plt.title(names[0], fontsize=fontsize_val)
    
    plt.subplot(222)
    plt.plot(sub_real_coeffs_norm[:, 1], YPredict_norm[:, 1], 'ob',
             np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), 'g')
#    plt.xlabel('true concentrations', fontsize=fontsize_val)
#    plt.ylabel('concentrations by CNN', fontsize=fontsize_val)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid()
    plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
    plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
    plt.setp(plt.gca().get_lines(), linewidth=3)
    plt.title(names[1], fontsize=fontsize_val)
    
    plt.subplot(223)
    plt.plot(sub_real_coeffs_norm[:, 2], YPredict_norm[:, 2], 'ob',
             np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), 'g')
    plt.xlabel('True concentrations', fontsize=fontsize_val)
    plt.ylabel('Concentrations by CNN  ', fontsize=fontsize_val)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid()
    plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
    plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
    plt.setp(plt.gca().get_lines(), linewidth=3)
    plt.title(names[2], fontsize=fontsize_val) 
    
    plt.savefig('H:/spectra_data/4_single_groups.png', dpi=300)
    plt.show()
    
    plt.figure(figsize=(12, 12))
    
    plt.subplot(221)
    plt.plot(sub_real_coeffs_norm[:, 0], ls_coeffs_norm[:, 0], 'ob',
             np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), 'g')
    plt.xlabel('True concentrations', fontsize=fontsize_val)
    plt.ylabel('Concentrations by AsLS  ', fontsize=fontsize_val)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid()
    plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
    plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
    plt.setp(plt.gca().get_lines(), linewidth=3)
    plt.title(names[0], fontsize=fontsize_val)
    
    plt.subplot(222)
    plt.plot(sub_real_coeffs_norm[:, 1], ls_coeffs_norm[:, 1], 'ob',
             np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), 'g')
    #plt.xlabel('True concentrations', fontsize=fontsize_val)
    #plt.ylabel('Concentrations by AsLS  ', fontsize=fontsize_val)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid()
    plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
    plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
    plt.setp(plt.gca().get_lines(), linewidth=3)
    plt.title(names[1], fontsize=fontsize_val)
    
    plt.subplot(223)
    plt.plot(sub_real_coeffs_norm[:, 2], ls_coeffs_norm[:, 2], 'ob',
             np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), 'g')
    plt.xlabel('True concentrations', fontsize=fontsize_val)
    plt.ylabel('Concentrations by AsLS  ', fontsize=fontsize_val)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid()
    plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
    plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
    plt.setp(plt.gca().get_lines(), linewidth=3)
    plt.title(names[2], fontsize=fontsize_val)
    
    plt.savefig('H:/real_data/spectra_data/4_single_groups1.png', dpi=300)
    plt.show()
    ########################################################################
    plt.figure(figsize=(12, 12))

    plt.subplot(221)
    plt.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), 'g', 
             sub_real_coeffs_norm[:, 0], YPredict_norm[:, 0], 'ob',
             sub_real_coeffs_norm[:, 0], ls_coeffs_norm[:, 0], '*r', markersize=10)
    plt.xlabel('AsLS Full Model', fontsize=fontsize_val)
    plt.ylabel(' Partial Model', fontsize=fontsize_val)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid()
    plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
    plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
    plt.setp(plt.gca().get_lines(), linewidth=3)
    plt.text(0.6, 0.04, 'DPPC deleted', fontsize=34)
    plt.title(names[0], fontsize=fontsize_val)

    plt.subplot(222)
    plt.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), 'g',
             sub_real_coeffs_norm[:, 1], YPredict_norm[:, 1], 'ob',
             sub_real_coeffs_norm[:, 1], ls_coeffs_norm[:, 1], '*r', markersize=10)
    #plt.xlabel('AsLS Full Model', fontsize=fontsize_val)
    #plt.ylabel(' Partial Model', fontsize=fontsize_val)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid()
    plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
    plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
    plt.setp(plt.gca().get_lines(), linewidth=3)
    plt.title(names[1], fontsize=fontsize_val)
    
    plt.subplot(223)
    plt.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), 'g',
             sub_real_coeffs_norm[:, 2], YPredict_norm[:, 2], 'ob',
             sub_real_coeffs_norm[:, 2], ls_coeffs_norm[:, 2], '*r', markersize=10)
    #plt.xlabel('AsLS Full Model', fontsize=fontsize_val)
    #plt.ylabel(' Partial Model', fontsize=fontsize_val)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid()
    plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
    plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
    plt.setp(plt.gca().get_lines(), linewidth=3)
    plt.title(names[2], fontsize=fontsize_val)
   # plt.subplot(223)
   # plt.plot(sub_real_coeffs_norm[:, 0], ls_coeffs_norm[:, 0], 'ob',
   #           np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), 'g')
   # plt.xlabel('True concentrations', fontsize=fontsize_val)
   # plt.ylabel('Concentrations by AsLS', fontsize=fontsize_val)
   # plt.xlim(0, 1)
   # plt.ylim(0, 1)
   # plt.grid()
   # plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
   # plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
   # plt.setp(plt.gca().get_lines(), linewidth=3)
    
   # plt.subplot(224)
   # plt.plot(sub_real_coeffs_norm[:, 1], ls_coeffs_norm[:, 1], 'ob',
   #           np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), 'g')
   # plt.xlabel('True concentrations', fontsize=fontsize_val)
   # plt.xlim(0, 1)
   # plt.ylim(0, 1)
   # plt.grid()
   # plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
   # plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
   #  plt.setp(plt.gca().get_lines(), linewidth=3)
    
    plt.savefig('H:/real_data/Lipo6_all_spectradata/4—1_single_group.png', dpi=300)
    plt.show()
        
        
else:
    fontsize_val = 12
    div_cols = len(exp)
    div_rows = np.int32(np.ceil(recovered.shape[0]/div_cols))
    plt.figure(figsize=(div_cols*10, div_rows*5))
    count = 1
    for r in range(div_rows): # group
        for c in range(div_cols):
            if count <= recovered.shape[0]:
                plt.subplot(div_rows, div_cols, count)
                plt.plot(wn, np.transpose(recovered[count-1, :]), 'r',
                         wn, np.transpose(no_bg_mss[count-1, :]), 'g',
                         wn, -np.transpose(ls_recovered[count-1, :]), 'r',
                         wn, -np.transpose(no_bg_mss[count-1, :]), 'g')
#                plt.title('group '+str(count), fontsize=12)
#                plt.text(290, np.max([recovered[count-1, :], no_bg_mss[count-1, :]])/9*8, 
#                         'group %d'%(r+1)+', exp='+str(exp[c])+'s', fontsize=fontsize_val, color='b')
                plt.xlabel('Raman shift ($cm^{-1}$)', fontsize=fontsize_val)
                plt.ylabel('Intensity (a.u.)', fontsize=fontsize_val)
                plt.legend(['recovered spectra', 'reference spectra'], fontsize=fontsize_val, loc=1)
#                plt.text(290, np.max([recovered[count-1, :], no_bg_mss[count-1, :]])/3, 'CNN, absolute mse=%.2f'%cnn_mse[count-1], fontsize=fontsize_val)
#                plt.text(290, -np.max([recovered[count-1, :], no_bg_mss[count-1, :]])/3,'AsLS, absolute mse=%.2f'%ls_mse[count-1], fontsize=fontsize_val)
                plt.xlim((np.min(wn), np.max(wn)))
                plt.grid()
                plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
                plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val) 
                plt.setp(plt.gca().get_lines(), linewidth=3)
                count += 1
            
    plt.savefig('H:/real_data/Lipo6_all_spectradata/5_all_recovered.png', dpi=300)
    plt.show()
    ############################################################################
    
    ############################################################################
# In[] print time cost
print('total time cost of cnn is', (cnn_normalize+cnn_predict), 's')
print('total time cost of asls is', (asls_smooth+asls_bg_cor+asls_predict), 's')
print('mean minima is', np.min(np.mean(mixture, axis=1)))






