# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 15:56:31 2021

@author: admin
"""

import numpy as np
import matplotlib.pyplot as plt
import spectra_process.subpys as subpys
import time



################################################################################
# define function for creating mixed spectra
def spectra_generator(pures, noise_level, pow_val=5, poly_enhance=1.0):
    ################################################################
    # generate random spectra coefficients, range from 0 to 1
    rand_coeffs = np.random.rand(1, pures.shape[0])*poly_enhance
    # create simulated mixed spectra
    f = np.matmul(rand_coeffs, pures)
    ###############################################################
    if pow_val != 0:
        # randomize polynomial order
        index = np.random.permutation(np.arange(pow_val))
        # create background
        base = subpys.myploy(index[0]+1, pures.shape[1])
        base = base*poly_enhance
        for k in range(base.shape[0]):
            if np.random.randn(1) <= 0:
                base[k, :] = np.flip(base[k, :])
        ################################
        # create baseline
        base_coeffs = np.random.rand(1, base.shape[0])*2 - 1
        baseline = np.matmul(base_coeffs, base)
        ################################
        #base_concat = np.concatenate((pures, baseline), axis=0)
        ################################
        f = f + baseline       
    ###############################################################   
    # add noise
    qtz_bg = np.mean(pures[-1, :])*(1 - rand_coeffs[0, -1])
    nosie = noise_level*qtz_bg*np.random.randn(1, pures.shape[1])
    ################################
    # f = f - np.min(f, axis=1)
    mixed_spectra = f + nosie
    mixed_spectra = mixed_spectra - np.min(mixed_spectra, axis=1)
    ################################
    coeff = rand_coeffs[0, :(pures.shape[0])]
#    coeff = coeff/np.sum(coeff) # normalization
    
    return coeff, mixed_spectra
################################################################################
def main(unused_argv):
    # load pure spectra
    data = np.load('G:/spectra_data/my_measurement_tmp1.npz')
    CL = data['CL']*5000
    CytoC = data['CytoC']*5000
    DNA = data['DNA']*5000
    qtz_ss = data['qtz_ss']/4
    wn = data['wn']
    names = data['pure_names']
    
    pures = np.reshape(np.concatenate((CL, CytoC,DNA, qtz_ss)), (4, wn.shape[0]))
    #pures = pures[[0, 1], :] # binary model
    pures = pures[[ 0,1,2,3], :] # ternary model
    ############################################################################
    random_index = np.int(1e6)
    poly_enhance = 1.0 # 10, 5
    noise_level = 0.02# 0.01, 0.02
    pow_val = 6 # 6, 2
    # pures = pures/10
    ############################################################################
    fontsize_val = 24
    plt.figure(figsize=(10, 7))
    plt.plot(wn, np.transpose(pures[:, :]))
    # plt.title('pure spectra', fontsize=fontsize_val)
    plt.xlabel('Raman shift ($cm^{-1}$)', fontsize=fontsize_val)
    plt.ylabel('Intensity (a.u.)', fontsize=fontsize_val)
    plt.xlim((np.min(wn), np.max(wn)))
    plt.grid()
    plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
    plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
    plt.setp(plt.gca().get_lines(), linewidth=4)
    plt.ylim((np.min(pures), np.max(pures)))
    plt.legend(names[:pures.shape[0]], loc=1, fontsize=fontsize_val)
    plt.savefig('G:/output/generated/pure_spectra.png', dpi=300)
    plt.show()
    
    plt.figure(figsize=(10, 7))
    if noise_level == 0:
        plt.plot(wn, np.transpose(subpys.myploy(pow_val, pures.shape[1])*poly_enhance))
    else:
        plt.plot(wn, np.transpose(subpys.myploy(pow_val, pures.shape[1])*poly_enhance))
    # plt.title('polynomial backgrounds', fontsize=fontsize_val)
    plt.xlabel('Raman shift ($cm^{-1}$)', fontsize=fontsize_val)
    plt.ylabel('Intensity (a.u.)', fontsize=fontsize_val)
    plt.xlim((np.min(wn), np.max(wn)))
    plt.grid()
    plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
    plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
    plt.setp(plt.gca().get_lines(), linewidth=4)
    plt.savefig('G:/output/generated/polynomial_background.png', dpi=300)
    plt.show()
    ############################################################################
    coeff = np.zeros((random_index, pures.shape[0]))
    mixed_spectra = np.zeros((pures.shape[1], random_index))
    # create mixed spectra
    time_start = time.time()
    for ij in range(random_index):
        coeff[ij, :], mixed_spectra[:, ij] = spectra_generator(pures, noise_level, pow_val, poly_enhance)
    time_end = time.time()
    ############################################################################
    mixture_minima = np.min(np.mean(mixed_spectra, axis=0))
#    mixed_spectra = mixed_spectra - np.min(mixed_spectra)
    ############################################################################
    plt.figure(figsize=(10, 7))
    plt.plot(np.transpose(wn), mixed_spectra[:, :100])
    # plt.title('simulated mixed spectra', fontsize=fontsize_val)
    plt.xlabel('Raman shift ($cm^{-1}$)', fontsize=fontsize_val)
    plt.ylabel('Intensity (a.u.)', fontsize=fontsize_val)
    plt.xlim((np.min(wn), np.max(wn)))
    #plt.ylim((0, 100))
    plt.grid()
    plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
    plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
    plt.setp(plt.gca().get_lines(), linewidth=0.5)
    plt.savefig('G:/output/generated/generated_mixed_spectra.png', dpi=300)
    plt.show()
    
    np.save('G:/spectra_data/ternary/X_train.npy', np.transpose(mixed_spectra))
    np.save('G:/spectra_data/ternary/Y_train.npy', coeff[:, :(pures.shape[0]-1)])
    print('totally cost:', time_end-time_start, 's')
    print('minimized mean of mixture:', mixture_minima)
    print('minima of mixture:', np.min(mixed_spectra))
    print('maxima of mixture:', np.max(mixed_spectra))
    print('minima of peak height:', np.min(np.max(mixed_spectra, axis=0) - np.min(mixed_spectra, axis=0)))
    print('noise base:', np.mean(pures[-1, :])*noise_level*poly_enhance)

if __name__ == "__main__":
    main(0)