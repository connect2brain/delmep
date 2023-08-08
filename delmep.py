#!/usr/bin/env python
# -*- coding: utf-8 -*-

# DELMEP - A deep learning algorithm for automated annotation of motor evoked potential latencies
#
# This file is part of delmep package which is released under copyright.
# See file LICENSE or go to website for full license details.
# Copyright (C) 2018 Victor Hugo Souza - All Rights Reserved
#
# Homepage: https://github.com/connect2brain/delmep
# License: GNU General Public License v3.0
#
# Authors: Diego Milardovich (@dmilardovich) and Victor Hugo Souza (@vhosouza)
# Date: 10/05/2023

from io import BytesIO
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
import requests
from scipy import signal
import tensorflow as tf

# Building the NN ---
NN = Sequential()
NN.add(Dense(30, input_dim=120, activation='relu'))
NN.add(Dense(30, activation='relu'))
NN.add(Dense(1, activation='relu'))

mse = tf.keras.losses.MeanSquaredError()
NN.compile(loss=mse, optimizer='adam')
NN.load_weights('NN_main_new')
# ---

# %%
def compute_latency_delmep(mep, window_length):
    """

    :param mep: 120-samples MEP trace, trimmed to 10-50 after the TMS stimulation.
    :param window_length: windows length for the filtering
    :return: array of annotated latencies
    """
    
    ###############################################################################
    # Pre-processing the MEPs
    ###############################################################################

    # The MEPs are already trimmed, in the following steps they will also be 
    # filtered, smoothed, centered and amplitude normalized. 
    
    ###############################################################################
    # Filtering the MEPs
    ###############################################################################
    
    # The MEPs are smoothed with a moving average filter with a window length of 3 
    # samples, to reduce the high-frequency noise of the recordings.
    
    meps_filtered_test = []
    
    ###############################################################################
    # Filtering the testing dataset MEPs
    ###############################################################################
    
    # Filtering the initial value(s) of the MEPs
    new_mep_filtered_test = np.ones(len(mep))
    for j in range(0, window_length):
        filtered_value = 0
        length = 0
        for k in range(0, window_length + 1 + j):
            length = length + 1
            filtered_value = filtered_value + mep[k]
        filtered_value = filtered_value/length
        new_mep_filtered_test[j] = filtered_value

    # Fintering the central values of the MEPs
    for j in range(window_length, len(mep) - window_length):
        filtered_value = 0
        length = 0
        for k in range(j - window_length, j + window_length + 1):
            length = length + 1
            filtered_value = filtered_value + mep[k]
        filtered_value = filtered_value/length
        new_mep_filtered_test[j] = filtered_value

    # Filtering the final value(s) of the MEPs
    for j in range(len(mep) - window_length, len(mep)):
        filtered_value = 0
        length = 0
        for k in range(-window_length - 1, 0):
            length = length + 1
            filtered_value = filtered_value + mep[k]
        filtered_value
        filtered_value = filtered_value/length
        new_mep_filtered_test[j] = filtered_value
    
    meps_filtered_test.append(new_mep_filtered_test)

    ###############################################################################
    # Centering the MEPs
    ###############################################################################
    
    # The MEPs are centered by computing their mean value in the first 15 ms after 
    # the stimulation and then subtracting the mean value from every value. This 
    # step reduces the impact of low frequency noise in the measurements, by 
    # counteracting the shifting it produces in the mean mep value.
    
    for j in range(len(meps_filtered_test)):
        meps_filtered_test[j] = meps_filtered_test[j] - np.mean(meps_filtered_test[j][0:14])

    ###############################################################################
    # Normalizing the MEPs
    ###############################################################################
    
    # The mep values are normalized within 0 to 1, to mitigate the effects of the
    # large variations in amplitude.
    
    for j in range(len(meps_filtered_test)):
        meps_filtered_test[j] = meps_filtered_test[j]/(max(abs(meps_filtered_test[j])))

    ###############################################################################
    # Absolute value of MEPs
    ###############################################################################
    
    # Every sample of the MEPs is replaced by its absolute value. 
    
    for j in range(len(meps_filtered_test)):
        meps_filtered_test[j] = abs(meps_filtered_test[j])

    ###############################################################################
    # Using the main NN to predict the latencies of the MEPs
    ###############################################################################
    
    # Converting the MEPs and latencies to arrays
    meps_filtered_test = np.array(meps_filtered_test)
    
    # Using the main NN to compute the latencies of the MEPs in the training and 
    # testing datasets.
    latency = NN.predict(meps_filtered_test)

    return (latency)

# %%
###############################################################################
# Retrieve the first five MEPs from Bigoni's open-access dataset 
###############################################################################

# URL of the raw data file
url = "https://raw.githubusercontent.com/clovbig/MEP_latency/main/data/subject_1/EMG_data_SP.npy"

# Send a GET request to retrieve the content
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Load the content as a NumPy array
    data = np.load(BytesIO(response.content), allow_pickle=True).T
    epochs_emg = data[0:5]
else:
    raise RuntimeError("Failed to retrieve the data. Status code: {}".format(response.status_code))


###############################################################################
# Trimming the MEP
###############################################################################

# MEP metadata
fs = 5000            # Hz
trigger_time = 1000  # ms
start_time = 10      # ms
end_time = 50        # ms
    
# Convert the initial and final times into index values
start_index = int((trigger_time / 1000) * fs + (start_time / 1000) * fs)
end_index = int((trigger_time / 1000) * fs + (end_time / 1000) * fs)


for j in range(len(epochs_emg)):
    
    # Cut the MEP to 10 - 50 ms
    mep_data = epochs_emg[j][start_index:end_index]
        
    # Subsample the MEP to 3000 Hz (i.e., 120 samples between 10-50 ms) to be
    # compatible with DELMEP's input.
    mep_subsampled = signal.resample(mep_data, 120)

    # Assess the MEP latency with DELMEP
    # Compute the latency with DELMEP
    latency_delmep = compute_latency_delmep(mep_subsampled, 3)
    
    # Plot the MEP and DELMEP latency
    # Convert the index numbers into time, for DELMEP MEP
    x_time = np.linspace(start=start_time, stop=end_time, num=len(mep_subsampled))
    
    # create figure and plot data
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.plot(x_time, mep_subsampled)
    ax.axvline(x=latency_delmep[0], color='maroon', label='DELMEP latency', linewidth=3)

    # customize axes and add legend
    ax.set_xlabel('Time (ms)', fontsize=11)
    ax.set_ylabel('EMG (mV)', fontsize=11)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend(fontsize=8, loc='lower right')

    # apply tight layout and show figure
    fig.tight_layout()

    # save plots
    # fig.savefig('MEP_'+str(j)+'.png')

    plt.show()
