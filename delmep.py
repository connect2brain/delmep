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

import numpy as np
from keras.models import Sequential
import tensorflow as tf
from keras.layers import Dense

# Building the main NN ---
NN_main = Sequential()
NN_main.add(Dense(30, input_dim=120, activation='relu'))
NN_main.add(Dense(30, activation='relu'))
NN_main.add(Dense(1, activation='relu'))

mse = tf.keras.losses.MeanSquaredError()
NN_main.compile(loss=mse, optimizer='adam')
NN_main.load_weights('NN_main_new')
# ---


def delmep_latency(mep, window_length):
    """

    :param mep: 120-samples MEP trace, trimmed to 10-50 after the TMS stimulation.
    :param window_length: windows length for the filtering
    :return: array of annotated latencies
    """
    
    meps_raw_test_ok = mep.copy()
    
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
    new_mep_filtered_test = np.ones(len(meps_raw_test_ok))
    for j in range(0, window_length):
        filtered_value = 0
        length = 0
        for k in range(0, window_length + 1 + j):
            length = length + 1
            filtered_value = filtered_value + meps_raw_test_ok[k]
        filtered_value = filtered_value/length
        new_mep_filtered_test[j] = filtered_value

    # Fintering the central values of the MEPs
    for j in range(window_length, len(meps_raw_test_ok) - window_length):
        filtered_value = 0
        length = 0
        for k in range(j - window_length, j + window_length + 1):
            length = length + 1
            filtered_value = filtered_value + meps_raw_test_ok[k]
        filtered_value = filtered_value/length
        new_mep_filtered_test[j] = filtered_value

    # Filtering the final value(s) of the MEPs
    for j in range(len(meps_raw_test_ok) - window_length, len(meps_raw_test_ok)):
        filtered_value = 0
        length = 0
        for k in range(-window_length - 1, 0):
            length = length + 1
            filtered_value = filtered_value + meps_raw_test_ok[k]
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
    no_corr_prediction_test = NN_main.predict(meps_filtered_test)

    return (no_corr_prediction_test[0])
