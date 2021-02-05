# -*- coding: utf-8 -*-
"""
Created on Fri Feb 05 2021

Module of constants

Contributors:   Amorim, FE
                Chapot, RL
                Moulin, TC
                Lee, J
                Amaral, OB

File description:
Model based on BiorXiv (Auth et al., 2018) later published in Frontiers (Auth et al., 2020).
"""

import math
import numpy as np

## PARAMETERS

Session_qt = 10;                # Session quantity
Session_qt_M1 = 10;                # Session quantity
Session_qt_M2 = 10;                # Session quantity

Learning_time = 5;              # Learning time in seconds; Original = 5;
Rest_time = 1;                  # Rest time in seconds; Original = 1;
Test_time = 0.5;                # Test time in seconds; Original = 0.5;

neurons_input = 36;
neuron_memory = 900;
neuron_lin_col_memory = int(math.sqrt(neuron_memory));
neuron_lin_col_input = int(math.sqrt(neurons_input));

tau_time = 0.01;                # membrane time constant (memory area)
R_memory = 1/11;                # membrane resistance (memory area); Original = 1/11
Ik_active = 130;                # Input rate (learning)
Ik_inactive = 0;                # Input rate (without learning)
Alpha_rate = 100;               # maximum firing rate
Beta_steepness = 0.05;          # sigmoid steepness
Epsilon_inflexion = 130;        # sigmoid inflexion point
mu_time = 1/15;                 # plasticity time constant
Ft_rate = 0.1;                  # target firing rate
FR_inh = np.zeros((1,1));       # Initial Firing rate (Global inhibitory unit)
Krec = 60;                      # scaling time constant (recurrent); Original = 60
Kff = 720;                      # scaling time constant (feed-forward); Original = 720
Krec_DESTAB = 60;               # scaling time constant (recurrent); Original = 60
Kff_DESTAB = 720;              # scaling time constant (feed-forward); Original = 720
tau_inh = 0.02;                 # membrane time constant (inhibitory unit)
R_inh = 1;                      # membrane resistance (inhibitory unit)
W_inh_i = np.ones((1,900));     # synaptic weight to inhibitory unit
W_inh_i = W_inh_i * 0.6;        # Original value = W_inh_i * 0.6
W_i_inh = 6000;                 # synaptic weight from inhibitory unit; Original = 1200;
W_i_inh = np.ones((neuron_memory, 1)) * W_i_inh;
dt = 0.005;                     # Step size; Original: 0.005
Ik_test = 130;                  # Input rate (Test)
            
radius_n = 4;                   # number of neurons to connect
ff_connections = 4;             # number of connections between a memory neuron and n neurons of input area



## INPUTS
# Memory 1

Input_S1_learn = np.zeros((neurons_input, 1));       # Input 1-Paper S1
Input_S1_test = np.zeros((neurons_input, 1));        # Input test - Paper S1(whole s1)
Input_S1_test_r6 = np.zeros((neurons_input, 1));     # Input test S1 - S1 Retrieval with only 6 neurons 
Input_S1_test_r8 = np.zeros((neurons_input, 1));     # Input test S1 - S1 Retrieval with only 8 neurons
Input_S1_test_r10 = np.zeros((neurons_input, 1));    # Input test S1 - S1 Retrieval with only 10 neurons
Input_S1_test_r12 = np.zeros((neurons_input, 1));    # Input test S1 - S1 Retrieval with only 12 neurons
Input_M1_learn_size12 = np.zeros((neurons_input, 1));# Input M1_size_12 - S1 learning with 12 active input neurons


# Memory 2
Input_S2_learn = np.zeros((neurons_input, 1));       # Input 2-Paper s2
Input_S2_learn_ov2 = np.zeros((neurons_input, 1));   # Input  - S2 with 4 overlap neurons
Input_S2_learn_ov4 = np.zeros((neurons_input, 1));   # Input  - S2 with 4 overlap neurons
Input_S2_learn_ov6 = np.zeros((neurons_input, 1));   # Input  - S2 with 6 overlap neurons
Input_S2_learn_ov8 = np.zeros((neurons_input, 1));   # Input  - S2 with 8 overlap neurons
Input_S2_learn_ov10 = np.zeros((neurons_input, 1));  # Input  - S2 with 10 overlap neurons
Input_S2_learn_ov12 = np.zeros((neurons_input, 1));  # Input  - S2 with 12 overlap neurons
Input_S2_learn_ov14 = np.zeros((neurons_input, 1));  # Input  - S2 with 14 overlap neurons
Input_S2_learn_ov16 = np.zeros((neurons_input, 1));  # Input  - S2 with 16 overlap neurons
Input_M2_learn_size12 = np.zeros((neurons_input, 1));# Input M2_size_12 - S2 learning with 12 active input neurons
Input_S2_test = np.zeros((neurons_input, 1));        # Input test - Paper S2(whole s2)
Input_S2_test_r2 = np.zeros((neurons_input, 1));     # Input test S2 - S2 Retrieval with only 2 neurons
Input_S2_test_r4 = np.zeros((neurons_input, 1));     # Input test S2 - S2 Retrieval with only 4 neurons
Input_S2_test_r6 = np.zeros((neurons_input, 1));     # Input test S2 - S2 Retrieval with only 6 neurons
Input_S2_test_r8 = np.zeros((neurons_input, 1));     # Input test S2 - S2 Retrieval with only 8 neurons
Input_S2_test_r10 = np.zeros((neurons_input, 1));    # Input test S2 - S2 Retrieval with only 10 neurons
Input_S2_test_r12 = np.zeros((neurons_input, 1));    # Input test S2 - S2 Retrieval with only 12 neurons
Input_S2_test_r14 = np.zeros((neurons_input, 1));    # Input test S2 - S2 Retrieval with only 14 neurons
Input_S2_test_r16 = np.zeros((neurons_input, 1));    # Input test S2 - S2 Retrieval with only 16 neurons

Input_M2_M1_test_ov4 = np.zeros((neurons_input, 1)); # Input test - Memory S2 with 4 neurons overlapping with S1
Input_M2_M1_test_ov6 = np.zeros((neurons_input, 1)); # Input test - Memory S2 with 6 neurons overlapping with S1
Input_M2_M1_test_ov8 = np.zeros((neurons_input, 1)); # Input test - Memory S2 with 8 neurons overlapping with S1
Input_M2_M1_test_ov10 = np.zeros((neurons_input, 1));# Input test - Memory S2 with 10 neurons overlapping with S1
Input_M2_M1_test_ov12 = np.zeros((neurons_input, 1));# Input test - Memory S2 with 12 neurons overlapping with S1

# OVERLAPS ALONE
Input_M1_test_ov0 = np.zeros((neurons_input, 1));    # Input test - Overlap 0 neurons
Input_M1_test_ov2 = np.zeros((neurons_input, 1));    # Input test - Overlap 2 neurons
Input_M1_test_ov4 = np.zeros((neurons_input, 1));    # Input test - Overlap 4 neurons
Input_M1_test_ov6 = np.zeros((neurons_input, 1));    # Input test - Overlap 6 neurons
Input_M1_test_ov8 = np.zeros((neurons_input, 1));    # Input test - Overlap 8 neurons
Input_M1_test_ov9 = np.zeros((neurons_input, 1));    # Input test - Overlap 9 neurons
Input_M1_test_ov9_type_2 = np.zeros((neurons_input, 1));    # Input test - Overlap 9 neurons - last ones
Input_M1_test_ov10 = np.zeros((neurons_input, 1));   # Input test - Overlap 10 neurons
Input_M1_test_ov12 = np.zeros((neurons_input, 1));   # Input test - Overlap 12 neurons
Input_M1_test_ov14 = np.zeros((neurons_input, 1));   # Input test - Overlap 14 neurons
Input_M1_test_ov16 = np.zeros((neurons_input, 1));   # Input test - Overlap 16 neurons

Input_one = np.zeros((neurons_input, 1));            # Input 30 - Testing 1 Input_S2 = np.zeros((neurons_inpneuron
Input_rest=np.zeros((neurons_input,1));              # Input Rest


# Active neurons receives an input of 130 hz
for jj in range (neurons_input):
    # INPUTS FOR MEMORY S1
    # Input 1  is one half of input area neurons with 130 hz and the other half remains inactive at 0 Hz
    #
    if jj <= (neurons_input/2)-1:
        Input_S1_learn[jj][0] = Ik_active;
    
    # Input M1_size_12 is 12 input area neurons
    if jj <= (neurons_input/3)-1:
        Input_M1_learn_size12[jj][0] = Ik_active;
    
    # Input_S1_Qt_6 is memory 1 with only 6 neurons
    if jj <= 17 and jj >= 12:
        Input_S1_test_r6[jj][0] = Ik_test;
    
    # Input_S1_Qt_8 is memory 1 with only 8 neurons
    if jj <= 17 and jj >= 10:
        Input_S1_test_r8[jj][0] = Ik_test;
    
    # Input_S1_Qt_10 is memory 1 with only 10 neurons
    if jj <= 17 and jj >= 8:
        Input_S1_test_r10[jj][0] = Ik_test;
        
    # Input_S1_Qt_12 is memory 1 with only 12 neurons
    if jj <= 17 and jj >= 6:
        Input_S1_test_r12[jj][0] = Ik_test;
    
    # Input_test_1 is the whole memory S1 (Similar to paper)
    if jj <= (neurons_input/2)-1:
        Input_S1_test[jj][0] = Ik_test;
        
        
    # INPUTS FOR MEMORY S2
    # Input 2 is other half of input area neurons
    if jj > (neurons_input/2)-1:
        Input_S2_learn[jj][0] = Ik_active;
    
    # Input M2_size_12 is 12 input area neurons
    if jj > (neurons_input/2)-1+6:
        Input_M2_learn_size12[jj][0] = Ik_active;
    
    # Input_test_2 is the whole memory S2 (Similar to paper)
    if jj > (neurons_input/2)-1:
        Input_S2_test[jj][0] = Ik_test;
    
    # Input 6 is a memory with 2 neurons overlapped with memory 1
    if jj <= 1 or jj >= 20:
        Input_S2_learn_ov2[jj][0] = Ik_active;
        
    # Input 6 is a memory with 4 neurons overlapped with memory 1
    if jj <= 3 or jj >= 22:
        Input_S2_learn_ov4[jj][0] = Ik_active;
    
    # Input 7 is a memory with 6 neurons overlapped with memory 1
    if jj <= 5 or jj >= 24:
        Input_S2_learn_ov6[jj][0] = Ik_active;
        
    # Input 8 is a memory with 8 neurons overlapped with memory 1
    if jj <= 7 or jj >= 26:
         Input_S2_learn_ov8[jj][0] = Ik_active;
        
    # Input 9 is a memory with 10 neurons overlapped with memory 1
    if jj <= 9 or jj >= 28:
        Input_S2_learn_ov10[jj][0] = Ik_active;
        
    # Input 9 is a memory with 12 neurons overlapped with memory 1
    if jj <= 11 or jj >= 30:
        Input_S2_learn_ov12[jj][0] = Ik_active;
    
    # Input 9 is a memory with 14 neurons overlapped with memory 1
    if jj <= 13 or jj >= 32:
        Input_S2_learn_ov14[jj][0] = Ik_active;
        
    # Input 9 is a memory with 16 neurons overlapped with memory 1
    if jj <= 15 or jj >= 34:
        Input_S2_learn_ov16[jj][0] = Ik_active;
    
    
    # Input_S2_Qt_6 is memory 1 with only 6 neurons
    if jj <= 23 and jj >= 18:
        Input_S2_test_r6[jj][0] = Ik_test;
    
    # Input_S2_Qt_8 is memory 1 with only 8 neurons
    if jj <= 25 and jj >= 18:
        Input_S2_test_r8[jj][0] = Ik_test;
    
    # Input_S2_Qt_10 is memory 1 with only 10 neurons
    if jj <= 27 and jj >= 18:
        Input_S2_test_r10[jj][0] = Ik_test;
    
    # Input_S2_Qt_12 is memory 1 with only 12 neurons
    if jj <= 29 and jj >= 18:
        Input_S2_test_r12[jj][0] = Ik_test;
    
    
    # INPUTS FOR OVERLAPS OR OTHERS
    # Input one is a test input with only one neuron active
    Input_one[3][0] = Ik_active;
            
    # Input_test_ov4 is a input test with 4 neurons overlapped in memory 1
    if jj <= 1:
        Input_M1_test_ov2[jj][0] = Ik_test;
    
    # Input_test_M2_ov4 is memory 2 with 4 neurons overlapped in memory 1
    if jj <= 3 or jj >= 22:
       Input_M2_M1_test_ov4[jj][0] = Ik_test;
        
    # Input_test_ov4 is a input test with 4 neurons overlapped in memory 1
    if jj <= 3:
        Input_M1_test_ov4[jj][0] = Ik_test;
        
    # Input_test_M2_ov6 is memory 2 with 6 neurons overlapped in memory 1
    if jj <= 5 or jj >= 24:
        Input_M2_M1_test_ov6[jj][0] = Ik_test;
        
    # Input_test_ov6 is a input test with 6 neurons overlapped in memory 1
    if jj <= 5:
        Input_M1_test_ov6[jj][0] = Ik_test;
        
    # Input_test_M2_ov8 is memory 2 with 8 neurons overlapped in memory 1
    if jj <= 7 or jj >= 26:
        Input_M2_M1_test_ov8[jj][0] = Ik_test;
        
    # Input_test_ov8 is a input test with 8 neurons overlapped in memory 1
    if jj <= 7:
        Input_M1_test_ov8[jj][0] = Ik_test;
        
    # Input_test_ov9 is a input test with 8 neurons overlapped in memory 1
    if jj <= 8:
        Input_M1_test_ov9[jj][0] = Ik_test;
    
    # Input_test_ov9 is a input test with 8 neurons overlapped in memory 1
    if jj <= 17 and jj > 8:
        Input_M1_test_ov9_type_2[jj][0] = Ik_test;
    
    # Input_test_M2_ov10 is memory 2 with 10 neurons overlapped in memory 1
    if jj <= 9 or jj >= 28:
        Input_M2_M1_test_ov10[jj][0] = Ik_test;
        
    # Input_test_ov10 is a input test with 10 neurons overlapped in memory 1
    if jj <= 9:
        Input_M1_test_ov10[jj][0] = Ik_test;
        
    # Input_test_M2_ov2 is memory 2 with 12 neurons overlapped in memory 1
    if jj <= 11 or jj >= 30:
        Input_M2_M1_test_ov12[jj][0] = Ik_test;
        
    # Input_test_ov12 is a input test with 12 neurons overlapped in memory 1
    if jj <= 11:
        Input_M1_test_ov12[jj][0] = Ik_test;
        
    # Input_test_ov12 is a input test with 12 neurons overlapped in memory 1
    if jj <= 13:
        Input_M1_test_ov14[jj][0] = Ik_test;
        
    # Input_test_ov12 is a input test with 12 neurons overlapped in memory 1
    if jj <= 15:
        Input_M1_test_ov16[jj][0] = Ik_test;