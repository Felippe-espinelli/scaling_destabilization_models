# -*- coding: utf-8 -*-
"""
Created on Fri Feb 05 2021

Module of essential functions described in the paper

Contributors:   Amorim, FE
                Chapot, RL
                Moulin, TC
                Lee, J
                Amaral, OB

File description:
Model based on BiorXiv (Auth et al., 2018) later published in Frontiers (Auth et al., 2020).
"""

###############################################################################
# IMPORT
###############################################################################
import numpy as np
import model_constants_v2 as CONST
import math
import random

###############################################################################
# FUNCTIONS
###############################################################################

#################### MODEL FUNCTIONS ####################
# Equation 1 - Membrane potential of each neuron i in the memory area
def membrane_potential(U_exc, tau, R_memory, W_rec, dt, W_i_inh, W_ff, F, F_inh, F_Input):
#    # Excitatory membrane potential (recurrent)
#    dU_exc_rec = (R_memory * (W_rec @ F));
#    # Excitatory membrane potential (feed forward)
#    dU_exc_ff = (R_memory * (W_ff @ F_Input));
#    # Inhibitory influence
#    dU_inh = R_memory * (W_i_inh @ F_inh);
#    # Update
#    U_exc = U_exc + (-U_exc/tau + dU_exc_rec - dU_inh + dU_exc_ff) * dt;
    U_exc = U_exc + (-U_exc/tau + (R_memory * (W_rec @ F)) - R_memory * (W_i_inh @ F_inh) + (R_memory * (W_ff @ F_Input))) * dt;
    return U_exc

# Equation 2 - Firing rate and Eq. 4 for inhibitory
def firing_rate(Alpha_rate, Beta_steepness, Epsilon_inflexion, U):
    F_i = Alpha_rate/(1+ np.exp(Beta_steepness*(Epsilon_inflexion - U)));
    return F_i

# Equation 3 - Membrane potential of the global inhibitory unit
def membrane_potential_inh(U_inh, tau_inh, R_inh, W_inh_i, F, dt):
#    # Inhibitory activity
#    dU_inh = (-U_inh/tau_inh + R_inh * (W_inh_i @ F)) * dt;
#    # Update
#    U_inh = U_inh + dU_inh;
    U_inh = U_inh + ((-U_inh/tau_inh + R_inh * (W_inh_i @ F)) * dt);
    return U_inh

# Equation 5 - Weight changes (feed-forward connections)
def weight_change_ff(mu_time,F_i, Input, Kff, Ft_rate, weight, connections, dt):

#    # Homeostatic plasticity
#    HSP = (1/Kff) * (Ft_rate*np.ones((CONST.neuron_memory,1)) - F_i);
#    HSP = HSP * (weight ** 2);
#    
#    # Hebbian plasticity
#    HLP = F_i @ Input;
#    
#    # Weight combination and connections
#    dW = mu_time * (HLP + HSP) * connections;
    dW = mu_time * ((F_i @ Input) + (((1/Kff) * (Ft_rate*np.ones((CONST.neuron_memory,1)) - F_i)) * (weight ** 2))) * connections;
    # Total weight changes
    weight = weight + (dW * dt);
        
    return weight

def weight_change_ff_ANISO(mu_time,F_i, Input, Kff, Ft_rate, weight, connections, dt, Aniso_value):

#    # Homeostatic plasticity
#    HSP = (1/Kff) * (Ft_rate*np.ones((CONST.neuron_memory,1)) - F_i);
#    HSP = HSP * (weight ** 2);
#    
#    # Hebbian plasticity
#    HLP = 0;
#    
#    # Weight combination and connections
#    dW = mu_time * (HLP + HSP) * connections;
    
    dW = mu_time * (((F_i @ Input))*Aniso_value + ((1/Kff) * (Ft_rate*np.ones((CONST.neuron_memory,1)) - F_i)) * (weight ** 2)) * connections;
    
    # Total weight changes
    weight = weight + (dW * dt);
        
    return weight

def weight_change_ff_DESTAB(mu_time,F_i, Input, Kff_DESTAB, Ft_rate, weight, connections, dt):

#    # Homeostatic plasticity
#    HSP = (1/Kff) * (Ft_rate*np.ones((CONST.neuron_memory,1)) - F_i);
#    HSP = HSP * (weight ** 2);
#    
#    # Hebbian plasticity
#    HLP = 0;
#    
#    # Weight combination and connections
#    dW = mu_time * (HLP + HSP) * connections;
    
    dW = mu_time * (((F_i @ Input)) + ((1/Kff_DESTAB) * (Ft_rate*np.ones((CONST.neuron_memory,1)) - F_i)) * (weight ** 2)) * connections;
    
    # Total weight changes
    weight = weight + (dW * dt);
        
    return weight

# Equation 6 - Weight changes (recurrent connections)
def weight_change_rec(mu_time,F_i, F_j, Krec, Ft_rate, weight, connections, dt):

#    # Homeostatic plasticity
#    HSP = (1/Krec) * (Ft_rate*np.ones((CONST.neuron_memory,1)) - F_i);
#    HSP = HSP * (weight ** 2);
#    
#    # Hebbian plasticity
#    HLP = F_i @ F_j;
#    
#    # Weight combination and connections
#    dW = mu_time * (HLP + HSP) * connections;
    dW = mu_time * ((F_i @ F_j) + (((1/Krec) * (Ft_rate*np.ones((CONST.neuron_memory,1)) - F_i)) * (weight ** 2))) * connections;
    # Total weight changes
    weight = weight + (dW * dt);
        
    return weight

# Weight changes (recurrent connections - Anisomycin)
def weight_change_rec_ANISO(mu_time,F_i, F_j, Krec, Ft_rate, weight, connections, dt, Aniso_value):

#    # Homeostatic plasticity
#    HSP = (1/Krec) * (Ft_rate*np.ones((CONST.neuron_memory,1)) - F_i);
#    HSP = HSP * (weight ** 2);
#    
#    # Hebbian plasticity
#    HLP = 0;
#    
#    # Weight combination and connections
#    dW = mu_time * (HLP + HSP) * connections;
    dW = mu_time * ((F_i @ F_j)*Aniso_value + (((1/Krec) * (Ft_rate*np.ones((CONST.neuron_memory,1)) - F_i)) * (weight ** 2))) * connections;
    # Total weight changes
    weight = weight + (dW * dt);
        
    return weight

# Weight changes (recurrent connections - Destabilization modulation)
def weight_change_rec_DESTAB(mu_time,F_i, F_j, Krec_DESTAB, Ft_rate, weight, connections, dt):

#    # Homeostatic plasticity
#    HSP = (1/Krec) * (Ft_rate*np.ones((CONST.neuron_memory,1)) - F_i);
#    HSP = HSP * (weight ** 2);
#    
#    # Hebbian plasticity
#    HLP = 0;
#    
#    # Weight combination and connections
#    dW = mu_time * (HLP + HSP) * connections;
    dW = mu_time * ((F_i @ F_j) + (((1/Krec_DESTAB) * (Ft_rate*np.ones((CONST.neuron_memory,1)) - F_i)) * (weight ** 2))) * connections;
    # Total weight changes
    weight = weight + (dW * dt);
        
    return weight


# Calculate distance for grid connectivity
def calculateDistance(x1,y1,x2,y2):  
     dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
     return dist  


# Define random initial conditions
def Initial_conditions():
    ## INITIAL MEMBRANE POTENTIAL
    U_exc = np.random.randn(CONST.neuron_memory,1);      # Initial membrane potential
    
    ## INITIAL WEIGHTS
    W_rec = math.sqrt( (CONST.Krec*(CONST.Alpha_rate ** 2)) / (CONST.Alpha_rate - CONST.Ft_rate) );        # Initial recurrent weight
    W_rec = np.ones((CONST.neuron_memory, CONST.neuron_memory)) * W_rec;
    W_rec = W_rec * 0.25;

    W_ff = math.sqrt( (CONST.Kff*(CONST.Alpha_rate*130)) / (CONST.Alpha_rate - CONST.Ft_rate) );        # Initial feed-forward weight
    W_ff = np.ones((CONST.neuron_memory, CONST.neurons_input)) * W_ff;
    s = np.random.uniform(0, 0.7, CONST.neuron_memory*CONST.neurons_input);                 # Draw uniform distribution {0, 0.7}
    s = s.reshape(CONST.neuron_memory, CONST.neurons_input);
    W_ff = W_ff * s;                # Scalar multiplication; Weight feed forward * unif. distribution

    
    ## INITIAL FIRING RATE
    ff_mem = U_exc;
    
    ## GRID CONNECTIVITY - RECURRENT
    connections_rec = np.zeros((CONST.neuron_memory, CONST.neuron_memory));
    counter = 0;
    
    # Loop to build coordinates for recurrent connections
    for ii in range (CONST.neuron_lin_col_memory):
        for jj in range (CONST.neuron_lin_col_memory):
            
                   
            # Create temporary zero matrix (e.g. 30x30)
            temp_matrix = np.zeros((CONST.neuron_lin_col_memory, CONST.neuron_lin_col_memory));
            
            for matrix_line_index in range (ii-CONST.radius_n, ii+CONST.radius_n+1):
                for matrix_column_index in range (jj-CONST.radius_n, jj+CONST.radius_n+1):
                    
                    # Calculate distance between two coordinates
                    temp_distance = calculateDistance(ii, jj, matrix_line_index, matrix_column_index);
                    
                    # Check if the distance between two coordinates are less than the radius
                    if temp_distance <= CONST.radius_n:
                        matrix_column_index_temp = matrix_column_index;
                        matrix_line_index_temp = matrix_line_index;
                        if matrix_column_index >= CONST.neuron_lin_col_memory:
                            matrix_column_index_temp = matrix_column_index - CONST.neuron_lin_col_memory;
                        if matrix_line_index >= CONST.neuron_lin_col_memory:
                            matrix_line_index_temp = matrix_line_index - CONST.neuron_lin_col_memory;
                        # Put 1 if connection between neuron i and j is true
                        temp_matrix[matrix_line_index_temp][matrix_column_index_temp] = 1; 
            
            
            # Build grid connectivity as a matrix where each line i is the connection between neuron i and j.
            connections_rec[counter] = temp_matrix.reshape(1, CONST.neuron_memory);
            # Counter for the loop
            counter = counter + 1;
    
    
    ## GRID CONNECTIVITY - FEED FORWARD
    
    # Create temporary zero matrix (memory area neurons x input area neurons)
    connections_ff = np.zeros((CONST.neuron_memory, CONST.neurons_input));
    
    # Loop to build the coordinates of the connections
    for ii in range (CONST.neuron_memory):
        
        # random number generator to define connections
        temp_connection = random.sample(range(CONST.neurons_input), CONST.ff_connections)
        
        for jj in range (CONST.ff_connections):
            # Put 1 if connection between neuron i (memory area) and j (input area) is true
            connections_ff[ii][temp_connection[jj]] = 1;
    
    
    # INITIATE VARIABLES
    W_rec = W_rec * connections_rec;
    W_ff = W_ff * connections_ff;
    U_inh_protocol = 0.0001;
    
    return U_exc, W_rec, connections_rec, connections_ff, W_ff, ff_mem, U_inh_protocol
