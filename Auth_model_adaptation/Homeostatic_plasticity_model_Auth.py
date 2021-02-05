"""
Contributors:   Amorim, FE
                Chapot, RL
                Moulin, TC
                Lee, J
                Amaral, OB

Main code

File description:
Model based on BiorXiv (Auth et al., 2018) later published in Frontiers (Auth et al., 2020).

The user can generate the data using the section SIMULATION PROTOCOLS. 
"""   

#################### LEARNING & REST ####################
def rest_step(Dict_const, U, F, U_inh,\
               FR_inh, W_rec_protocol, W_ff_protocol, connections_rec,\
               connections_ff):

# REST TIME
    for jj in range (0, int(CONST.Rest_time/CONST.dt)):
            
        # Update membrane potential from memory neurons
        U = EF.membrane_potential(U, CONST.tau_time, CONST.R_memory,\
                               W_rec_protocol, CONST.dt, CONST.W_i_inh,\
                               W_ff_protocol, F, FR_inh,\
                               CONST.Input_rest);
        
        # Update membrane potential from global inhibitory unit
        U_inh = EF.membrane_potential_inh(U_inh, CONST.tau_inh,\
                                       CONST.R_inh, CONST.W_inh_i, F,\
                                       CONST.dt);
                               
        # Update Firing rate from memory neurons
        F = EF.firing_rate(CONST.Alpha_rate, CONST.Beta_steepness, \
                        CONST.Epsilon_inflexion, U);
                               
                                           
        # Update Firing rate from global inhibitory unit
        FR_inh = EF.firing_rate(CONST.Alpha_rate, CONST.Beta_steepness, CONST.Epsilon_inflexion,\
                             U_inh);
    
        # Update weight matrix from recurrent connections (memory area -> memory area)
        W_rec_protocol = EF.weight_change_rec(CONST.mu_time, F,\
                                           np.transpose(F), CONST.Krec,\
                                           CONST.Ft_rate, W_rec_protocol,\
                                           connections_rec, CONST.dt);
                                               
        # Update weight matrix from Feed Forward connections (Input area -> memory area)
        W_ff_protocol = EF.weight_change_ff(CONST.mu_time, F,\
                                         np.transpose(CONST.Input_rest), CONST.Kff,\
                                         CONST.Ft_rate, W_ff_protocol,\
                                         connections_ff, CONST.dt);

    return U, F, U_inh, FR_inh, W_rec_protocol, W_ff_protocol


def variable_learn_memory(Dict_const, U, F, U_inh,\
               FR_inh, W_rec_protocol, W_ff_protocol, Input, Variable_session):
    
    # Initiate lists
    F_list = [];
    F_inh_list = [];
    W_rec_list = [];
    W_ff_list = [];
    U_list = [];
    U_inh_list = [];
    
    
    for mm in range (0, Variable_session):
    
        # REST TIME
        U, F, U_inh, FR_inh, W_rec_protocol, W_ff_protocol = rest_step(Dict_const,\
                                                                       U, F, U_inh,\
                                                                       FR_inh, W_rec_protocol,\
                                                                       W_ff_protocol)  
    
        # SESSION TIME
        for jj in range (0, int(CONST.Learning_time/CONST.dt)):
            # Update membrane potential from memory neurons
            U = EF.membrane_potential(U, CONST.tau_time, CONST.R_memory,\
                                   W_rec_protocol, CONST.dt, CONST.W_i_inh,\
                                   W_ff_protocol, F,\
                                   FR_inh, Input);
            
            # Update membrane potential from global inhibitory unit
            U_inh = EF.membrane_potential_inh(U_inh, CONST.tau_inh,\
                                           CONST.R_inh, CONST.W_inh_i, F,\
                                           CONST.dt);
                                           
            # Update Firing rate from memory neurons
            F = EF.firing_rate(CONST.Alpha_rate, CONST.Beta_steepness,\
                            CONST.Epsilon_inflexion, U);
        
            
                                           
            # Update Firing rate from global inhibitory unit
            FR_inh = EF.firing_rate(CONST.Alpha_rate, CONST.Beta_steepness, CONST.Epsilon_inflexion,\
                                 U_inh);
    
            # Update weight matrix from recurrent connections (memory area -> memory area)
            W_rec_protocol = EF.weight_change_rec(CONST.mu_time,F,\
                                               np.transpose(F), CONST.Krec,\
                                               CONST.Ft_rate, W_rec_protocol,\
                                               connections_rec, CONST.dt);
            
            # Update weight matrix from Feed Forward connections (Input area -> memory area)
            W_ff_protocol = EF.weight_change_ff(CONST.mu_time,F,\
                                             np.transpose(Input), CONST.Kff, CONST.Ft_rate,\
                                             W_ff_protocol, connections_ff, CONST.dt);
    
    
        F_list.append(F);
        F_inh_list.append(FR_inh);
        W_rec_list.append(W_rec_protocol);
        W_ff_list.append(W_ff_protocol);
        U_list.append(U);
        U_inh_list.append(U_inh)
        
    return  F_list, F_inh_list, W_rec_list, W_ff_list, U_list, U_inh_list



#################### LEARNING LOOPS #########################
def learn_ses_loop(Dict_const, U, F, U_inh,\
               FR_inh, W_rec_protocol, W_ff_protocol, connections_rec,\
               connections_ff, Input, Session_qt):
    
    # Initiate lists
    F_list = [];
    F_inh_list = [];
    W_rec_list = [];
    W_ff_list = [];
    U_list = [];
    U_inh_list = [];
    

    
    for mm in range (0, Session_qt):
    
        # REST TIME
        U, F, U_inh, FR_inh, W_rec_protocol, W_ff_protocol = rest_step(Dict_const,\
                                                                       U, F, U_inh,\
                                                                       FR_inh, W_rec_protocol,\
                                                                       W_ff_protocol, connections_rec,\
                                                                       connections_ff)  
    
        # SESSION TIME
        for jj in range (0, int(CONST.Learning_time/CONST.dt)):
            # Update membrane potential from memory neurons
            U = EF.membrane_potential(U, CONST.tau_time, CONST.R_memory,\
                                   W_rec_protocol, CONST.dt, CONST.W_i_inh,\
                                   W_ff_protocol, F,\
                                   FR_inh, Input);
            
            # Update membrane potential from global inhibitory unit
            U_inh = EF.membrane_potential_inh(U_inh, CONST.tau_inh,\
                                           CONST.R_inh, CONST.W_inh_i, F,\
                                           CONST.dt);
                                           
            # Update Firing rate from memory neurons
            F = EF.firing_rate(CONST.Alpha_rate, CONST.Beta_steepness,\
                            CONST.Epsilon_inflexion, U);
        
            
                                           
            # Update Firing rate from global inhibitory unit
            FR_inh = EF.firing_rate(CONST.Alpha_rate, CONST.Beta_steepness, CONST.Epsilon_inflexion,\
                                 U_inh);
    
            # Update weight matrix from recurrent connections (memory area -> memory area)
            W_rec_protocol = EF.weight_change_rec(CONST.mu_time,F,\
                                               np.transpose(F), CONST.Krec,\
                                               CONST.Ft_rate, W_rec_protocol,\
                                               connections_rec, CONST.dt);
            
            # Update weight matrix from Feed Forward connections (Input area -> memory area)
            W_ff_protocol = EF.weight_change_ff(CONST.mu_time,F,\
                                             np.transpose(Input), CONST.Kff, CONST.Ft_rate,\
                                             W_ff_protocol, connections_ff, CONST.dt);
        
        
                                        
    
        F_list.append(F);
        F_inh_list.append(FR_inh);
        W_rec_list.append(W_rec_protocol);
        W_ff_list.append(W_ff_protocol);
        U_list.append(U);
        U_inh_list.append(U_inh)
        
    return  F_list, F_inh_list, W_rec_list, W_ff_list, U_list, U_inh_list


def learn_ses_loop_ANISO(Dict_const, U, F, U_inh,\
               FR_inh, W_rec_protocol, W_ff_protocol, connections_rec,\
               connections_ff, Input, Session_qt, Aniso_value):
    
    # Initiate lists
    F_list = [];
    F_inh_list = [];
    W_rec_list = [];
    W_ff_list = [];
    U_list = [];
    U_inh_list = [];
    

    
    for mm in range (0, Session_qt):
    
        # REST TIME
        U, F, U_inh, FR_inh, W_rec_protocol, W_ff_protocol = rest_step(Dict_const,\
                                                                       U, F, U_inh,\
                                                                       FR_inh, W_rec_protocol,\
                                                                       W_ff_protocol, connections_rec,\
                                                                       connections_ff)  
    
        # SESSION TIME
        for jj in range (0, int(CONST.Learning_time/CONST.dt)):
            # Update membrane potential from memory neurons
            U = EF.membrane_potential(U, CONST.tau_time, CONST.R_memory,\
                                   W_rec_protocol, CONST.dt, CONST.W_i_inh,\
                                   W_ff_protocol, F,\
                                   FR_inh, Input);
            
            # Update membrane potential from global inhibitory unit
            U_inh = EF.membrane_potential_inh(U_inh, CONST.tau_inh,\
                                           CONST.R_inh, CONST.W_inh_i, F,\
                                           CONST.dt);
                                           
            # Update Firing rate from memory neurons
            F = EF.firing_rate(CONST.Alpha_rate, CONST.Beta_steepness,\
                            CONST.Epsilon_inflexion, U);
        
            
                                           
            # Update Firing rate from global inhibitory unit
            FR_inh = EF.firing_rate(CONST.Alpha_rate, CONST.Beta_steepness, CONST.Epsilon_inflexion,\
                                 U_inh);
    
            # Update weight matrix from recurrent connections (memory area -> memory area)
            W_rec_protocol = EF.weight_change_rec_ANISO(CONST.mu_time,F,\
                                               np.transpose(F), CONST.Krec,\
                                               CONST.Ft_rate, W_rec_protocol,\
                                               connections_rec, CONST.dt, Aniso_value);
            
            # Update weight matrix from Feed Forward connections (Input area -> memory area)
            W_ff_protocol = EF.weight_change_ff_ANISO(CONST.mu_time,F,\
                                             np.transpose(Input), CONST.Kff, CONST.Ft_rate,\
                                             W_ff_protocol, connections_ff, CONST.dt, Aniso_value);
    
    
        F_list.append(F);
        F_inh_list.append(FR_inh);
        W_rec_list.append(W_rec_protocol);
        W_ff_list.append(W_ff_protocol);
        U_list.append(U);
        U_inh_list.append(U_inh)
        
    return  F_list, F_inh_list, W_rec_list, W_ff_list, U_list, U_inh_list

def learn_ses_loop_DESTAB(Dict_const, U, F, U_inh,\
               FR_inh, W_rec_protocol, W_ff_protocol, connections_rec,\
               connections_ff, Input, Session_qt):
    
    # Initiate lists
    F_list = [];
    F_inh_list = [];
    W_rec_list = [];
    W_ff_list = [];
    U_list = [];
    U_inh_list = [];
    

    
    for mm in range (0, Session_qt):
    
        # REST TIME
        U, F, U_inh, FR_inh, W_rec_protocol, W_ff_protocol = rest_step(Dict_const,\
                                                                       U, F, U_inh,\
                                                                       FR_inh, W_rec_protocol,\
                                                                       W_ff_protocol, connections_rec,\
                                                                       connections_ff)  
    
        # SESSION TIME
        for jj in range (0, int(CONST.Learning_time/CONST.dt)):
            # Update membrane potential from memory neurons
            U = EF.membrane_potential(U, CONST.tau_time, CONST.R_memory,\
                                   W_rec_protocol, CONST.dt, CONST.W_i_inh,\
                                   W_ff_protocol, F,\
                                   FR_inh, Input);
            
            # Update membrane potential from global inhibitory unit
            U_inh = EF.membrane_potential_inh(U_inh, CONST.tau_inh,\
                                           CONST.R_inh, CONST.W_inh_i, F,\
                                           CONST.dt);
                                           
            # Update Firing rate from memory neurons
            F = EF.firing_rate(CONST.Alpha_rate, CONST.Beta_steepness,\
                            CONST.Epsilon_inflexion, U);
        
            
                                           
            # Update Firing rate from global inhibitory unit
            FR_inh = EF.firing_rate(CONST.Alpha_rate, CONST.Beta_steepness, CONST.Epsilon_inflexion,\
                                 U_inh);
    
            # Update weight matrix from recurrent connections (memory area -> memory area)
            W_rec_protocol = EF.weight_change_rec_DESTAB(CONST.mu_time,F,\
                                               np.transpose(F), CONST.Krec_DESTAB,\
                                               CONST.Ft_rate, W_rec_protocol,\
                                               connections_rec, CONST.dt);
            
            # Update weight matrix from Feed Forward connections (Input area -> memory area)
            W_ff_protocol = EF.weight_change_ff_DESTAB(CONST.mu_time,F,\
                                             np.transpose(Input), CONST.Kff_DESTAB, CONST.Ft_rate,\
                                             W_ff_protocol, connections_ff, CONST.dt);
    
    
        F_list.append(F);
        F_inh_list.append(FR_inh);
        W_rec_list.append(W_rec_protocol);
        W_ff_list.append(W_ff_protocol);
        U_list.append(U);
        U_inh_list.append(U_inh)
        
    return  F_list, F_inh_list, W_rec_list, W_ff_list, U_list, U_inh_list

#################### PROTOCOLS #########################
# PROTOCOL USING ANISOMYCIN IN M2 LEARNING SESSION

# PROTOCOL 1 FUNCTION
# Memory area is updated (U and F). Protocol that has a variation on the learning 
# and test input cue from 1 to input size. Two memories memory are learned.
# This function return the avg recurrent weight and feed forward
    
def Protocol_1(Dict_const, repetitions, Var_lear_input_M1, Var_lear_input_M2, Aniso_value, aniso_on_off, destab_on_off):
    
    from itertools import product
    
    qt_of_inputs = 10;    
    figures_switch = 0;
    extinction_def = 2;     
    
    # Initiate lists and calculate variables
    M1_train_weights = np.zeros((qt_of_inputs, 1));
    M2_train_weights = np.zeros((qt_of_inputs, 1));
    M2_train_weights_ext = np.zeros((qt_of_inputs, 1));
    M1_control_weights = np.zeros((qt_of_inputs, 1));
    M2_control_weights = np.zeros((qt_of_inputs, 1));
    M2_control_weights_ext = np.zeros((qt_of_inputs, 1));
    M1_aniso_weights = np.zeros((qt_of_inputs, 1));
    M2_aniso_weights = np.zeros((qt_of_inputs, 1));
    M2_aniso_weights_ext = np.zeros((qt_of_inputs, 1));
    M1_size = np.ones((qt_of_inputs, 1));
    M2_size = np.zeros((qt_of_inputs, 1));
    overlap = np.zeros((qt_of_inputs, 1));
    normalized_W1_overlap = np.zeros((qt_of_inputs, 1));
    
    connections_rec_all = np.zeros((qt_of_inputs+1, 900, 900));
    connections_ff_all = np.zeros((qt_of_inputs+1, 900, 36));
    
    weights_ff_all_train = np.zeros((qt_of_inputs+1, 900, 36));
    weights_rec_all_train = np.zeros((qt_of_inputs+1, 900, 900));
    F_all_train = np.zeros((qt_of_inputs+1, 900, 1));
    F_inh_all_train = np.zeros((qt_of_inputs+1, 1));
    U_all_train = np.zeros((qt_of_inputs+1, 900, 1));
    U_inh_all_train = np.zeros((qt_of_inputs+1, 1));

    weights_ff_all_cntr = np.zeros((qt_of_inputs+1, 900, 36));
    weights_rec_all_cntr = np.zeros((qt_of_inputs+1, 900, 900));
    F_all_cntr = np.zeros((qt_of_inputs+1, 900, 1));
    F_inh_all_cntr = np.zeros((qt_of_inputs+1, 1));
    U_all_cntr = np.zeros((qt_of_inputs+1, 900, 1));
    U_inh_all_cntr = np.zeros((qt_of_inputs+1, 1));

    weights_ff_all_aniso = np.zeros((qt_of_inputs+1, 900, 36));
    weights_rec_all_aniso = np.zeros((qt_of_inputs+1, 900, 900));
    F_all_aniso = np.zeros((qt_of_inputs+1, 900, 1));
    F_inh_all_aniso = np.zeros((qt_of_inputs+1, 1));
    U_all_aniso = np.zeros((qt_of_inputs+1, 900, 1));
    U_inh_all_aniso = np.zeros((qt_of_inputs+1, 1));


    ########## Number of repetitions ##########
    
        
    # Define initial conditions
    U, W_rec, connections_rec, connections_ff, W_ff, F, U_inh = EF.Initial_conditions();

        
    ####### STIMULUS 1 ##########
    F_learning_1, F_inh_learning_1, W_rec_learning_1, W_ff_learning_1, \
    U_learning_1, U_inh_learning_1 = learn_ses_loop(Dict_const,\
                                            U, F,\
                                            U_inh, CONST.FR_inh,\
                                            W_rec,\
                                            W_ff, connections_rec,\
                                            connections_ff, Var_lear_input_M1, CONST.Session_qt_M1);
    
                                                   
    if figures_switch == 1:                                                
        figure_avg_rec_weight_V2(W_rec_learning_1[-1], Dict_const, connections_rec, 1);
        
        
                                                    
    ####### DEFINING MEMORY CLUSTERS #######
    # TRAINING MEMORY
    # Binary condition (after first learning)
    R_M1 = np.array(F_learning_1[-1]);                  
    R_M1[R_M1 <= 0.5*CONST.Alpha_rate] = 0;
    R_M1[R_M1 >= 0.5*CONST.Alpha_rate] = 1;
    
    # Finding index coordinates to use in weight matrix
    R_M1_index = np.where(R_M1 == 1);
    R_M1_index = R_M1_index[0]; # To remove a tuple
    M1_value = len(R_M1_index);
    M1_size = M1_size * M1_value
    combinations_M1 = list(product(R_M1_index, R_M1_index));                                                    
    
    
    
    ####### WEIGHT STRENGTH IN EACH CLUSTER #######
    # TRAINING SESSION
    # First memory 
    W_sum_M1_train = cluster_mean_of_weights(W_rec_learning_1, combinations_M1);
    
    
    
    if extinction_def == 1:
        ####### STIMULUS 2 - Extinction memory ##########
        F_learning_2, F_inh_learning_2, W_rec_learning_2, W_ff_learning_2, \
        U_learning_2, U_inh_learning_2 = learn_ses_loop(Dict_const,\
                                                U_learning_1[-1], F_learning_1[-1], U_inh_learning_1[-1],\
                                                F_inh_learning_1[-1], W_rec_learning_1[-1],\
                                                W_ff_learning_1[-1], connections_rec,\
                                                connections_ff, Var_lear_input_M2[:, [qt_of_inputs-1]], CONST.Session_qt_M2);                                                
        
        
        if figures_switch == 1:                                                
            figure_avg_rec_weight_V2(W_rec_learning_2[-1], Dict_const, connections_rec, 2);
                                                        
        ####### DEFINING MEMORY CLUSTERS #######
        # EXTINCTION MEMORY
        # Defining neuron activity post memory 2 learning session
        # Binary condition (after Second learning)
        R_M2 = np.array(F_learning_2[-1]);                  
        R_M2[R_M2 <= 0.5*CONST.Alpha_rate] = 0;
        R_M2[R_M2 >= 0.5*CONST.Alpha_rate] = 1;                                                
        
        
        # Finding index coordinates to use in weight matrix
        R_M2_index = np.where(R_M2 == 1);
        R_M2_index = R_M2_index[0]; # To remove a tuple
        allowed_combinations_ext = list(product(R_M2_index, R_M2_index));
        
    else:
        # Finding index coordinates to use in weight matrix
        # Combinations of every neurons outside training cluster
        R_M2_index = np.where(R_M1 == 0);
        R_M2_index = R_M2_index[0]; # To remove a tuple
        combinations_M_Ext = list(product(R_M2_index, R_M2_index));
        
        # Combinations of every possible connection in rec memory area
        connection_index = np.where(connections_rec == 1);
        connection_index_i = connection_index[0];
        connection_index_j = connection_index[1];
        merged_connection = tuple(zip(connection_index_i, connection_index_j));
        merged_connection = list(merged_connection);
        
        # Combination of possible connections and neurons outside training cluster
        allowed_combinations_ext = list(set(merged_connection) & set(combinations_M_Ext));
        
                                                       
    ####### LOOP FOR INPUTS #######                                             
    for jj in range (0, qt_of_inputs):                   
        ####### STIMULUS 2 - without aniso ##########
        F_learning_2, F_inh_learning_2, W_rec_learning_2, W_ff_learning_2, \
        U_learning_2, U_inh_learning_2 = learn_ses_loop(Dict_const,\
                                                U_learning_1[-1], F_learning_1[-1], U_inh_learning_1[-1],\
                                                F_inh_learning_1[-1], W_rec_learning_1[-1],\
                                                W_ff_learning_1[-1], connections_rec,\
                                                connections_ff, Var_lear_input_M2[:, [jj]], CONST.Session_qt_M2);                                                
        
                                                   
        if figures_switch == 1:                                                
            figure_avg_rec_weight_V2(W_rec_learning_2[-1], Dict_const, connections_rec, 3);
       
                
        ####### STIMULUS 2 - With aniso ##########
        if aniso_on_off == 1:
            F_learning_3, F_inh_learning_3, W_rec_learning_3, W_ff_learning_3, \
            U_learning_3, U_inh_learning_3 = learn_ses_loop_ANISO(Dict_const,\
                                                                U_learning_1[-1], F_learning_1[-1], U_inh_learning_1[-1],\
                                                                F_inh_learning_1[-1], W_rec_learning_1[-1],\
                                                                W_ff_learning_1[-1], connections_rec,\
                                                                connections_ff, Var_lear_input_M2[:, [jj]], CONST.Session_qt_M2, Aniso_value);
           
            if figures_switch == 1:                                                
                figure_avg_rec_weight_V2(W_rec_learning_3[-1], Dict_const, connections_rec, 4);
                
        if destab_on_off == 1 and aniso_on_off == 0:
            F_learning_3, F_inh_learning_3, W_rec_learning_3, W_ff_learning_3, \
            U_learning_3, U_inh_learning_3 = learn_ses_loop_DESTAB(Dict_const,\
                                                                U_learning_1[-1], F_learning_1[-1], U_inh_learning_1[-1],\
                                                                F_inh_learning_1[-1], W_rec_learning_1[-1],\
                                                                W_ff_learning_1[-1], connections_rec,\
                                                                connections_ff, Var_lear_input_M2[:, [jj]], CONST.Session_qt_M2);
           
            if figures_switch == 1:                                                
                figure_avg_rec_weight_V2(W_rec_learning_3[-1], Dict_const, connections_rec, 4);

                                                  
        ####### DEFINING MEMORY CLUSTERS #######
        # REACTIVATION MEMORY
        # Defining neuron activity post memory 2 learning session
        # Binary condition (after Second learning)
        R_M2 = np.array(F_learning_2[-1]);                  
        R_M2[R_M2 <= 0.5*CONST.Alpha_rate] = 0;
        R_M2[R_M2 >= 0.5*CONST.Alpha_rate] = 1;                                                
        
        # Finding index coordinates to use in weight matrix
        R_M2_index = np.where(R_M2 == 1);
        R_M2_index = R_M2_index[0]; # To remove a tuple
        M2_size[jj] = len(R_M2_index);
        combinations_M2 = list(product(R_M2_index, R_M2_index));
        
                
        ####### WEIGHT STRENGTH IN EACH CLUSTER #######
        ## TRAINING SESSION
        # Second memory
        
        W_sum_M2_train = cluster_mean_of_weights(W_rec_learning_1, combinations_M2);
        # Extinction memory   
        W_sum_M2_train_ext = cluster_mean_of_weights(W_rec_learning_1, allowed_combinations_ext);

        ## REACTIVATION SESSION - CONTROL
        ## First memory
        W_sum_M1_cntrl = cluster_mean_of_weights(W_rec_learning_2, combinations_M1);
        
        ## Second memory        
        W_sum_M2_cntrl = cluster_mean_of_weights(W_rec_learning_2, combinations_M2);


        # Extinction memory        
        W_sum_M2_cntrl_ext = cluster_mean_of_weights(W_rec_learning_2, allowed_combinations_ext);

        
        ## REACTIVATION SESSION - ANISO        
        ## First memory
        W_sum_M1_aniso = cluster_mean_of_weights(W_rec_learning_3, combinations_M1);

        ## Second memory
        W_sum_M2_aniso = cluster_mean_of_weights(W_rec_learning_3, combinations_M2);
        
        # Extinction memory        
        W_sum_M2_aniso_ext = cluster_mean_of_weights(W_rec_learning_3, allowed_combinations_ext);
 

        # OUTPUTS
        M1_train_weights[jj] = np.array(W_sum_M1_train);                                                                                                                       
        M2_train_weights[jj] = np.array(W_sum_M2_train);                                                                                                                       
        M2_train_weights_ext[jj] = np.array(W_sum_M2_train_ext);                                                                                                                       
        M1_control_weights[jj] = np.array(W_sum_M1_cntrl);                                                                                                                       
        M2_control_weights[jj] = np.array(W_sum_M2_cntrl);                                                                                                                       
        M2_control_weights_ext[jj] = np.array(W_sum_M2_cntrl_ext);                                                                                                                       
        M1_aniso_weights[jj] = np.array(W_sum_M1_aniso);                                                                                                                       
        M2_aniso_weights[jj] = np.array(W_sum_M2_aniso);                                                                                                                       
        M2_aniso_weights_ext[jj] = np.array(W_sum_M2_aniso_ext);                                                                                                                       
        
        # Overlap between memory 1 and 2
        overlap[jj] = len(set(R_M1_index) & set(R_M2_index));
        if len(R_M1_index)==0:
            normalized_W1_overlap[jj] = 0;
        else:
            normalized_W1_overlap[jj] = (overlap[jj]/len(R_M1_index))*100;
        
        # RAW OUTPUTS - GENERAL
        connections_rec_all[jj,:,:] = connections_rec;
        connections_ff_all[jj,:,:] = connections_ff;
        
        # RAW OUTPUTS - Training
        weights_ff_all_train[jj,:,:] = W_ff_learning_1[-1];
        weights_rec_all_train[jj,:,:] = W_rec_learning_1[-1];
        F_all_train[jj,:] = F_learning_1[-1];
        F_inh_all_train[jj,:] = F_inh_learning_1[-1];
        U_all_train[jj,:] = U_learning_1[-1];
        U_inh_all_train[jj,:] = U_inh_learning_1[-1];
        
        # RAW OUTPUTS - Control
        weights_ff_all_cntr[jj,:,:] = W_ff_learning_2[-1];
        weights_rec_all_cntr[jj,:,:] = W_rec_learning_2[-1];
        F_all_cntr[jj,:] = F_learning_2[-1];
        F_inh_all_cntr[jj,:] = F_inh_learning_2[-1];
        U_all_cntr[jj,:] = U_learning_2[-1];
        U_inh_all_cntr[jj,:] = U_inh_learning_2[-1];
         
        # RAW OUTPUTS - Aniso
        weights_ff_all_aniso[jj,:,:] = W_ff_learning_3[-1];
        weights_rec_all_aniso[jj,:,:] = W_rec_learning_3[-1];
        F_all_aniso[jj,:] = F_learning_3[-1];
        F_inh_all_aniso[jj,:] = F_inh_learning_3[-1];
        U_all_aniso[jj,:] = U_learning_3[-1];
        U_inh_all_aniso[jj,:] = U_inh_learning_3[-1];
        
        
    # OUTPUTS
    M1_train_weights = M1_train_weights.transpose();                                                                                                                       
    M2_train_weights = M2_train_weights.transpose();
    M2_train_weights_ext = M2_train_weights_ext.transpose();                                                                                                                                                                                                                                             
    M1_control_weights = M1_control_weights.transpose();                                                                                                                       
    M2_control_weights = M2_control_weights.transpose();
    M2_control_weights_ext = M2_control_weights_ext.transpose();                                                                                                                      
    M1_aniso_weights = M1_aniso_weights.transpose();                                                                                                                       
    M2_aniso_weights = M2_aniso_weights.transpose();
    M2_aniso_weights_ext = M2_aniso_weights_ext.transpose();
    M1_size = M1_size.transpose();
    M2_size = M2_size.transpose();
    normalized_W1_overlap = normalized_W1_overlap.transpose();
    overlap = overlap.transpose();
    
    
    
    return  M1_train_weights, M2_train_weights,\
            M2_train_weights_ext, M1_control_weights, M2_control_weights, M2_control_weights_ext, M1_aniso_weights,\
            M2_aniso_weights, M2_aniso_weights_ext, overlap, normalized_W1_overlap, M1_size, M2_size, \
            weights_ff_all_train, weights_rec_all_train, F_all_train, F_inh_all_train, U_all_train, U_inh_all_train, \
            weights_ff_all_cntr, weights_rec_all_cntr, F_all_cntr, F_inh_all_cntr, U_all_cntr, U_inh_all_cntr, \
            weights_ff_all_aniso, weights_rec_all_aniso, F_all_aniso, F_inh_all_aniso, U_all_aniso, U_inh_all_aniso, \
            connections_rec_all, connections_ff_all



# PROTOCOL 2 FUNCTION
# Memory area is updated (U and F). Protocol that has a variation on the learning 
# and test input cue from 1 to input size. Two memories memory are learned.
# This function return the avg recurrent weight and feed forward
    
def Protocol_2(Dict_const, repetitions, Aniso_value, connec_rec, \
                F_train, W_rec_train, F_cntrl, W_rec_cntrl, F_aniso, W_rec_aniso):
        
    qt_of_inputs = 10;    
    figures_switch = 0  ;
    threshold = 40;
    threshold_cluster_size = 30;
    all_connection_switch = 0;
    neighborhood = 1;
    calculate_mean = 1;
    
    # Initiate lists and calculate variables
    M1_train_weights = np.zeros((qt_of_inputs, 1));
    M1_control_weights = np.zeros((qt_of_inputs, 1));
    M2_control_weights = np.zeros((qt_of_inputs, 1));
    M1_aniso_weights = np.zeros((qt_of_inputs, 1));
    M2_aniso_weights = np.zeros((qt_of_inputs, 1));
    Global_mean_W_train = np.zeros((qt_of_inputs, 1));
    Global_mean_W_cntrl = np.zeros((qt_of_inputs, 1));
    Global_mean_W_aniso = np.zeros((qt_of_inputs, 1));

    ########## Number of repetitions ##########
    if figures_switch == 1:                                                
        figure_avg_rec_weight_V2(W_rec_train[0,0,:], Dict_const, connec_rec[0,0,:], 1, 1);
                 
            
    ####### LOOP FOR INPUTS #######                                             
    for jj in range (0, qt_of_inputs):                   
                                                           
        if figures_switch == 1:                                                
            figure_avg_rec_weight_V2(W_rec_cntrl[0, jj, :], Dict_const, connec_rec[0,0,:], 3, jj);
       
               
        if figures_switch == 1:                                                
            figure_avg_rec_weight_V2(W_rec_aniso[0, jj, :], Dict_const, connec_rec[0,0,:], 4, jj);
                    
        
        ####### SETTING CLUSTER INDEX #######
        train_cluster_train, _ = define_2_clusters(W_rec_train[0,jj,:], W_rec_train[0,jj,:], connec_rec[0,jj,:], threshold, neighborhood);
        train_cluster_cntrl, control_cluster_M2 = define_2_clusters(W_rec_train[0,jj,:], W_rec_cntrl[0, jj, :], connec_rec[0,jj,:], threshold, neighborhood);
        train_cluster_aniso, aniso_cluster_M2 = define_2_clusters(W_rec_train[0,jj,:], W_rec_aniso[0, jj, :], connec_rec[0,jj,:], threshold, neighborhood);
        
        ####### WEIGHT STRENGTH IN EACH CLUSTER #######
        ## TRAINING SESSION
        W_sum_M1_train, All_mean_W_train = cluster_results_separated(W_rec_train[0, jj, :], train_cluster_train, connec_rec[0,jj,:], \
                                                                     threshold_cluster_size, all_connection_switch, calculate_mean);
        
        ## REACTIVATION SESSION - CONTROL
        ## First memory
        W_sum_M1_cntrl, All_mean_W_cntrl = cluster_results_separated(W_rec_cntrl[0, jj, :], train_cluster_cntrl, connec_rec[0,jj,:], \
                                                                     threshold_cluster_size, all_connection_switch, calculate_mean);
        ## Second memory        
        W_sum_M2_cntrl, All_mean_W_cntrl = cluster_results_separated(W_rec_cntrl[0, jj, :], control_cluster_M2, connec_rec[0,jj,:], \
                                                                     threshold_cluster_size, all_connection_switch, calculate_mean);
        
        ## REACTIVATION SESSION - ANISO        
        ## First memory
        W_sum_M1_aniso, All_mean_W_aniso = cluster_results_separated(W_rec_aniso[0, jj, :], train_cluster_aniso, connec_rec[0,jj,:], \
                                                                     threshold_cluster_size, all_connection_switch, calculate_mean);
        ## Second memory
        W_sum_M2_aniso, All_mean_W_aniso = cluster_results_separated(W_rec_aniso[0, jj, :], aniso_cluster_M2, connec_rec[0,jj,:], \
                                                                     threshold_cluster_size, all_connection_switch, calculate_mean);
        
        
        # OUTPUTS
        M1_train_weights[jj] = np.array(W_sum_M1_train);                                                                                                                       
        M1_control_weights[jj] = np.array(W_sum_M1_cntrl);                                                                                                                       
        M2_control_weights[jj] = np.array(W_sum_M2_cntrl);                                                                                                                       
        M1_aniso_weights[jj] = np.array(W_sum_M1_aniso);                                                                                                                       
        M2_aniso_weights[jj] = np.array(W_sum_M2_aniso);                                                                                                                       
        Global_mean_W_train[jj] = np.array(All_mean_W_train);
        Global_mean_W_cntrl[jj] = np.array(All_mean_W_cntrl);
        Global_mean_W_aniso[jj] = np.array(All_mean_W_aniso);
        
    # OUTPUTS
    M1_train_weights = M1_train_weights.transpose(); 
    M1_control_weights = M1_control_weights.transpose();
    M2_control_weights = M2_control_weights.transpose();
    M1_aniso_weights = M1_aniso_weights.transpose();
    M2_aniso_weights = M2_aniso_weights.transpose();
    Global_mean_W_train = Global_mean_W_train.transpose();
    Global_mean_W_cntrl = Global_mean_W_cntrl.transpose();
    Global_mean_W_aniso = Global_mean_W_aniso.transpose();
    
    return  M1_train_weights,\
            M1_control_weights, M2_control_weights, M1_aniso_weights,\
            M2_aniso_weights,\
            Global_mean_W_train, Global_mean_W_cntrl, Global_mean_W_aniso

#################### SMALL FUNCTIONS ####################

def cluster_mean_of_weights(W_rec, combinations):
## First memory
    # Finding recurrent weight values in coordinates
    W_sum_Memory = [];
    W_rec_final = W_rec[-1];
    for i in range(len(combinations)):
        W_sum_Memory = np.append(W_sum_Memory, W_rec_final[combinations[i]]);
    
    # Mean of rec weigths
    if len(W_sum_Memory) == 0:
        W_sum_Memory = 0;
    else:
        W_sum_Memory = sum(W_sum_Memory)/len(W_sum_Memory);
        
    return W_sum_Memory

def cluster_mean_of_weights_results(W_rec_final, combinations):
## First memory
    # Finding recurrent weight values in coordinates
    W_sum_Memory = [];
    for i in range(len(combinations)):
        W_sum_Memory = np.append(W_sum_Memory, W_rec_final[combinations[i]]);
    
    # Mean of rec weigths
    if len(W_sum_Memory) == 0:
        W_sum_Memory = 0;
    else:
        W_sum_Memory = sum(W_sum_Memory)/len(W_sum_Memory);
        
    return W_sum_Memory

def define_2_clusters(W_train, W_retrieval, connections, threshold, neighborhood):
    # Initiate parameters
    train_cluster = [];
    retrieval_cluster = np.empty((0));
    
    weight_train_tresholded = np.zeros((CONST.neuron_memory,1));
    weight_retrieval_tresholded = np.zeros((CONST.neuron_memory,1));
    
    # Acquire mean of weights for each neuron
    # Train Session
    for ii in range(0,len(W_train)-1):
        index_temp = np.nonzero(connections[ii] == 1);
        weight_train_tresholded[ii][0] = statistics.mean(W_train[ii][index_temp]);
    # Retrieval Session
    for ii in range(0,len(W_retrieval)-1):
        index_temp = np.nonzero(connections[ii] == 1);
        weight_retrieval_tresholded[ii][0] = statistics.mean(W_retrieval[ii][index_temp]);
    
    # Set 0 if mean is less than threshold; 1 if is above threshold    
    weight_train_tresholded[weight_train_tresholded < threshold] = 0;
    weight_retrieval_tresholded[weight_retrieval_tresholded < threshold] = 0;
    weight_train_tresholded[weight_train_tresholded >= threshold] = 1;
    weight_retrieval_tresholded[weight_retrieval_tresholded >= threshold] = 1;
    
    # Acquire index of neurons above threshold
    W_index_train = np.nonzero(weight_train_tresholded == 1);
    W_index_retrieval = np.nonzero(weight_retrieval_tresholded == 1);
    # tuple to int64
    W_index_train = W_index_train[0];
    W_index_retrieval = W_index_retrieval[0];
    
    
    train_cluster = W_index_train;
    
    if neighborhood == 1:
        for iii in range(0, len(W_index_retrieval)):
            counter = 0;
            for jjj in range(0, len(W_index_train)):
                # Condition - Search in every coordinate for the existence of a connection between training neuron and retrieval active neuron
                if connections[W_index_retrieval[iii], W_index_train[jjj]] == 1:
                   counter = 1;  
            if counter == 1:                    
                train_cluster = np.append(train_cluster, [W_index_retrieval[iii]], axis=0);
            else:
                retrieval_cluster = np.append(retrieval_cluster, [W_index_retrieval[iii]], axis=0);
        
        train_cluster = np.unique(train_cluster);
    else:
        # Use only neurons that belongs to train cluster
        train_cluster = np.unique(train_cluster);
        
        retrieval_cluster = np.setdiff1d(W_index_retrieval, W_index_train, assume_unique=True);
        
    # Output
    return train_cluster, retrieval_cluster

# CALCULATE CLUSTER MEAN WEIGHT
def cluster_results_separated(Weight_session, cluster_index, connections, threshold_cluster_size, all_connection_switch, calculate_mean):
    Mean_weights = np.zeros((CONST.neuron_memory,1));
    W_cluster = [];
    
    # Acquire mean of weights for each neuron
    # Train Session
    for ii in range(0,len(Weight_session)-1):
        index_temp = np.nonzero(connections[ii] == 1);
        Mean_weights[ii][0] = statistics.mean(Weight_session[ii][index_temp]);
    
    # Calculate global mean weight
    global_mean_weight = sum(Mean_weights)/len(Mean_weights);    
    
    # Calculate cluster mean weight
    if calculate_mean == 1:
        # If cluster is too small, mean connection will be the mean global value
        if len(cluster_index) < threshold_cluster_size:
            W_cluster = global_mean_weight; # Since there is no cluster, weight is the global mean weight
        else:
            cluster_index = cluster_index.astype(int)       # Turn float into integer to function as an index
            # Mean weight of cluster using all possible and existing connections from cluster neuron (even connections made with neurons outside cluster)
            if all_connection_switch == 1:
                W_cluster = sum(Mean_weights[cluster_index])/len(cluster_index);
            else:
                # Get all existing connections
                connection_index = np.where(connections == 1);
                connection_index_i = connection_index[0];
                connection_index_j = connection_index[1];
                all_possible_connections = tuple(zip(connection_index_i, connection_index_j));
                all_possible_connections = list(all_possible_connections);
                
                # Get all connections from cluster neurons to cluster neuron (even the non-existent)
                connections_inside_cluster = list(product(cluster_index, cluster_index));
                
                # Intersect between all existing connections and all connections between cluster neurons
                combinations = list(set(all_possible_connections) & set(connections_inside_cluster))
                
                # Get the weights from every coordinate obtained on intersect part
                for i in range(len(combinations)):
                    W_cluster = np.append(W_cluster, Weight_session[combinations[i]]);
                
                # Mean connection between neurons inside cluster
                W_cluster = sum(W_cluster)/len(W_cluster);
    
    # Calculate cluster weight using the sum of neurons
    else:
        # If cluster is too small, mean connection will be the mean global value
        if len(cluster_index) < threshold_cluster_size:
            W_cluster = np.array([0]); # Since there is no cluster, weight is the global mean weight
        else:
            cluster_index = cluster_index.astype(int)       # Turn float into integer to function as an index
            # Mean weight of cluster using all possible and existing connections from cluster neuron (even connections made with neurons outside cluster)
            if all_connection_switch == 1:
                W_cluster = sum(Mean_weights[cluster_index]);
            else:
                # Get all existing connections
                connection_index = np.where(connections == 1);
                connection_index_i = connection_index[0];
                connection_index_j = connection_index[1];
                all_possible_connections = tuple(zip(connection_index_i, connection_index_j));
                all_possible_connections = list(all_possible_connections);
                
                # Get all connections from cluster neurons to cluster neuron (even the non-existent)
                connections_inside_cluster = list(product(cluster_index, cluster_index));
                
                # Intersect between all existing connections and all connections between cluster neurons
                combinations = list(set(all_possible_connections) & set(connections_inside_cluster))
                
                # Get the weights from every coordinate obtained on intersect part
                for i in range(len(combinations)):
                    W_cluster = np.append(W_cluster, Weight_session[combinations[i]]);
                
                # Mean connection between neurons inside cluster
                W_cluster = sum(W_cluster);
                
    return W_cluster, global_mean_weight

#################### FIGURES ####################
    
# FIGURE FOR RECURRENT REACTIVATION
def figure_reactivation(F, Dict_const):
    # Adjustments
    F = F.reshape(int(math.sqrt(CONST.neuron_memory)) ,int(math.sqrt(CONST.neuron_memory)));
    
    # Figure: Recurrent activation
    fig, ax = plt.subplots()

    
    ax.imshow(F, cmap=plt.cm.Oranges,vmin=0, vmax=100, interpolation='nearest')
    ax.set_title('Recurrent activation')

    # Move left and bottom spines outward (if needed)
    ax.spines['left'].set_position(('outward', 0))
    ax.spines['bottom'].set_position(('outward', 0))
    # Hide the right and top spines
#    ax.spines['right'].set_visible(False)
#    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    plt.show()
    
    return

# FIGURE FOR MULTIPLES SESSSIONS ACTIVATION
def figure_reactivation_multi(All_F, Dict_const):
    # Adjustments
    
    
    # Figure: Recurrent activation
    
    fig3, axes = plt.subplots(repetitions, CONST.Session_qt_M2)
    fig3.suptitle('Activity from each session', fontsize = 16)
    
    for ax in axes:
        F = All_F[ax]
        for loops in range(0,CONST.Session_qt_M2):
            F_loop = F[loops]
            F_loop = F_loop.reshape(int(math.sqrt(CONST.neuron_memory)) ,int(math.sqrt(CONST.neuron_memory)));
            ax.imshow(F_loop, cmap=plt.cm.Oranges,vmin=0, vmax=100, interpolation='nearest')
            ax.set_title('Recurrent activation')
        
            # Move left and bottom spines outward (if needed)
            ax.spines['left'].set_position(('outward', 0))
            ax.spines['bottom'].set_position(('outward', 0))
            # Hide the right and top spines
        #    ax.spines['right'].set_visible(False)
        #    ax.spines['top'].set_visible(False)
            # Only show ticks on the left and bottom spines
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')

    plt.show()
    
    return
    
    

# FIGURE FOR AVERAGE INCOMING RECURRENT SYNAPTIC WEIGHT
def figure_avg_rec_weight(W, Dict_const):
    
    W_avg = np.zeros((CONST.neuron_memory,1));
    
   
    for ii in range(0,len(W)-1):
        index_temp = np.nonzero(connections_rec[ii] == 1);
        W_avg[ii][0] = statistics.mean(W[ii][index_temp]);
                
    W_avg = W_avg.reshape(int(math.sqrt(CONST.neuron_memory)), int(math.sqrt(CONST.neuron_memory)));
    
     # Figure
    fig, ax = plt.subplots()

    
    ax.imshow(W_avg, cmap=plt.cm.Greens, interpolation='nearest')
    ax.set_title('Recurrent Weight')

    # Move left and bottom spines outward (if needed)
    ax.spines['left'].set_position(('outward', 0))
    ax.spines['bottom'].set_position(('outward', 0))
    # Hide the right and top spines
#    ax.spines['right'].set_visible(False)
#    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    plt.show()
    
    return

# FIGURE FOR AVERAGE INCOMING RECURRENT SYNAPTIC WEIGHT
def figure_avg_rec_weight_V2(W, Dict_const, connections_rec, session, loop):
    
    W_avg = np.zeros((CONST.neuron_memory,1));
    
   
    for ii in range(0,len(W)-1):
        index_temp = np.nonzero(connections_rec[ii] == 1);
        W_avg[ii][0] = statistics.mean(W[ii][index_temp]);
                
    W_avg = W_avg.reshape(int(math.sqrt(CONST.neuron_memory)), int(math.sqrt(CONST.neuron_memory)));
    
     # Figure
    fig, ax = plt.subplots()

    
    ax.imshow(W_avg, cmap=plt.cm.Greens, vmin=0, vmax=100, interpolation='nearest')
    if session ==1:
        ax.set_title('Recurrent weight after training session');
    elif session == 2:
        ax.set_title('Recurrent weight for extinction memory');
    elif session == 3:
        ax.set_title('Recurrent weight after reactivation session - Control');
    elif session == 4:
        ax.set_title('Recurrent weight after reactivation session - Aniso');
    else:
        ax.set_title('Recurrent weight');

    # Move left and bottom spines outward (if needed)
    ax.spines['left'].set_position(('outward', 0))
    ax.spines['bottom'].set_position(('outward', 0))
    # Hide the right and top spines
#    ax.spines['right'].set_visible(False)
#    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    
    fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Greens, norm=plt.Normalize(vmin=0, vmax=100)), orientation='vertical');
    
    plt.show()
    
    if session==1:
        fig.savefig('simulation_Train'+str(loop)+'.eps', format='eps')
    elif session==3:
        fig.savefig('simulation_CNTRL'+str(loop)+'.eps', format='eps')
    elif session==4:
        fig.savefig('simulation_PSI'+str(loop)+'.eps', format='eps')
    
    return

def figure_RVO(RVO_mean, RVO_std, Dict_const):
    
    
    
    # Define y axis
    y = np.arange(0,1,1+1/np.count_nonzero(CONST.Input_1));
    
    plt.scatter(RVO_mean,y);
    
    return


if __name__ == '__main__':
    
    ###############################################################################
    # MAIN CODE
    ###############################################################################
    
    ## IMPORT
    import numpy as np
    import math
    import matplotlib.pyplot as plt
    import statistics
    import essential_functions as EF
    import model_constants_v2 as CONST
    from itertools import product
    
    ###############################################################################
    # Dictionary
    ###############################################################################
        
    #Dict_const = {"tau_exc": CONST.tau_time,"R": CONST.R_memory,"dt": CONST.dt,\
    #              "W_i_inh": CONST.W_i_inh, "I_1": CONST.Input_S1_learn,\
    #              "I_2": CONST.Input_S2_learn, "tau_inh": CONST.tau_inh, "R_inh": CONST.R_inh,\
    #              "W_inh_i": CONST.W_inh_i, "Alpha": CONST.Alpha_rate,\
    #              "Beta": CONST.Beta_steepness, "Epsilon": CONST.Epsilon_inflexion,\
    #              "mu": CONST.mu_time, "K_rec": CONST.Krec, "FT": CONST.Ft_rate,\
    #              "K_FF": CONST.Kff, "I_rest": CONST.Input_rest,\
    #              "S_qt": CONST.Session_qt, "I_5": CONST.Input_5, "Rest": CONST.Rest_time,\
    #              "I_6": CONST.Input_6,"T_time": CONST.Test_time};
                  
    Dict_const = {};
                  
                  
    #############################################################################
    # SIMULATION PROTOCOLS
    ###############################################################################
    '''
    simulation_protocol = 1: Generate the data to be loaded later for the figures protocol. 
    Save the whole session data as a spydata file after running since they took a while to run.
    
    simulation_protocol = 2: Generate the figures for the paper.
    Put the exact values of the amount of simulations and aniso value used on the loaded file.
    
    '''
    
    
    # PROTOCOL VARIABLE
    # DEFINE WHICH PROTOCOL IS GOING TO BE USED
    simulation_protocol = 1;
    
    ##############################################################################
    
    ###### Protocol loop (memory area update -> Inhibitory update -> W update)
    
    # Generate original results to be loaded after. Please, save them as a spydata file after running since they took a while to run.
    if simulation_protocol == 1:
        
        ##### Variable parameters #####
        repetitions = 20;           # Number of simulations
        aniso_on_off = 1;           # 0: reexposure without aniso; 1: reexposure with aniso
        Aniso_value = 1;            # 1: No influence on Hebbian plasticity. Lower values: Inhibition; Greater values: More Hebbian plasticity
        destab_on_off = 0;          # 0: reexposure without destabilization modulation; 1: reexposure with destabilization modulation
        
        ##### Define Matrices and vectors #####
        learning_input_M1 = CONST.Input_S1_learn;
        learning_input_M2 = np.zeros((36,10));
        
                
        # Changing learning input values
        learning_input_M2[:, [9]] = CONST.Input_S2_learn
        learning_input_M2[:, [8]] = CONST.Input_S2_learn_ov2
        learning_input_M2[:, [7]] = CONST.Input_S2_learn_ov4
        learning_input_M2[:, [6]] = CONST.Input_S2_learn_ov6
        learning_input_M2[:, [5]] = CONST.Input_S2_learn_ov8
        learning_input_M2[:, [4]] = CONST.Input_S2_learn_ov10
        learning_input_M2[:, [3]] = CONST.Input_S2_learn_ov12
        learning_input_M2[:, [2]] = CONST.Input_S2_learn_ov14
        learning_input_M2[:, [1]] = CONST.Input_S2_learn_ov16
        learning_input_M2[:, [0]] = CONST.Input_S1_learn
        
        # Define outputs
        M1_training = np.zeros((repetitions,10));
        M2_training = np.zeros((repetitions,10));
        M2_training_ext = np.zeros((repetitions,10));
        M1_reex_control = np.zeros((repetitions,10));
        M2_reex_control = np.zeros((repetitions,10));
        M2_reex_control_ext = np.zeros((repetitions,10));
        M1_reex_aniso = np.zeros((repetitions,10));
        M2_reex_aniso = np.zeros((repetitions,10));
        M2_reex_aniso_ext = np.zeros((repetitions,10));
        Sum_R_M1 = np.zeros((repetitions,10));
        Sum_R_M2 = np.zeros((repetitions,10));
        Overlap = np.zeros((repetitions,10));
        normalized_W1_overlap = np.zeros((repetitions,10));
        M1_size = np.zeros((repetitions,10));
        M2_size = np.zeros((repetitions,10));
        
        RAW_connections_rec_all = np.zeros((repetitions, 11, 900, 900));
        RAW_connections_ff_all = np.zeros((repetitions, 11, 900, 36));
        
        RAW_weights_ff_all_train = np.zeros((repetitions, 11, 900, 36));
        RAW_weights_rec_all_train = np.zeros((repetitions, 11, 900, 900));
        RAW_F_all_train = np.zeros((repetitions, 11, 900, 1));
        RAW_F_inh_all_train = np.zeros((repetitions, 11, 1));
        RAW_U_all_train = np.zeros((repetitions, 11, 900, 1));
        RAW_U_inh_all_train = np.zeros((repetitions, 11, 1));
    
        RAW_weights_ff_all_cntr = np.zeros((repetitions, 11, 900, 36));
        RAW_weights_rec_all_cntr = np.zeros((repetitions, 11, 900, 900));
        RAW_F_all_cntr = np.zeros((repetitions, 11, 900, 1));
        RAW_F_inh_all_cntr = np.zeros((repetitions, 11, 1));
        RAW_U_all_cntr = np.zeros((repetitions, 11, 900, 1));
        RAW_U_inh_all_cntr = np.zeros((repetitions, 11, 1));
    
        RAW_weights_ff_all_aniso = np.zeros((repetitions, 11, 900, 36));
        RAW_weights_rec_all_aniso = np.zeros((repetitions, 11, 900, 900));
        RAW_F_all_aniso = np.zeros((repetitions, 11, 900, 1));
        RAW_F_inh_all_aniso = np.zeros((repetitions, 11, 1));
        RAW_U_all_aniso = np.zeros((repetitions, 11, 900, 1));
        RAW_U_inh_all_aniso = np.zeros((repetitions, 11, 1));
        
    # Run protocol
        for jjj in range(repetitions):            
            M1_training[[jjj], :], M2_training[[jjj], :], M2_training_ext[[jjj], :], M1_reex_control[[jjj], :], M2_reex_control[[jjj], :],\
            M2_reex_control_ext[[jjj], :], M1_reex_aniso[[jjj], :], M2_reex_aniso[[jjj], :],\
            M2_reex_aniso_ext[[jjj], :], Overlap[[jjj], :], normalized_W1_overlap[[jjj], :],\
            M1_size[[jjj], :], M2_size[[jjj], :], \
            RAW_weights_ff_all_train[[jjj], :], RAW_weights_rec_all_train[[jjj], :], RAW_F_all_train[[jjj], :], RAW_F_inh_all_train[[jjj], :], RAW_U_all_train[[jjj], :], RAW_U_inh_all_train[[jjj], :], \
            RAW_weights_ff_all_cntr[[jjj], :], RAW_weights_rec_all_cntr[[jjj], :], RAW_F_all_cntr[[jjj], :], RAW_F_inh_all_cntr[[jjj], :], RAW_U_all_cntr[[jjj], :], RAW_U_inh_all_cntr[[jjj], :], \
            RAW_weights_ff_all_aniso[[jjj], :], RAW_weights_rec_all_aniso[[jjj], :], RAW_F_all_aniso[[jjj], :], RAW_F_inh_all_aniso[[jjj], :], RAW_U_all_aniso[[jjj], :], RAW_U_inh_all_aniso[[jjj], :], \
            RAW_connections_rec_all[[jjj], :], RAW_connections_ff_all[[jjj], :] = Protocol_1(Dict_const, repetitions, learning_input_M1, learning_input_M2, Aniso_value, aniso_on_off, destab_on_off);
   
                              
            
    # Generate figures of papers based on the simulations that were previously stored
    # LOAD RESULTS BEFORE RUNNING THIS SIMULATION PROTOCOL
    elif simulation_protocol == 2:
        
        ##### Variable parameters #####
        repetitions = 20;       # Number of simulations in the data
        Aniso_value = 0.75;     # Aniso_value when acquiring data
       
        
        # Define outputs
        M1_training = np.zeros((repetitions,10));
        M1_reex_control = np.zeros((repetitions,10));
        M2_reex_control = np.zeros((repetitions,10));
        M1_reex_aniso = np.zeros((repetitions,10));
        M2_reex_aniso = np.zeros((repetitions,10));
        Sum_R_M1 = np.zeros((repetitions,10));
        Sum_R_M2 = np.zeros((repetitions,10));
        global_mean_W_train = np.zeros((repetitions,10));
        global_mean_W_cntrl = np.zeros((repetitions,10));
        global_mean_W_aniso = np.zeros((repetitions,10));
        
        for jjj in range(repetitions):
        #for jjj in range(17 ,18):
            M1_training[[jjj], :],\
            M1_reex_control[[jjj], :], M2_reex_control[[jjj], :],\
            M1_reex_aniso[[jjj], :], M2_reex_aniso[[jjj], :],\
            global_mean_W_train[[jjj], :], global_mean_W_cntrl[[jjj], :], global_mean_W_aniso[[jjj], :],\
            = Protocol_2(Dict_const, repetitions, Aniso_value, RAW_connections_rec_all[[jjj],:], \
                   RAW_F_all_train[[jjj],:], RAW_weights_rec_all_train[[jjj],:], RAW_F_all_cntr[[jjj],:], RAW_weights_rec_all_cntr[[jjj],:],\
                   RAW_F_all_aniso[[jjj],:], RAW_weights_rec_all_aniso[[jjj],:]);
                           
                           
        if repetitions > 1:
            # Figure 1 - Training Session
            # Calculate mean of columns (each different input)
            W_Training_M1_std = np.std(M1_training, axis=0);
            W_Training_M1_mean = np.mean(M1_training, axis=0);
            W_Training_global_std = np.std(global_mean_W_train, axis=0);
            W_Training_global_mean = np.mean(global_mean_W_train, axis=0);
            
            x = np.arange(0,19,2);
            fig1 = plt.figure()
            plt.errorbar(x, W_Training_M1_mean, label='Memory 1', yerr=W_Training_M1_std);
            plt.errorbar(x, W_Training_global_mean, label='Global weight', yerr=W_Training_global_std);
            plt.ylabel('Weight value')
            plt.xlabel('Reexposure time');
            plt.legend(loc='lower right');
            plt.ylim(0, 100);
            plt.xticks(np.arange(0, 19, step=2));
            plt.title('Training Session', fontsize = 16);             
            fig1.savefig('training_session.eps', format='eps')
            
            # Figure 2 - Memory 1 Reactivation Session
            # Calculate mean of columns (each different input)
            W_React_M1_CNTRL_std = np.std(M1_reex_control, axis=0);
            W_React_M1_CNTRL_mean = np.mean(M1_reex_control, axis=0);
            W_React_M1_ANISO_std = np.std(M1_reex_aniso, axis=0);
            W_React_M1_ANISO_mean = np.mean(M1_reex_aniso, axis=0);
            W_Control_global_std = np.std(global_mean_W_cntrl, axis=0);
            W_Control_global_mean = np.mean(global_mean_W_cntrl, axis=0);
            W_Aniso_global_std = np.std(global_mean_W_aniso, axis=0);
            W_Aniso_global_mean = np.mean(global_mean_W_aniso, axis=0);
                        
            x = np.arange(0,19,2);
            fig2 = plt.figure()
            plt.errorbar(x, W_React_M1_CNTRL_mean, label='Control', yerr=W_React_M1_CNTRL_std);
            plt.errorbar(x, W_React_M1_ANISO_mean, label='Aniso', yerr=W_React_M1_ANISO_std);
            plt.errorbar(x, W_Control_global_mean, label='Global weight - Control', yerr=W_Control_global_std);
            plt.errorbar(x, W_Aniso_global_mean, label='Global weight - Aniso', yerr=W_Aniso_global_std);
            plt.ylabel('Weight value')
            plt.xlabel('Reexposure time');
            plt.legend(loc='lower right');
            plt.ylim(0, 100);
            plt.xticks(np.arange(0, 19, step=2));
            plt.title('Reactivation Session - Memory 1', fontsize = 16);
            fig2.savefig('react_mem1.eps', format='eps')
            
            # Figure 3 - Memory 2 Reactivation Session
            # Calculate mean of columns (each different input)
            W_React_M2_CNTRL_std = np.std(M2_reex_control, axis=0);
            W_React_M2_CNTRL_mean = np.mean(M2_reex_control, axis=0);
            W_React_M2_ANISO_std = np.std(M2_reex_aniso, axis=0);
            W_React_M2_ANISO_mean = np.mean(M2_reex_aniso, axis=0);
            
            x = np.arange(0,19,2);
            fig3 = plt.figure()
            plt.errorbar(x, W_React_M2_CNTRL_mean, label='Control', yerr=W_React_M2_CNTRL_std);
            plt.errorbar(x, W_React_M2_ANISO_mean, label='Aniso', yerr=W_React_M2_ANISO_std);
            plt.errorbar(x, W_Control_global_mean, label='Global weight - Control', yerr=W_Control_global_std);
            plt.errorbar(x, W_Aniso_global_mean, label='Global weight - Aniso', yerr=W_Aniso_global_std);
            plt.ylabel('Weight value')
            plt.xlabel('Reexposure time');
            plt.legend(loc='lower right');
            plt.ylim(0, 100);
            plt.xticks(np.arange(0, 19, step=2));
            plt.title('Reactivation Session - Memory 2', fontsize = 16);
            fig3.savefig('react_mem2.eps', format='eps')
            
            # Figure 4 - Ratio between memories Reactivation Session
            # Calculate mean of columns (each different input)            
            W_React_ratio_CNTRL_std = np.zeros((10));
            W_React_ratio_CNTRL_mean = np.zeros((10));
            W_React_ratio_ANISO_std = np.zeros((10));
            W_React_ratio_ANISO_mean = np.zeros((10));
            
            for i in range(0,10):
                temp_M1 = M1_reex_control[:,i];
                temp_M2 = M2_reex_control[:,i];
                W_React_ratio_CNTRL_std[i] = np.std(np.divide(temp_M1, temp_M2,\
                                        out=np.zeros_like(temp_M1), where=temp_M2!=0))
                
                W_React_ratio_CNTRL_mean[i] = np.mean(np.divide(temp_M1, temp_M2,\
                                        out=np.zeros_like(temp_M1), where=temp_M2!=0))
                
                temp_M1 = M1_reex_aniso[:,i];
                temp_M2 = M2_reex_aniso[:,i];
                W_React_ratio_ANISO_std[i] = np.std(np.divide(temp_M1, temp_M2,\
                                        out=np.zeros_like(temp_M1), where=temp_M2!=0))
                
                W_React_ratio_ANISO_mean[i] = np.mean(np.divide(temp_M1, temp_M2,\
                                        out=np.zeros_like(temp_M1), where=temp_M2!=0))
            
            x = np.arange(0,19,2);
            fig4 = plt.figure()
            plt.errorbar(x, W_React_ratio_CNTRL_mean, label='Control', yerr=W_React_ratio_CNTRL_std);
            plt.errorbar(x, W_React_ratio_ANISO_mean, label='Aniso', yerr=W_React_ratio_ANISO_std);
            plt.ylabel('Ratio value')
            plt.xlabel('Reexposure time');
            plt.legend(loc='lower right');
            plt.ylim(0, 5);
            plt.xticks(np.arange(0, 19, step=2));
            plt.title('Ratio between memory  1 and memory 2', fontsize = 16);
            fig4.savefig('react_ratio_non_norm.eps', format='eps')
            
            # Figure 5 - Ratio between memories Reactivation Session - Normalized
            # Calculate mean of columns (each different input)
            W_React_ratio_CNTRL_std_norm = np.zeros((10));
            W_React_ratio_CNTRL_mean_norm = np.zeros((10));
            W_React_ratio_ANISO_std_norm = np.zeros((10));
            W_React_ratio_ANISO_mean_norm = np.zeros((10));
            for i in range(0,10):
                W_React_ratio_CNTRL_std_norm[i] = np.std(np.divide(M1_reex_control[:,i], M1_reex_control[:,i] + M2_reex_control[:,i],\
                                        out=np.zeros_like(M1_reex_control[:,i]), where=M1_reex_control[:,i]!=0), axis=0)
                W_React_ratio_CNTRL_mean_norm[i] = np.mean(np.divide(M1_reex_control[:,i], M1_reex_control[:,i] + M2_reex_control[:,i],\
                                        out=np.zeros_like(M1_reex_control[:,i]), where=M1_reex_control[:,i]!=0), axis=0)
                W_React_ratio_ANISO_std_norm[i] = np.std(np.divide(M1_reex_aniso[:,i], M1_reex_aniso[:,i] + M2_reex_aniso[:,i],\
                                        out=np.zeros_like(M1_reex_aniso[:,i]), where=M1_reex_aniso[:,i]!=0), axis=0)
                W_React_ratio_ANISO_mean_norm[i] = np.mean(np.divide(M1_reex_aniso[:,i], M1_reex_aniso[:,i] + M2_reex_aniso[:,i],\
                                        out=np.zeros_like(M1_reex_aniso[:,i]), where=M1_reex_aniso[:,i]!=0), axis=0)
            
            x = np.arange(0,19,2);
            fig5 = plt.figure()
            plt.errorbar(x, W_React_ratio_CNTRL_mean_norm, label='Control', yerr=W_React_ratio_CNTRL_std_norm);
            plt.errorbar(x, W_React_ratio_ANISO_mean_norm, label='Aniso', yerr=W_React_ratio_ANISO_std_norm);
            plt.ylabel('Normalized value')
            plt.xlabel('Reexposure time');
            plt.legend(loc='lower right');
            plt.ylim(0, 1);
            plt.xticks(np.arange(0, 19, step=2));
            plt.title('Normalized - Memory 1 weight / Total weight', fontsize = 16);
            fig5.savefig('react_ratio_norm.eps', format='eps')

            