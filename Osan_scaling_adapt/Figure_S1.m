%%% Run Main structure - Memories were burned first, than the synaptic
%%% weights were used to make the retrieval. Multiple session extinction of
%%% 4 days using different mismatchs to represent minor shock and regular
%%% extinction without shock
%%% Ix1: Non-related Memory
%%% Ix2: Context A + Shock
%%% Ix3: Context A + Non-Shock
%%% Ix4: Context B + Shock
%%% Ix5: Context B + Non-Shock
%%% Mismatch closer to 0 equals to shock memory and closer to 10, non-shock
%%% memory
clear; close all; clc;


global non_related_memories_quantity with_overlap with_overlap_non_related same_memory

context_size = 4;
non_related_memories_quantity = 1;
with_overlap = 0;                               % 0 = Without overlap; 1 = With overlap
with_overlap_non_related = 0;                   % 0 = Without overlap; 1 = With overlap
same_memory = 1;
simulation_quantity = 1;


if context_size == 1
    model_main_structure_v2;
elseif context_size == 4
    model_main_structure_cxt_4_memories;
elseif context_size == 6
    model_main_structure_cxt_6_memories;
    
end

freezing_probabilities = [0.1 0.9 0.1 0.9 0.1 0.1 0.1 0.1];

% Remove patterns
patterns_h = patterns_h(1:3,:);
%% Setting parameters
global decay Ix

cue_factor = 1;                                   % Multiplying factor for retrieval cues

% Non related memory size
non_related_size = 14;


shock_neurons_factor = 1;                           % Multiplying factor for shock neurons input

% Mismatch neurons define number of neurons becoming active between shock
% and non-shock neurons. If 0, input is equal Ix4 (Context B + Tone +
% Shock). If 10, input is equal Ix5 (Context B + Tone + Non-Shock)

max_mismatch = 10;
% mismatch_for_no_shock_group = 10;           % number of mismatch neurons (max = 10) in extinction
% mismatch_for_minor_shock_group = 6;
decayrate = 0;
decayrate_NR = 0;
ratio_HLP_SC = 1;
general_SC_on_off = 1;
vT = 0;


%%% TRAINING
S_training = 0.8;
Learning_rounds_training = 1;
Training_factor = 1;                            % Multiplying factor for conditioning input


%%% REACTIVATION
S_max_value_reexposure = 1;
S_values_reexposure = 0:0.025:S_max_value_reexposure;
ratio_HLP_SC_react = 1;
ratio_HLP_SC_aniso = 1;
Quant_ext_sessions = 1;
Learning_round_extinction = 1;
Input_reexposure_factor = cue_strength;
Reactivation_choice_within_retrieval = 0;               % 0 = Retrieval tests; 1 = Within activity

%%% RETRIEVAL TEST
% Retrieval context
% 1 = Cxt A; 2 = Cxt B; 3 = Cxt A + Tone; 4 = Cxt B + Tone
retrieval_stimulus_choice = 1;


%%% NOISE PARAMETERS
% Noise
Weight_noise = 0.1;
Learning_noise = 0.1;
Reexposure_noise = 0.1;
Retrieval_noise = 0.1;

% Noise state
% 1 = On; 0 = Off
Learning_noise_state = 1;           % 0: Off; 1: On
Reexposure_noise_state = 1;         % 0: Off; 1: On
Retrieval_noise_state = 1;          % 0: Off; 1: On
Input_noise = 1;                    % 0: Off; 1: On

% On and Off parameters (0 = off; 1 = on)
Renewal_on_off = 0;
Re_training_on_off = 0;                              % 0 = off; 1 = on


%% Conditionals before protocol

% Retrieval stimulus type
if retrieval_stimulus_choice == 1
    retrieval_stimulus = Ixcue_CXT_A;
elseif retrieval_stimulus_choice == 2
    retrieval_stimulus = Ixcue_CXT_B;
elseif retrieval_stimulus_choice == 3
    retrieval_stimulus = Ixcue_CXT_A_tone;
elseif retrieval_stimulus_choice == 4
    retrieval_stimulus = Ixcue_CXT_B_tone;
end


% % Noises adaptations
% Learning_noise = 1/Learning_noise;
% Reexposure_noise = 1/Reexposure_noise;


%% BURNING FIRST MEMORY - Non-related Memory
% weight_update = zeros(nr_neurons_h, nr_neurons_h);
% weight_update = (Weight_noise*rand(nr_neurons_h, nr_neurons_h))-0.05;

decay = decayrate_NR;

S = S_training;
weight_first_session_control = zeros(100,100,simulation_quantity);

for LLLL = 1:simulation_quantity
    % CONTROL GROUP
    weight_update = (Weight_noise*rand(nr_neurons_h, nr_neurons_h))-0.05;
    
    for iiiii = 1:non_related_memories_quantity
        
        Ix1 = Ix_Non_related(:,:,iiiii,LLLL);
        Ix1_update = Ix1 * Training_factor;
        Ix = Ix1_update;
        
        if Input_noise == 1
            % TEST FOR NOISE IN Ix (In range from 0.9 to 1.1)
            aa = 0.9;   % Min possible value
            bb = 1.1;   % Max possible value
            RN = (bb-aa).*rand(100,1) + aa; % Random numbers from a uniform distribution (aa, bb)
            Ix = Ix .* RN;
        end
        
        weight_first_session_control(:,:,LLLL) = Weight_rules_controlled_noise_V2(weight_update, nr_learning_rounds, t_initial, t_final, nr_neurons_h, patterns_h,...
            Ix, decay, S, ratio_HLP_SC, saturation, Learning_noise, Learning_noise_state, general_SC_on_off, vT);
        weight_update = weight_first_session_control(:,:,LLLL);
        
    end
    
    
end

%% BURNING SECOND MEMORY - Training (Context A + Shock)

decay = decayrate;
% decay = 0;
S = S_training;


weight_second_session_control = zeros(100,100,simulation_quantity, length(S_values_reexposure));
Ratio_session_control_training_retr = zeros(100,1,simulation_quantity, length(S_values_reexposure));
Activity_retrieval_control_training = zeros(1,100,simulation_quantity, length(S_values_reexposure));

for TTTT = 1:length(S_values_reexposure)
    for LLLL = 1:simulation_quantity
        % CONTROL GROUP
        Ix2_update = Ix2 * Training_factor;
        Ix2_update(Shock_neurons) = Ix2_update(Shock_neurons) * shock_neurons_factor;
        Ix = Ix2_update;
        
        if Input_noise == 1
            % TEST FOR NOISE IN Ix (In range from 0.9 to 1.1)
            aa = 0.9;   % Min possible value
            bb = 1.1;   % Max possible value
            RN = (bb-aa).*rand(100,1) + aa; % Random numbers from a uniform distribution (aa, bb)
            Ix = Ix .* RN;
        end
        
        
        weight_update = weight_first_session_control(:,:,LLLL);
        weight_second_session_control(:,:,LLLL, TTTT) = Weight_rules_controlled_noise_V2(weight_update, Learning_rounds_training, t_initial, t_final, nr_neurons_h,...
            patterns_h, Ix, decay,  S_training, ratio_HLP_SC, saturation, Learning_noise, Learning_noise_state, general_SC_on_off, vT);
        
        weight_update = weight_second_session_control(:,:,LLLL, TTTT);
        Ix = cue_factor * retrieval_stimulus;
        retrieval_controlled_noise;
        session_control_train_cue_shock = shock_neuron_activity;
        session_control_train_cue_Non_shock = Non_shock_neuron_activity;
        % To adjust the ratio
        session_control_train_cue_Non_shock = session_control_train_cue_Non_shock + session_control_train_cue_shock;
        Ratio_session_control_training_retr(:,:,LLLL, TTTT) = (session_control_train_cue_shock ./ session_control_train_cue_Non_shock);
        Activity_retrieval_control_training(:,:,LLLL, TTTT) = mean(mean_act_all_neurons);
        
        clear session_no_footshock_train_cue shock_neuron_activity Non_shock_neuron_activity mean_act_all_neurons session_no_footshock_train_cue_Non_shock
        
        
    end
end
%% REACTIVATION SESSION

Ratio_session_control_extinction_cue = zeros(100 , 1, Quant_ext_sessions, simulation_quantity, max_mismatch+1, length(S_values_reexposure));
weight_extinction_control = zeros(nr_neurons_h, nr_neurons_h, Quant_ext_sessions, simulation_quantity, max_mismatch+1, length(S_values_reexposure));
React_control_within_session_activity = zeros(1, nr_neurons_h, Quant_ext_sessions, simulation_quantity, max_mismatch+1, length(S_values_reexposure));
Activity_ret_all_neurons_control = zeros(1, nr_neurons_h, Quant_ext_sessions, simulation_quantity, max_mismatch+1, length(S_values_reexposure));

% Parameters that we could change
decay = decayrate;
% S = S_extinction_react;
nr_learning_rounds = Learning_round_extinction;

for TTTT = 1:length(S_values_reexposure)
    for LLLL = 1:simulation_quantity
        
        % Mismatch defined as the same as Osan et al 2011
        for mismatch_neurons = 0:max_mismatch                   % ALL OVERLAPS
            
            % Osan Mismatch Input
            sigfunction=1./(1.+exp((-mismatch_neurons + (max_mismatch/2))/1));
            Ix = (Ix3 - Ix2)*sigfunction + Ix2;
            if(mismatch_neurons >= max_mismatch)
                Ix = Ix3;
            end
            
            % Random removal
            
%             mismatch_neurons_to_remove = sort(randperm(10,mismatch_neurons));
%             mismatch_neurons_to_add = sort(randperm(10,mismatch_neurons));
%             
%             Ix = Ix2;
%             Ix(find(Ix == learning_strength)) = Input_reexposure_factor;
%             Ix(find(Ix == -learning_strength)) = -Input_reexposure_factor;
%             Ix(Shock_neurons) = Input_reexposure_factor;
%             Ix(Shock_neurons(mismatch_neurons_to_remove)) = -Input_reexposure_factor;
%             Ix(Non_shock_neurons(mismatch_neurons_to_add)) = Input_reexposure_factor;
            
            if Input_noise == 1
                % TEST FOR NOISE IN Ix (In range from 0.9 to 1.1)
                aa = 0.9;   % Min possible value
                bb = 1.1;   % Max possible value
                RN = (bb-aa).*rand(100,1) + aa; % Random numbers from a uniform distribution (aa, bb)
                Ix = Ix .* RN;
            end
            
            
            % CONTROL
            weight_update = weight_second_session_control(:,:,LLLL, TTTT);
            
            %  Reactivation Session
            for iii = 1:Quant_ext_sessions
                [weight_extinction_control(:, :, iii, LLLL, mismatch_neurons+1, TTTT), React_control_within_session_activity(:, :, iii, LLLL, mismatch_neurons+1, TTTT)] = Weight_rules_post_react_controlled_noise_V2(weight_update, nr_learning_rounds,...
                    t_initial, t_final, nr_neurons_h, patterns_h, Ix, decay, S_values_reexposure(TTTT), ratio_HLP_SC_react, saturation, Reexposure_noise, Reexposure_noise_state, general_SC_on_off, vT);
                weight_update = weight_extinction_control(:, :, iii, LLLL, mismatch_neurons+1, TTTT);
                %     first_extinction_no_shock_session_activity = first_extinction_no_shock_session_activity(end, :);
            end
            
            if Reactivation_choice_within_retrieval == 0
                for iiii = 1:Quant_ext_sessions
                    weight_update = weight_extinction_control(:,:,iiii, LLLL, mismatch_neurons+1, TTTT);
                    Ix = cue_factor * retrieval_stimulus;
                    retrieval_controlled_noise;
                    old_reexposure_control(mismatch_neurons+1, :, TTTT) = ode23_attractors_cortex;
                    session_CONTROL_react_cue_shock(:,:,iiii, LLLL, mismatch_neurons+1, TTTT) = shock_neuron_activity;
                    session_CONTROL_react_cue_Non_shock(:,:,iiii, LLLL, mismatch_neurons+1, TTTT) = Non_shock_neuron_activity;
                    % To adjust the ratio
                    session_CONTROL_react_cue_Non_shock_and_shock(:,:,iiii, LLLL, mismatch_neurons+1, TTTT) = abs(session_CONTROL_react_cue_Non_shock(:,:,iiii, LLLL, mismatch_neurons+1, TTTT)...
                        ) + abs(session_CONTROL_react_cue_shock(:,:,iiii, LLLL, mismatch_neurons+1, TTTT));
                    Ratio_session_control_extinction_cue(:,:,iiii, LLLL, mismatch_neurons+1, TTTT) = abs(session_CONTROL_react_cue_shock(:,:,iiii, LLLL, mismatch_neurons+1, TTTT)...
                        ) ./ session_CONTROL_react_cue_Non_shock_and_shock(:,:,iiii, LLLL, mismatch_neurons+1, TTTT);
                    Activity_ret_all_neurons_control(:,:,iiii, LLLL, mismatch_neurons+1, TTTT) = mean(mean_act_all_neurons);
                    
                    %                 clear session_control_train_cue_Non_shock shock_neuron_activity session_control_train_cue_shock Non_shock_neuron_activity mean_act_all_neurons
                    
                end
            end
            
            
            
        end
    end
end


%% FREEZING TEST - OLD FORM - OSAN RETRIEVAL PROPABILITIES

% Probabilities
p_reexposure_control(:,:,:) = 100*(rand(size(old_reexposure_control)) < freezing_probabilities(old_reexposure_control(:,:,:)));

% Mean calculation
mean_reexposure_control = mean(p_reexposure_control, 2);

mean_reexposure_control = squeeze(mean_reexposure_control);
mean_reexposure_control = mean_reexposure_control';


%% FIGURE - HEATMAP FOR CONTROL GROUP
set(gcf,'renderer','Painters');
heatmap_control = heatmap(mean_reexposure_control, 'CellLabelColor','none');
heatmap_control.Title = 'Vehicle';
heatmap_control.XLabel = 'Reexposure time (t)';
heatmap_control.YLabel = 'S value in Reexposure learning';
heatmap_control.Colormap = jet;
heatmap_control.YDisplayLabels = num2cell(0:0.025:1);
heatmap_control.XDisplayLabels = num2cell(0:1:10);
colorbar;
caxis([0, 100]);


