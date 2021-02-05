function [weight_update,y] = Weight_rules_post_react_controlled_noise(weight_update, nr_learning_rounds, t_initial, t_final, nr_neurons_h,...
    patterns_h, Ix, decay, S, ratio_HLP_SC, saturation,Reexposure_noise, Reexposure_noise_state, SC_on_off, vT)
clear ode23_attractors_cortex
clear Wtemp Stemp

Wtemp{1}=weight_update;

weight_update = (1-decay)*weight_update;
normalized_cue = (Ix/max(Ix)+1)/2;

for ii = 1:nr_learning_rounds
    
    if Reexposure_noise_state == 1
        rand('state', round(rand(1)*10)+round(rand(1)*10)+round(rand(1)*10)+1);
    end
    
    y0 = rand(nr_neurons_h, 1)/Reexposure_noise;
    [t, y] = ode23(@dy6, [t_initial t_final], y0);
    
    % p1 is the cue-induced pattern
    SteadyState=y(end, :);
    p1 = round(SteadyState); % p1 is just used for classification
    
    %%%%% Classification of the Steady state:
    p3 = find([sum((repmat(round(p1), 8, 1) == patterns_h)'), sum((repmat(1 - round(p1), 8, 1) == patterns_h)')] == nr_neurons_h);
    
    ode23_attractors_cortex(ii) = 0;
    if(~isempty(p3))
        ode23_attractors_cortex(ii) = p3(1);
    end
    
    %     term1=S*(p101'*p101) ; % ie, synthesis of positive connections among retrieved neurons;
    %     term2=-S*((1 - p101)'*p101) ; % ie, synthesis of negative connections between retrieved and non retrieved
    
    % terms due to synthesis of retrieved atractor
    term1=S*(SteadyState'*SteadyState) ; % ie, synthesis of positive connections among retrieved neurons;
    term2=-S*((1 - SteadyState)'*SteadyState) ;
    
    % Synaptic Scaling
    W_test = weight_update;
%     W_test(find(W_test < 0)) = 0;
    dW = ratio_HLP_SC * (ones(nr_neurons_h ,1) * (vT * ones(1, nr_neurons_h) - SteadyState));
    term3 = (dW.*W_test.^2);
    term3(find(W_test < 0)) = 0;
    
    
    dw0 = term1 + term2 + SC_on_off*term3;
    
    temp1 = find(dw0 > 0 & weight_update > 0); 
    dw0(temp1) = dw0(temp1) .* (1 - weight_update(temp1));
    temp1 = find(dw0 < 0 & weight_update < 0);
    dw0(temp1) = dw0(temp1) .* (1 + weight_update(temp1));
    clear temp1
    
    weight_update = weight_update + dw0;
    
    %     w0 = w0 + 0*term34 + 1*(term3+term4);
    %
    weight_update(find(weight_update > saturation)) = saturation;
    weight_update(find(weight_update < -saturation)) = -saturation;
    %
    %     w0(find(term3 < 0 & w0  < 0)) = 0;
    %     w0(find(term4 > 0 & w0  > 0)) = 0;
    %
    %     w0 = w0 + term1 + term2 ;
    %
    Wtemp{ii+1}=weight_update;
    Stemp{ii}=SteadyState;
    
    
end