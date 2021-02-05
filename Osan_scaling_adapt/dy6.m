function z = dy6(t, x)

global weight_update
global Ix

% factor1 = 0.;
% factor2 = 0; 

z = -x + (1 + tanh(weight_update*x  + Ix))/2;





