%This function represents the Exponential Linear Unit activation function

function y = jer_ELU(x)
y = max(0.1.*(exp(x)-1),x);
end