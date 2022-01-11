%This function represents the sigmoid activation function

function y = jer_tanh(x)
y = (exp(x)-(exp(-x)))./((exp(x))+exp(-x));
end