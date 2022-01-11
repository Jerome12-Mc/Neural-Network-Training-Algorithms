%This function represents the Swish activation function

function y = jer_Swish(x)
z = 1./(1+exp(-x));
y = 1.5.*x.*z;
end