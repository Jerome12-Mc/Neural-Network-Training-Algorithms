%This function represents the Gaussian Error Linear Unit activation function

function y = jer_GELU(x)
y = 0.5.*x.*(1+tanh((sqrtm(2/pi)).*(x+0.0447152.^3)));
end