%This function represents the Leaky Rectified Linear Unit activation function

function y = jer_LeakyReLU(x)
y = max(0.1.*x,x);
end