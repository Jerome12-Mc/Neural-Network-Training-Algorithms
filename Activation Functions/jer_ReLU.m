%This function represents the Rectified Linear Unit activation function

function y = jer_ReLU(x)
y = max(0.01.*x,x);
end