%This is the Softmax activation function

function y = jer_Softmax(x)
ex = exp(x);
y = ex/sum(ex);
end