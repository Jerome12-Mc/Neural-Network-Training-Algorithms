%This function represents the derivative of the Leaky Rectified Linear Unit activation function

function y = jer_LeakyRelu_derivative(x)
  n=size(x,1);
  f_val = zeros(n,1);
  for k=1:n 
       if x(k)>=0
       y=1;
    else
       y=0.1;
       end
       f_val(k:k,1:1) = y;
  end
  y=f_val;
end

