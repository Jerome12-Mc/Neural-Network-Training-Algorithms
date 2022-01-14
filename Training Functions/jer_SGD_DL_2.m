% This function represents the Deep Learning Method (Backpropagation with SGD) 
%for a multi layer Neural Network with 3 hidden layers 

% Function variables
% w1       - weights from input to 1st hidden layer
% w2       - weights from 1st hidden layer to 2nd hidden layer
% w3       - weights from 2nd hidden layer to output layer
% input    - the input values
% observed - the actual/expected outputs for inputs
% T        - number of iterations for training

function [w1,w2,w3] = jer_SGD_DL_2(w1,w2,w3,input,observed,T)
t = 1; %Initial step
tol = 1e-05; %Tolerance
init = zeros(T,1);
%f_val = zeros(T,1);
for j = 1:T
alpha = 0.01; % The Learning rate
global inputs bias;
N = 2759; % This number represents the number of outputs we want

for k = 1:N
    
    %Convert to single column (the 'inputs' variable represents the
    %number of input nodes).
    %reshaped_input = reshape(input(:,:,k),inputs,1);
    reshaped_input = reshape(input(k,:),inputs,1);

    %Now we specify the inputs and outputs for each hidden layer
    input_of_hidden_layer1 = w1*reshaped_input ;%+ bias;
    output_of_hidden_layer1 = jer_Swish(input_of_hidden_layer1);

    input_of_hidden_layer2 = w2*output_of_hidden_layer1; % + bias;
    output_of_hidden_layer2 = jer_Swish(input_of_hidden_layer2);    
    
    % For more hidden layers keep adding this line
    
    input_of_output_node = w3*output_of_hidden_layer2;
    final_output = jer_LeakyReLU(input_of_output_node);
    
    observed_transpose = observed(k, :)';
    error = observed_transpose - final_output;
   % error2 = error.^2;
   % init(k:k,1:3) = error2;
   
    
    %Printing the error values and they should be decreasing as the network is trained
    %fprintf('%i\n', error)
    
    %Now we implement back propagation using delta rule
   % delta = ((final_output)+(jer_sigmoid(input_of_output_node).*(1.5-final_output))).*error;
     delta = jer_LeakyRelu_derivative(input_of_output_node).*error;

     error_of_hidden_layer2 = w3'*delta;
    %delta2 = jer_LeakyRelu_derivative(input_of_hidden_layer2).*error_of_hidden_layer2;
    delta2 = ((output_of_hidden_layer2)+(jer_sigmoid(input_of_hidden_layer2).*(1.5-output_of_hidden_layer2))).*error_of_hidden_layer2;
    
    error_of_hidden_layer1 = w2'*delta2;
   %delta1 = jer_LeakyRelu_derivative(input_of_hidden_layer1).*error_of_hidden_layer1;
    delta1 = ((output_of_hidden_layer1)+(jer_sigmoid(input_of_hidden_layer1).*(1.5-output_of_hidden_layer1))).*error_of_hidden_layer1;

    
   
    update_of_w3 = alpha*delta*output_of_hidden_layer2';
    update_of_w2 = alpha*delta2*output_of_hidden_layer1';
    update_of_w1 = alpha*delta1*reshaped_input';
    
    w1 = w1 + update_of_w1;
    w2 = w2 + update_of_w2;
    w3 = w3 + update_of_w3;

end
%  s = sum(init);
%  MSE = s/t; 
%  s2 = sum(MSE)/3;
%  f_val(j:j,1:1) = s2 ;
%  fprintf('%i\n', s2)
% Update initial step
        t = 1+t;
        fprintf('%i\n', t-1)
 %End the loop if the error reaches the desired value
       % if error<=tol
        %    break
        %end         
end
fprintf('The number of iterations is: %d\n',t-1);
fprintf('The error reached is: %d\n',error);
%fprintf('%i\n', s2);
end
    
    