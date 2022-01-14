% This function represents the Deep Learning Method (Backpropagation with Adam (Adaptive Momentum)) 
%for a multi layer Neural Network

% Function variables
% w1       - weights from input to 1st hidden layer
% w2       - weights from 1st hidden layer to 2nd hidden layer
% w3       - weights from 2nd hidden layer to output layer
% input    - the input values
% observed - the actual/expected outputs for inputs
% T        - number of iterations for training

function [w1,w2,w3] = jer_Nadam_DL_2(w1,w2,w3,input,observed,T)
t = 1; %Initial step
tol = 1e-09; %Tolerance
for j = 1:T
global inputs bias hidden outputs;
alpha = 0.001; % The Learning rate
beta_1 = 0.9;  % Discount factor
beta_2 = 0.999;
epsilon = 10e-08;
% Initial parameters set to 0
initial_mt1 = zeros(hidden,inputs);
initial_vt1 = zeros(hidden,inputs);
initial_mt2 = zeros(hidden,hidden);
initial_vt2 = zeros(hidden,hidden);
initial_mt3 = zeros(outputs,hidden);
initial_vt3 = zeros(outputs,hidden);
N = 2759; % This number represents the number of data we want trained
for k = 1:N
    
    %Convert to single column (the 'inputs' variable represents the
    %number of input nodes).
    %reshaped_input = reshape(input(:,:,k),inputs,[]);
    reshaped_input = reshape(input(k,:),inputs,1);
    
    %Now we specify the inputs and outputs for each hidden layer
    input_of_hidden_layer1 = w1*reshaped_input; %+ bias ;
    output_of_hidden_layer1 = jer_Swish(input_of_hidden_layer1);
    
    input_of_hidden_layer2 = w2*output_of_hidden_layer1; %+ bias;
    output_of_hidden_layer2 = jer_Swish(input_of_hidden_layer2);
    
    input_of_output_node = w3*output_of_hidden_layer2;
    final_output = jer_LeakyReLU(input_of_output_node);
    
    observed_transpose = observed(k, :)';
    error = observed_transpose - final_output;
    %error2 = error.^2;
    %init(k:k,1:3) = error2;

    %Printing the error values and they should be decreasing as the network is trained
    %fprintf('%i\n', error)
    
    %Now we implement back propagation using delta rule
    %delta = ((final_output)+(jer_sigmoid(input_of_output_node).*(1.5-final_output))).*error;
    delta = jer_LeakyRelu_derivative(input_of_output_node).*error;

     error_of_hidden_layer2 = w3'*delta;
    %delta2 = jer_LeakyRelu_derivative(input_of_hidden_layer2).*error_of_hidden_layer2;
    delta2 = ((output_of_hidden_layer2)+(jer_sigmoid(input_of_hidden_layer2).*(1.5-output_of_hidden_layer2))).*error_of_hidden_layer2;
    
    error_of_hidden_layer1 = w2'*delta2;
   %delta1 = jer_LeakyRelu_derivative(input_of_hidden_layer1).*error_of_hidden_layer1;
    delta1 = ((output_of_hidden_layer1)+(jer_sigmoid(input_of_hidden_layer1).*(1.5-output_of_hidden_layer1))).*error_of_hidden_layer1;


    %Generate Adam algorithm    

    dweight_w3 = delta*output_of_hidden_layer2';
    dweight_w2 = delta2*output_of_hidden_layer1';
    dweight_w1 = delta1*reshaped_input';

    % Update the initial parameters    
    initial_mt1 = (beta_1*initial_mt1)+((1-beta_1).*dweight_w1);
    initial_vt1 = (beta_2*initial_vt1)+((1-beta_2).*dweight_w1.^2);

    initial_mt2 = (beta_1*initial_mt2)+((1-beta_1).*dweight_w2);
    initial_vt2 = (beta_2*initial_vt2)+((1-beta_2).*dweight_w2.^2);

    initial_mt3 = (beta_1*initial_mt3)+((1-beta_1).*dweight_w3);
    initial_vt3 = (beta_2*initial_vt3)+((1-beta_2).*dweight_w3.^2);

    % Calculate bias correction
    initial_m_hat1 = initial_mt1./(1-beta_1^t);
    v_hat1 = initial_vt1./(1-beta_2^t);

    initial_m_hat2 = initial_mt2./(1-beta_1^t);
    v_hat2 = initial_vt2./(1-beta_2^t);

    initial_m_hat3 = initial_mt3./(1-beta_1^t);
    v_hat3 = initial_vt3./(1-beta_2^t);

    scale1 = (alpha./(sqrt(v_hat1)+epsilon)).*((beta_1*initial_m_hat1)+(((1-beta_1)./(1-beta_1^t))*dweight_w1));
    scale2 = (alpha./(sqrt(v_hat2)+epsilon)).*((beta_1*initial_m_hat2)+(((1-beta_1)./(1-beta_1^t))*dweight_w2));
    scale3 = (alpha./(sqrt(v_hat3)+epsilon)).*((beta_1*initial_m_hat3)+(((1-beta_1)./(1-beta_1^t))*dweight_w3));
   
 
    %Update weights   
    w1 = w1 + scale1;
    w2 = w2 + scale2;
    w3 = w3 + scale3;
        
end
%s = sum(init);
 % MSE = s/t; 
  %s2 = sum(MSE)/3;
  %f_val(j:j,1:1) = s2 ;
  %fprintf('%i\n', s2)
% Update initial step
        t = 1+t;
        fprintf('%i\n', t-1)
 %End the loop if the error reaches the desired value
        %if error<=tol
         %   break
        %end         
end
fprintf('The number of iterations is: %d\n',t-1);
fprintf('The error reached is: %d\n',error);
end
    
