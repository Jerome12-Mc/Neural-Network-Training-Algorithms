

%This script is for training the deep neural network

clc;
clear;
close all;

tic
load('PD_data_4_4.mat'); %Load extracted PD data here
global inputs bias b hidden outputs;
b = 0;
inputs = 4;
outputs = 3;
hidden = 21;

%Specify the wanted inputs in this section
%input = zeros(1,inputs,4474);
input = out.Input;

%Specify the known outputs for each input specified 
observed = out.Output;

%Generate random initial weights for each layer of the network
w1 = (2*rand(hidden,inputs)-1);
w2 = (2*rand(hidden,hidden)-1);
w3 = (2*rand(outputs,hidden)-1);
w4 = (2*rand(outputs,hidden)-1);

%Compute the bias vector
bias = [b;b;b;b;b;b;b;b;b;b;b;b;b;b;b;];

%Generate a stopping point for number of iterations
T = 10000;
%f_val = zeros(T,1);

%[w1,w2,w3,w4]=jer_Deep_SGD(w1,w2,w3,w4,input,observed,T);
[w1,w2,w3] = jer_Nadam_DL_2(w1,w2,w3,input,observed,T);

toc

save('DLNN_SGD_tim.mat') %Save trained network