# Neural Network Training Algorithms in MATLAB (from scratch)
Supervised learning training algorithms used for control of a CubeSat.

A Master's Final Year Project in Mechanical Engineering with Aeronautics

by Jerome McCave.

### IMPORTANT: 
Reproduction of this publication in whole or in part must include the customary bibliographic citation, including
author attribution, report title, etc.

### Suggested citation: 
Jerome A McCave (2022). Application of a Biologically Inspired Controller for Satellite Control. University of Glasgow,
James Watt School of Engineering. URL: https://github.com/Jerome12-Mc/Neural-Network-Training-Algorithms

## Description
Satellite attitude control has been an area of research since the advent of the space age. PI (Proportional Integral), PD (Proportional Derivative) and PID (Proportional Integral Derivative) controllers
remain as the standard approach to achieve proper attitude control in satellites. However, there are
numerous studies which attest to the limitations involved with this standard, and present hybrid controllers which are better suited. The main objective of this thesis was to investigate the applicability
of a biologically inspired control method for a CubeSat satellite. Neural Networks (NN) were the
chosen biologically inspired control method and it was constructed via Matlab and Simulink software
in an effort to train the NN with data obtained from a PD controller.

NNs were chosen for this use case as they operate on non-linear principles which made it well suited
for a CubeSat attitude control which is an example of a non-linear dynamic system. A Deep-Layer
NN was developed to test and compare the performance of several different supervised learning training algorithms. The selected training algorithms were Stochastic Gradient Descent (SGD), Gradient
Descent with Momentum (GDM), Decaying Momentum with GDM (Demon GDM), Adaptive Momentum (Adam), Decaying Momentum with Adam (Demon Adam), Nesterov and Adam (Nadam),
the Adam Variant (AMSGrad) and Quasi Hyperbolic Momentum (QHM). The training algorithms
were compared against mean square error (MSE), root mean square error (RMSE) and mean absolute
error values (MAE), time taken to complete training and control responses in the Simulink environment. The method by which these training algorithms were implemented are detailed in depth in this
report.

The results demonstrated that Adam and Nadam were the best performing training algorithms, having
achieved the lowest MSE, RMSE and MAE values. They also performed best in control responses,
having smoother responses than the other training algorithms and being closer to the target response
than the PD controller. Investigation on the impact of hyperparameters and activation functions associated with the NN structure were also carried out in this report. Results on this demonstrated that
activation functions, Leaky ReLU and Swish, performed better when applied to the NN. Detail on
values of selected hyperparameters for the aforementioned training algorithms were documented in
this report.
