% Signals Load File
% 
% By: Ian Pylkkanen
% December 2020
% 
% This file loads all the signals produced by the dynamic simulations performed in NX.
% Both the excitation signal and the response singals from each sensor are loaded.
% The data is saved in a .mat file for use in system_identification_demo_2.m

%%
clear all
close all

%% Create Excitation Signal
% -------- DISPLAY ONLY --------
% TO SHOW HOW SIGNAL WAS CREATED
% DO NOT UNCOMMENT

% Use randn.m to generate a Gaussian White Noise Signal

% n = 2500;                                 % Number of Samples
% x = randn(1,n)';                          % White Noise Signal
% fs = 2500;                                % Sampling Frequency
% t = (0:n-1)*(1/fs);                       % Sampling Time
% 
% plot(t,x)                                 % Plot signal to verify   
% data = [t',x];                            % Save in the form required for NX 
% csvwrite('excitation_signal.csv',data);   % Save the data in a .csv file

%% Load Exciation Signal

x = csvread('excitation_signal.csv');       % Load generated exciation signal
x = x(:,2);                                 % Save only Force (N) data

%% Load Sensor Singals
% Each load call will create a variable 'ans' which will need to be saved as the respective sensor number n
% Only the acceleration data (column 2) is saved and transposed
% i.e. yn = ans(2,:)'

% Sensor 1
load sensor1_resp.mat
y1 = ans(2,:)';

% Sensor 2
load sensor2_resp.mat
y2 = ans(2,:)';

% Sensor 3
load sensor3_resp.mat
y3 = ans(2,:)';

% Sensor 4
load sensor4_resp.mat
y4 = ans(2,:)';

% Sensor 5
load sensor5_resp.mat
y5 = ans(2,:)';

% Sensor 6
load sensor6_resp.mat
y6 = ans(2,:)';

%% Save .mat file
% Choose which sensor you would like to analyze
y = y1;                                     % Save sensor as 'y'      
clear ans                                   % Clear ans
save signals.mat                            % Save in .mat file

% Move to system_identification_demo_2.m file and call 'singals.mat'
    % The input signal MUST be saved as: 'x'
    % The output signal MUST be saved as: 'y'
    % All other variables in signals.mat will be ignored