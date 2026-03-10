clear; close all; clc;

rng(42);

fprintf('START TRAIN\n');
% [history, trainedNet] = training();
fprintf('FINISH TRAIN\n');

fprintf('START TEST\n');
detection();
fprintf('FINISH TEST\n');