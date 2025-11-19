%% Train PPO Agent for Quanser QUBE Pendulum
% This script trains a Proximal Policy Optimization (PPO) agent to control 
% the Quanser QUBE-Servo 2 inverted pendulum system.
% PPO is an on-policy algorithm that balances sample efficiency and stability.

clear; clc; close all;

% Ensure reproducibility
rng(0, 'twister');

fprintf('Starting QUBE Pendulum PPO Training...\n');
fprintf('======================================\n');

%% 1. Configure Environment
fprintf('1. Configuring environment...\n');

% System parameters
theta_limit = 5 * pi / 8;
dtheta_limit = 30;
volt_limit = 12;
Ts = 0.01; % Sample time

% Load Simulink model
mdl = 'rlQubeServo';
open_system(mdl);

% Observation and Action Specifications
obsInfo = rlNumericSpec([7 1]);
actInfo = rlNumericSpec([1 1], 'UpperLimit', 1, 'LowerLimit', -1);

% Create Environment
agentBlk = [mdl, '/RL Agent'];
env = rlSimulinkEnv(mdl, agentBlk, obsInfo, actInfo);
env.ResetFcn = @localResetFcn;

fprintf('   - Environment created.\n');

%% 2. Create PPO Agent
fprintf('2. Creating PPO agent...\n');

% Initialize agent with default networks (smaller for memory efficiency)
initOpts = rlAgentInitializationOptions('NumHiddenUnit', 256);

% PPO Agent Options (tuned for lower memory usage)
agentOpts = rlPPOAgentOptions(...
    'SampleTime', Ts, ...
    'ExperienceHorizon', 500, ...     % Reduced from 1024 (less memory per update)
    'MiniBatchSize', 128, ...          % Reduced from 256 (smaller batches)
    'ClipFactor', 0.2, ...
    'EntropyLossWeight', 0.01, ...
    'NumEpoch', 5, ...                % Reduced from 5 (faster updates, less memory reuse)
    'AdvantageEstimateMethod', 'gae', ...
    'GAEFactor', 0.95, ...
    'DiscountFactor', 0.99);

agentOpts.ActorOptimizerOptions.Algorithm = 'adam';
agentOpts.ActorOptimizerOptions.LearnRate = 3e-4;
agentOpts.ActorOptimizerOptions.GradientThreshold = 1;

agentOpts.CriticOptimizerOptions.Algorithm = 'adam';
agentOpts.CriticOptimizerOptions.LearnRate = 1e-3;
agentOpts.CriticOptimizerOptions.GradientThreshold = 1;

rng(0, 'twister');
agent = rlPPOAgent(obsInfo, actInfo, initOpts, agentOpts);

fprintf('   - PPO agent created.\n');

%% 3. Train Agent
fprintf('3. Starting training...\n');

% Training Options (memory-efficient settings)
maxEpisodes = 2000;  % Reduced from 2000
maxSteps = ceil(5/Ts);

trainOpts = rlTrainingOptions(...
    'MaxEpisodes', maxEpisodes, ...
    'MaxStepsPerEpisode', maxSteps, ...
    'ScoreAveragingWindowLength', 20, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'StopTrainingCriteria', 'AverageReward', ...
    'StopTrainingValue', 450, ...
    'SaveAgentCriteria', 'none', ...  % Disable auto-save to avoid disk corruption
    'UseParallel', true);

% Configure parallel mode (sync is more stable and uses less temp disk space)
trainOpts.ParallelizationOptions.Mode = 'async';
% trainOpts.ParallelizationOptions.StepsUntilDataIsSent = 32;  % Send data frequently
% trainOpts.ParallelizationOptions.DataToSendFromWorkers = 'Experiences';

% Train
trainingStats = train(agent, env, trainOpts);

%% 4. Simulate and Save
fprintf('4. Simulating and saving...\n');

% Simulate
simOpts = rlSimulationOptions('MaxSteps', maxSteps);
experience = sim(env, agent, simOpts);

% Save
save('ppoAgent.mat', 'agent', 'trainingStats', 'experience');
fprintf('   - Saved to ppoAgent.mat\n');

fprintf('Done.\n');

%% Helper Functions
function in = localResetFcn(in)
    % Reset initial conditions (Swing-up task)
    % Randomize angles to encourage swing-up learning from various positions
    theta0 = -pi/4 + rand * pi/2;
    phi0 = pi - pi/4 + rand * pi/2;
    
    in = setVariable(in, 'theta0', theta0);
    in = setVariable(in, 'phi0', phi0);
    in = setVariable(in, 'dtheta0', 0);
    in = setVariable(in, 'dphi0', 0);
end
