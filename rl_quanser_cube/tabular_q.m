%% Train Tabular Q-Learning Agent for Quanser QUBE Pendulum
% This script trains a tabular Q-learning agent to control the Quanser QUBE
% inverted pendulum using a discretized Simulink model.

clear; clc; close all;

% Ensure reproducibility
rng(0, 'twister');

fprintf('Starting QUBE Pendulum Tabular Q-Learning Training...\n');
fprintf('====================================================\n');

%% 1. Configuration
fprintf('1. Configuring environment...\n');

% Discretization bins (matching the Simulink model)
bins.phi = 12;          % Arm angle
bins.dphi = 12;         % Arm velocity
bins.theta = 16;        % Pendulum angle
bins.dtheta = 16;       % Pendulum velocity

numStates = prod(structfun(@(x) x, bins));
fprintf('   - State space size: %d bins (%d x %d x %d x %d)\n', ...
    numStates, bins.phi, bins.dphi, bins.theta, bins.dtheta);

% Discrete action space (7 levels, mapped in Simulink)
numActions = 7;
fprintf('   - Action space size: %d levels\n', numActions);

% Simulation and training parameters
Ts = 0.01;              % Sample time
Tf = 5;                 % Simulation time per episode
maxEpisodes = 5000;
maxSteps = Tf / Ts;

%% 2. Create the Environment
fprintf('2. Creating Simulink environment...\n');

% Define system limits (required by the Simulink model)
theta_limit = 5 * pi / 8;  % Pendulum angle limit
dtheta_limit = 20;         % Angular velocity limit
volt_limit = 8;           % Voltage limit

% Load the discretized Simulink model
mdl = 'rlQubeServo_discrete';
open_system(mdl);

% Define observation and action specifications
obsInfo = rlFiniteSetSpec(1:numStates);
actInfo = rlFiniteSetSpec(1:numActions);

% Create the environment
agentBlk = [mdl, '/RL Agent'];
env = rlSimulinkEnv(mdl, agentBlk, obsInfo, actInfo);

% Define the reset function
env.ResetFcn = @(in) qubeResetFcn(in);

fprintf('   - Simulink environment created successfully.\n');

%% 3. Create the Q-Learning Agent
fprintf('3. Creating Q-Learning agent...\n');

% Create Q-table and critic representation
qTable = rlTable(obsInfo, actInfo);
critic = rlQValueRepresentation(qTable, obsInfo, actInfo);

% Q-learning agent options
qOptions = rlQAgentOptions(...
    'SampleTime', Ts, ...
    'DiscountFactor', 0.99);
qOptions.EpsilonGreedyExploration.Epsilon = 1.0;
qOptions.EpsilonGreedyExploration.EpsilonDecay = 1.6e-6;
qOptions.EpsilonGreedyExploration.EpsilonMin = 0.1;

% Create the agent
agent = rlQAgent(critic, qOptions);
agent.AgentOptions.CriticOptimizerOptions.LearnRate = 0.05;

fprintf('   - rlQAgent created successfully.\n');

%% 4. Train the Agent
fprintf('4. Starting agent training...\n');

% Training options
trainOpts = rlTrainingOptions(...
    'MaxEpisodes', maxEpisodes, ...
    'MaxStepsPerEpisode', maxSteps, ...
    'ScoreAveragingWindowLength', 50, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'StopTrainingCriteria', 'AverageReward', ...
    'StopTrainingValue', 450);

% Train the agent
tic;
trainingStats = train(agent, env, trainOpts);
trainingTime = toc;

fprintf('\n   - Training completed in %.2f minutes.\n', trainingTime/60);

%% 5. Simulate and Save Results
fprintf('5. Simulating trained agent and saving results...\n');

% Simulate the trained agent
simOpts = rlSimulationOptions('MaxSteps', maxSteps);
experience = sim(env, agent, simOpts);

% Save results
save('tabularQAgent.mat', 'agent', 'trainingStats', 'experience', 'bins');
fprintf('   - Trained agent and results saved to tabularQAgent.mat.\n');

fprintf('\nProcess completed successfully!\n');
fprintf('====================================================\n');

%% Helper Functions

function in = qubeResetFcn(in)
    % Reset initial conditions to upright position for stabilization
    % Start with small random perturbations around the equilibrium
    
    % Pendulum angle: Small deviation from 0 (upright)
    % Using +/- 0.1 radians (approx +/- 6 degrees)
    theta0 = (rand - 0.5) * 0.2; 
    
    % Arm angle: Small deviation from 0 (center)
    phi0 = (rand - 0.5) * 0.2;
    
    in = setVariable(in, 'theta0', theta0);
    in = setVariable(in, 'phi0', phi0);
    in = setVariable(in, 'dtheta0', 0);
    in = setVariable(in, 'dphi0', 0);
end
