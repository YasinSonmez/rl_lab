function trainQubePendulumAgent(modelName)
    % TRAINQUBEPENDULUMAGENT - Train reinforcement learning agents to control Quanser QUBE pendulum
    %
    % This function implements a complete example for training RL agents to control
    % the Quanser QUBE-Servo 2 inverted pendulum system using MATLAB's 
    % Reinforcement Learning Toolbox.
    %
    % The function:
    % 1. Sets up the QUBE pendulum environment
    % 2. Creates a TD3 (Twin Delayed Deep Deterministic) agent
    % 3. Trains the agent to balance the pendulum
    % 4. Simulates the trained agent performance
    %
    % Requirements:
    % - MATLAB R2020b or later
    % - Reinforcement Learning Toolbox
    % - Simulink
    % - Simscape Multibody
    % - Simscape Electrical
    %
    % Author: Generated for RL Lab
    % Date: 2024
    
    % Ensure reproducibility
    rng(0, 'twister');
    
    fprintf('Starting QUBE Pendulum Reinforcement Learning Training...\n');
    fprintf('================================================\n');
    
    % Load and open the Simulink model
    if nargin < 1 || isempty(modelName)
        mdl = 'rlQubeServo';
    else
        mdl = modelName;
    end
    try
        open_system(mdl);
        fprintf('✓ Simulink model loaded: %s\n', mdl);
    catch ME
        fprintf('Error loading Simulink model: %s\n', ME.message);
        fprintf('Please ensure the model file exists and is accessible.\n');
        if contains(ME.message, 'created with a newer version', 'IgnoreCase', true)
            fprintf(['\nModel version mismatch detected.\n', ...
                     'This model was saved in a newer Simulink release.\n', ...
                     'Options to resolve:\n', ...
                     '  1) Open the model in its original MATLAB (newer) release and use:\n', ...
                     '     File > Save > Export Model to > Previous Version,\n', ...
                     '     then choose your current release.\n', ...
                     '  2) In your current MATLAB, open Preferences > Simulink > Warnings/Errors\n', ...
                     '     and change "Newer release model" behavior (not recommended for production).\n', ...
                     '  3) Provide a compatible model name when calling this function, e.g.:\n', ...
                     '     trainQubePendulumAgent(''''rlQubeServo_R2024b'''');\n\n']);
        end
        return;
    end
    
    % Define system parameters
    theta_limit = 5 * pi / 8;  % radians (pendulum angle limit)
    dtheta_limit = 30;         % radians/second (angular velocity limit)
    volt_limit = 12;           % volts (motor voltage limit)
    Ts = 0.005;                % sample time in seconds
    
    fprintf('✓ System parameters defined\n');
    fprintf('  - Pendulum angle limit: %.2f rad (%.1f°)\n', theta_limit, rad2deg(theta_limit));
    fprintf('  - Angular velocity limit: %.1f rad/s\n', dtheta_limit);
    fprintf('  - Voltage limit: %.1f V\n', volt_limit);
    fprintf('  - Sample time: %.3f s\n', Ts);
    
    % Define initial conditions
    theta0 = 0;      % initial pendulum angle
    phi0 = 0;        % initial motor arm angle
    dtheta0 = 0;     % initial pendulum angular velocity
    dphi0 = 0;       % initial motor arm angular velocity
    
    % Create observation and action specifications
    % Observations: [theta, phi, dtheta, dphi, cos(theta), sin(theta), cos(phi)]
    obsInfo = rlNumericSpec([7 1], 'Name', 'observations');
    obsInfo.Description = 'Pendulum and motor states';
    
    % Actions: motor voltage command [-1, 1] (scaled to [-12V, 12V])
    actInfo = rlNumericSpec([1 1], 'Name', 'actions', ...
        'UpperLimit', 1, 'LowerLimit', -1);
    actInfo.Description = 'Motor voltage command';
    
    fprintf('✓ Observation and action specifications created\n');
    fprintf('  - Observation space: %dx%d\n', obsInfo.Dimension(1), obsInfo.Dimension(2));
    fprintf('  - Action space: %dx%d\n', actInfo.Dimension(1), actInfo.Dimension(2));
    
    % Create the environment
    agentBlk = [mdl, '/RL Agent'];
    try
        env = rlSimulinkEnv(mdl, agentBlk, obsInfo, actInfo);
        env.ResetFcn = @resetFunction;
        fprintf('✓ Reinforcement learning environment created\n');
    catch ME
        fprintf('Error creating environment: %s\n', ME.message);
        return;
    end
    
    % Define the agent
    fprintf('Creating TD3 agent...\n');
    agent = createTD3Agent(obsInfo, actInfo, Ts);
    fprintf('✓ TD3 agent created successfully\n');
    
    % Set training options
    trainOpts = rlTrainingOptions(...
        'MaxEpisodes', 1000, ...
        'MaxStepsPerEpisode', ceil(5 / Ts), ...  % 5 seconds per episode
        'ScoreAveragingWindowLength', 5, ...
        'Verbose', false, ...
        'Plots', 'training-progress', ...
        'StopTrainingCriteria', 'AverageReward', ...
        'StopTrainingValue', 500, ...
        'SaveAgentCriteria', 'EpisodeReward', ...
        'SaveAgentValue', 400);
    
    fprintf('✓ Training options configured\n');
    fprintf('  - Max episodes: %d\n', trainOpts.MaxEpisodes);
    fprintf('  - Max steps per episode: %d\n', trainOpts.MaxStepsPerEpisode);
    fprintf('  - Stop training value: %.1f\n', trainOpts.StopTrainingValue);
    
    % Train the agent
    fprintf('\nStarting agent training...\n');
    fprintf('This may take several minutes to hours depending on your system.\n');
    fprintf('Training progress will be displayed in a separate window.\n\n');
    
    tic;
    trainingStats = train(agent, env, trainOpts);
    trainingTime = toc;
    
    fprintf('\n✓ Training completed in %.2f minutes\n', trainingTime/60);
    fprintf('Final average reward: %.2f\n', trainingStats.AverageReward(end));
    
    % Simulate the trained agent
    fprintf('\nSimulating trained agent...\n');
    simOptions = rlSimulationOptions('MaxSteps', ceil(5 / Ts));
    experience = sim(env, agent, simOptions);
    
    % Display results
    totalReward = sum(experience.Reward);
    fprintf('✓ Simulation completed\n');
    fprintf('Total reward: %.2f\n', totalReward);
    
    % Plot results
    plotTrainingResults(trainingStats, experience);
    
    % Save the trained agent
    save('trainedQubeAgent.mat', 'agent', 'trainingStats', 'experience');
    fprintf('\n✓ Trained agent saved to trainedQubeAgent.mat\n');
    
    fprintf('\nTraining and simulation completed successfully!\n');
    fprintf('================================================\n');
end

function in = resetFunction(in)
    % RESETFUNCTION - Reset function for randomizing initial conditions
    %
    % This function sets random initial conditions for each training episode
    % to improve the agent's robustness and generalization.
    
    % Random initial pendulum angle (small deviation from vertical)
    theta0 = randn * pi / 8;  % ±22.5 degrees
    
    % Random initial motor arm angle (small deviation from reference)
    phi0 = randn * pi / 8;   % ±22.5 degrees
    
    % Random initial angular velocities (small values)
    dtheta0 = randn * 0.1;   % ±0.1 rad/s
    dphi0 = randn * 0.1;     % ±0.1 rad/s
    
    % Set the variables in the simulation
    in = setVariable(in, 'theta0', theta0);
    in = setVariable(in, 'phi0', phi0);
    in = setVariable(in, 'dtheta0', dtheta0);
    in = setVariable(in, 'dphi0', dphi0);
end

function agent = createTD3Agent(obsInfo, actInfo, Ts)
    % CREATETD3AGENT - Create a TD3 (Twin Delayed Deep Deterministic) agent
    %
    % This function creates a TD3 agent with separate actor and critic networks
    % for continuous control of the QUBE pendulum system.
    
    % Define the critic network (Q-function approximator)
    % State path
    statePath = [
        featureInputLayer(obsInfo.Dimension(1), 'Normalization', 'none', 'Name', 'state')
        fullyConnectedLayer(400, 'Name', 'fc1')
        reluLayer('Name', 'relu1')
        fullyConnectedLayer(300, 'Name', 'fc2')
        reluLayer('Name', 'relu2')];
    
    % Action path
    actionPath = [
        featureInputLayer(actInfo.Dimension(1), 'Normalization', 'none', 'Name', 'action')
        fullyConnectedLayer(300, 'Name', 'fc3')];
    
    % Common path
    commonPath = [
        additionLayer(2, 'Name', 'add')
        reluLayer('Name', 'relu3')
        fullyConnectedLayer(1, 'Name', 'fc4')];
    
    % Combine paths
    criticNetwork = layerGraph(statePath);
    criticNetwork = addLayers(criticNetwork, actionPath);
    criticNetwork = addLayers(criticNetwork, commonPath);
    criticNetwork = connectLayers(criticNetwork, 'fc2', 'add/in1');
    criticNetwork = connectLayers(criticNetwork, 'fc3', 'add/in2');
    
    % Create critic representation
    criticOptions = rlRepresentationOptions(...
        'LearnRate', 1e-3, ...
        'GradientThreshold', 1);
    critic = rlQValueFunction(criticNetwork, obsInfo, actInfo, ...
        'Observation', {'state'}, 'Action', {'action'}, criticOptions);
    
    % Define the actor network (policy approximator)
    actorNetwork = [
        featureInputLayer(obsInfo.Dimension(1), 'Normalization', 'none', 'Name', 'state')
        fullyConnectedLayer(400, 'Name', 'fc1')
        reluLayer('Name', 'relu1')
        fullyConnectedLayer(300, 'Name', 'fc2')
        reluLayer('Name', 'relu2')
        fullyConnectedLayer(actInfo.Dimension(1), 'Name', 'fc3')
        tanhLayer('Name', 'tanh1')];
    
    % Create actor representation
    actorOptions = rlRepresentationOptions(...
        'LearnRate', 1e-4, ...
        'GradientThreshold', 1);
    actor = rlContinuousDeterministicActor(actorNetwork, obsInfo, actInfo, ...
        'Observation', {'state'}, 'Action', {'tanh1'}, actorOptions);
    
    % Set agent options
    agentOptions = rlTD3AgentOptions(...
        'SampleTime', Ts, ...
        'TargetSmoothFactor', 5e-3, ...
        'ExperienceBufferLength', 1e6, ...
        'MiniBatchSize', 256, ...
        'DiscountFactor', 0.99, ...
        'ExplorationModel', rl.option.GaussianActionNoise(...
            'Mean', 0, ...
            'StandardDeviation', 0.1, ...
            'StandardDeviationDecayRate', 1e-6));
    
    % Create the agent
    agent = rlTD3Agent(actor, critic, agentOptions);
end

function plotTrainingResults(trainingStats, experience)
    % PLOTTRAININGRESULTS - Plot training and simulation results
    
    figure('Name', 'QUBE Pendulum Training Results', 'Position', [100, 100, 1200, 800]);
    
    % Plot 1: Training progress
    subplot(2, 2, 1);
    plot(trainingStats.EpisodeIndex, trainingStats.EpisodeReward, 'b-', 'LineWidth', 1);
    hold on;
    plot(trainingStats.EpisodeIndex, trainingStats.AverageReward, 'r-', 'LineWidth', 2);
    xlabel('Episode');
    ylabel('Reward');
    title('Training Progress');
    legend('Episode Reward', 'Average Reward', 'Location', 'best');
    grid on;
    
    % Plot 2: Episode length
    subplot(2, 2, 2);
    plot(trainingStats.EpisodeIndex, trainingStats.EpisodeSteps, 'g-', 'LineWidth', 1);
    xlabel('Episode');
    ylabel('Steps');
    title('Episode Length');
    grid on;
    
    % Plot 3: Simulation results
    subplot(2, 2, 3);
    plot(experience.Observation.observations.Data(1, :), 'b-', 'LineWidth', 2);
    hold on;
    plot(experience.Observation.observations.Data(2, :), 'r-', 'LineWidth', 2);
    xlabel('Time Step');
    ylabel('Angle (rad)');
    title('Simulation: Pendulum and Motor Angles');
    legend('Pendulum Angle (θ)', 'Motor Angle (φ)', 'Location', 'best');
    grid on;
    
    % Plot 4: Action sequence
    subplot(2, 2, 4);
    plot(experience.Action.actions.Data, 'k-', 'LineWidth', 2);
    xlabel('Time Step');
    ylabel('Voltage Command');
    title('Simulation: Motor Voltage Command');
    grid on;
    
    sgtitle('QUBE Pendulum Reinforcement Learning Results');
end
