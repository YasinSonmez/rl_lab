function setup_environment()
    % SETUP_ENVIRONMENT - Setup function for QUBE Pendulum RL environment
    %
    % This function checks system requirements, sets up paths, and prepares
    % the environment for training reinforcement learning agents on the
    % Quanser QUBE pendulum system.
    %
    % Usage:
    %   setup_environment()
    
    fprintf('QUBE Pendulum RL Environment Setup\n');
    fprintf('==================================\n\n');
    
    %% Check MATLAB Version
    fprintf('Checking MATLAB version...\n');
    matlabVersion = version('-release');
    matlabYear = str2double(matlabVersion(1:4));
    
    if matlabYear < 2020
        warning('MATLAB R2020b or later is recommended for optimal performance');
    else
        fprintf('✓ MATLAB version: %s (compatible)\n', matlabVersion);
    end
    
    %% Check Required Toolboxes
    fprintf('\nChecking required toolboxes...\n');
    
    requiredToolboxes = {
        'Reinforcement Learning Toolbox', 'rl';
        'Deep Learning Toolbox', 'nnet';
        'Simulink', 'simulink';
        'Simscape Multibody', 'sm';
        'Simscape Electrical', 'powerlib'
    };
    
    missingToolboxes = {};
    
    for i = 1:size(requiredToolboxes, 1)
        toolboxName = requiredToolboxes{i, 1};
        toolboxId = requiredToolboxes{i, 2};
        
        if license('test', toolboxId)
            fprintf('✓ %s: Available\n', toolboxName);
        else
            fprintf('✗ %s: Missing\n', toolboxName);
            missingToolboxes{end+1} = toolboxName;
        end
    end
    
    if ~isempty(missingToolboxes)
        fprintf('\nWarning: Missing required toolboxes:\n');
        for i = 1:length(missingToolboxes)
            fprintf('  - %s\n', missingToolboxes{i});
        end
        fprintf('\nPlease install missing toolboxes using MATLAB Add-On Explorer\n');
    else
        fprintf('\n✓ All required toolboxes are available\n');
    end
    
    %% Check for Simulink Model
    fprintf('\nChecking for Simulink model...\n');
    
    modelFile = 'rlQubeServo.slx';
    if exist(modelFile, 'file')
        fprintf('✓ Simulink model found: %s\n', modelFile);
    else
        fprintf('✗ Simulink model not found: %s\n', modelFile);
        fprintf('Please ensure the model file is in the current directory\n');
        fprintf('You may need to download it from the MathWorks File Exchange\n');
    end
    
    %% Set Up Paths
    fprintf('\nSetting up MATLAB paths...\n');
    
    % Add current directory to path
    currentDir = pwd;
    addpath(currentDir);
    fprintf('✓ Added current directory to MATLAB path\n');
    
    % Check for GPU availability
    fprintf('\nChecking GPU availability...\n');
    if gpuDeviceCount > 0
        gpu = gpuDevice();
        fprintf('✓ GPU available: %s\n', gpu.Name);
        fprintf('  Memory: %.1f GB\n', gpu.AvailableMemory / 1e9);
        
        % Set GPU memory fraction for training
        if gpu.AvailableMemory > 4e9  % More than 4GB
            fprintf('✓ GPU memory sufficient for training\n');
        else
            fprintf('⚠ GPU memory may be limited for large networks\n');
        end
    else
        fprintf('⚠ No GPU available - training will use CPU\n');
        fprintf('  Consider using GPU for faster training\n');
    end
    
    %% Display System Information
    fprintf('\nSystem Information:\n');
    fprintf('------------------\n');
    fprintf('MATLAB Version: %s\n', version);
    fprintf('Operating System: %s\n', computer);
    % fprintf('Available Memory: %.1f GB\n', memory().MemAvailableAllArrays / 1e9);
    
    if gpuDeviceCount > 0
        fprintf('GPU: %s\n', gpu.Name);
    else
        fprintf('GPU: None detected\n');
    end
    
    %% Create Output Directory
    fprintf('\nSetting up output directory...\n');
    
    outputDir = 'training_results';
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
        fprintf('✓ Created output directory: %s\n', outputDir);
    else
        fprintf('✓ Output directory exists: %s\n', outputDir);
    end
    
    %% Display Usage Instructions
    fprintf('\nUsage Instructions:\n');
    fprintf('-------------------\n');
    fprintf('1. Run the main training function:\n');
    fprintf('   >> trainQubePendulumAgent()\n\n');
    fprintf('2. Or run the example usage script:\n');
    fprintf('   >> example_usage\n\n');
    fprintf('3. For custom training, modify parameters in:\n');
    fprintf('   - trainQubePendulumAgent.m (main function)\n');
    fprintf('   - createTD3Agent() (agent architecture)\n\n');
    
    %% Performance Recommendations
    fprintf('Performance Recommendations:\n');
    fprintf('---------------------------\n');
    
    if gpuDeviceCount > 0
        fprintf('✓ GPU detected - training will be accelerated\n');
    else
        fprintf('⚠ No GPU - consider using cloud computing or GPU-enabled system\n');
    end
    
    % if memory().MemAvailableAllArrays > 8e9
    %     fprintf('✓ Sufficient RAM available for training\n');
    % else
    %     fprintf('⚠ Limited RAM - consider reducing batch size or network size\n');
    % end
    
    fprintf('\nSetup completed successfully!\n');
    fprintf('==================================\n');
    
    %% Save Setup Information
    setupInfo = struct();
    setupInfo.matlabVersion = version;
    setupInfo.toolboxes = requiredToolboxes;
    setupInfo.gpuAvailable = gpuDeviceCount > 0;
    % setupInfo.memoryGB = memory().MemAvailableAllArrays / 1e9;
    setupInfo.timestamp = datetime('now');
    
    save('setup_info.mat', 'setupInfo');
    fprintf('\n✓ Setup information saved to setup_info.mat\n');
end
