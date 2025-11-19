# QUBE Pendulum Reinforcement Learning

This project implements multiple reinforcement learning algorithms to control the Quanser QUBE-Servo 2 inverted pendulum system using MATLAB/Simulink.

## Overview

The repository contains implementations of several RL algorithms:
- **TD3** (Twin Delayed Deep Deterministic): Baseline continuous control agent
- **SAC** (Soft Actor-Critic): Sample-efficient continuous control with entropy regularization
- **Tabular Q-Learning**: Discrete state-action space approach using discretized Simulink model
- **PPO** (Proximal Policy Optimization): Memory-efficient on-policy algorithm

All agents learn to swing up and balance the inverted pendulum by controlling motor voltage.

## Requirements

### Software
- **MATLAB R2025a** (recommended)
- Operating System: Windows, macOS, or Linux

### Required MATLAB Toolboxes
Install these toolboxes via MATLAB Add-On Explorer or `matlab.addons.install`:

1. **Reinforcement Learning Toolbox** - Core RL algorithms and training
2. **Simulink** - Environment modeling and simulation
3. **Simscape Multibody** - Mechanical system dynamics (pendulum physics)
4. **Simscape Electrical** - DC motor modeling

### Optional (Recommended)
- **Parallel Computing Toolbox** - Parallel training for SAC and PPO (4-8× speedup)
- **GPU Computing** - CUDA-enabled GPU for faster network training

### Hardware (Optional)
- Quanser QUBE-Servo 2 system for real hardware deployment
- USB interface for hardware connection

## Installation

### 1. Install MATLAB
Download and install [MATLAB R2024a](https://www.mathworks.com/products/matlab.html) or later.

### 2. Install Required Toolboxes
**Option A: Using MATLAB Add-On Explorer (GUI)**
```matlab
% In MATLAB Command Window:
matlab.addons.explorerapp
```
Search and install:
- Reinforcement Learning Toolbox
- Simulink
- Simscape Multibody
- Simscape Electrical
- Parallel Computing Toolbox (optional)

**Option B: Using Command Line**
```matlab
% Check which toolboxes are installed
ver

% Install toolboxes programmatically (requires license)
% Note: Replace with actual product identifiers from your license
matlab.addons.install('reinforcement-learning')
matlab.addons.install('deep-learning')
```

### 3. Clone Repository

### 4. Verify Installation
Run the setup script in MATLAB:
```matlab
setup_environment
```

This will:
- Check all required toolboxes are installed
- Verify MATLAB version compatibility
- Add necessary paths to MATLAB search path
- Test GPU availability (if applicable)
- Display system configuration summary

**Expected Output:**
```
Checking MATLAB version... ✓ R2024a
Checking toolboxes...
  ✓ Reinforcement Learning Toolbox
  ✓ Deep Learning Toolbox
  ✓ Simulink
  ✓ Simscape Multibody
  ✓ Simscape Electrical
  ✓ Control System Toolbox
  ✓ Parallel Computing Toolbox
GPU: NVIDIA GeForce RTX 3080 (available)
Setup complete!
```

## Repository Structure

```
rl_lab/
├── setup_environment.m              # Verify installation and setup paths
├── trainQubePendulumAgent.m         # TD3 baseline trainer
├── example_usage.m                  # Load and simulate trained agents
├── README.md                        # This file
└── rl_quanser_cube/
    ├── sac.m                        # SAC algorithm trainer
    ├── tabular_q.m                  # Tabular Q-learning trainer
    ├── ppo.m                        # PPO algorithm trainer
    ├── rlQubeServo.slx              # Simulink model (continuous)
    └── rlQubeServo_discrete.slx     # Simulink model (discretized for Q-learning)
```

## Quick Start

### 1. Verify Installation
```matlab
% In MATLAB, navigate to project directory
cd /path/to/rl_lab

% Run setup script
setup_environment
```

### 2. Train an Agent

**Option A: Train SAC Agent (Recommended - Fast & Sample Efficient)**
```matlab
% Navigate to algorithm directory
cd rl_quanser_cube

% Train SAC agent (uses parallel training by default)
sac
% Output: Saves trained agent to 'sacAgent.mat'
```

**Option B: Train Tabular Q-Learning Agent**
```matlab
cd rl_quanser_cube
tabular_q
% Output: Saves trained agent to 'tabularQAgent.mat'
```

**Option C: Train PPO Agent (Memory-Efficient)**
```matlab
cd rl_quanser_cube
ppo
% Output: Saves trained agent to 'ppoAgent.mat'
```

**Option D: Train TD3 Agent (Original Baseline)**
```matlab
% From project root
trainQubePendulumAgent()
% Output: Saves trained agent to 'trainedQubeAgent.mat'
```

### 3. Load and Test Trained Agent
```matlab
% Use the example_usage script
example_usage

% Or manually:
load('rl_quanser_cube/sacAgent.mat', 'agent');  % or ppoAgent.mat, etc.

% Recreate environment
mdl = 'rlQubeServo';
open_system(mdl);
obsInfo = rlNumericSpec([7 1]);
actInfo = rlNumericSpec([1 1], 'UpperLimit', 1, 'LowerLimit', -1);
env = rlSimulinkEnv(mdl, [mdl '/RL Agent'], obsInfo, actInfo);

% Simulate
simOpts = rlSimulationOptions('MaxSteps', ceil(5/0.01));
experience = sim(env, agent, simOpts);
fprintf('Total reward: %.2f\n', sum(experience.Reward));
```

## System Description

### State Space (Observations)
The agent receives 7 observations:
- `theta`: Pendulum angle (rad)
- `phi`: Motor arm angle (rad)  
- `dtheta`: Pendulum angular velocity (rad/s)
- `dphi`: Motor arm angular velocity (rad/s)
- `cos(theta)`: Cosine of pendulum angle
- `sin(theta)`: Sine of pendulum angle
- `cos(phi)`: Cosine of motor arm angle

### Action Space
- Single continuous action: Motor voltage command
- Range: [-1, 1] (scaled to [-12V, 12V])
- Sample time: 0.005 seconds

### Reward Function
The reward function encourages:
- Keeping the pendulum upright (θ ≈ 0)
- Maintaining motor arm at reference position (φ ≈ 0)
- Minimizing control effort
- Penalizing large deviations from equilibrium



## References

- [MathWorks Documentation: Train Agents to Control Quanser QUBE Pendulum](https://www.mathworks.com/help/reinforcement-learning/ug/train-agents-to-control-quanser-qube-pendulum.html)
- [Quanser QUBE-Servo 2 Documentation](https://www.quanser.com/products/qube-servo-2/)
- [MATLAB Reinforcement Learning Toolbox](https://www.mathworks.com/products/reinforcement-learning.html)

## License

This project is provided as an educational example. Please refer to MATLAB and Quanser licensing terms for commercial use.
