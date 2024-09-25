# CartPole version one

This repository contains a solution for the CartPole-v1 problem of the gymnasium library with **Deep Reinforcement Learning**.<br>
The project focuses on two major algorithms: ```DQN``` and ```SARSA```, and evaluates their performance in solving the **CartPole-v1** problem.

## Problem Description

The **CartPole-v1** environment involves balancing a pole on a cart that moves along a frictionless track. The agent's task is to prevent the pole from falling by applying forces to the cart. Below are the key features and conditions:

| **Property**             | **Details**                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| **Goal**                 | Keep the pole balanced for as long as possible.                             |
| **Reward**               | +1 for every time step the pole is balanced.                                |
| **Termination Conditions**| Pole angle exceeds 12° or cart position exceeds the track boundaries.       |
| **Maximum Episode Length**| 500 time steps.  |

### State Variables:
The state is represented by four variables:
``` 
Cart Position (x)
Cart Velocity (ẋ)
Pole Angle (θ)
Pole Angular Velocity (θ̇)
```
### Actions:
```
0 Push cart to the left.
1 Push cart to the right.
```

The task is episodic, and a well-trained agent aims to keep the pole balanced for the maximum reward of 500. For more information, visit the [CartPole-v1 documentation](https://gymnasium.farama.org/environments/classic_control/cart_pole/).

## Assignment Objectives

1. **Algorithm Implementation**:
   - Implement **DQN** and **SARSA** algorithms and train agents separately.
   - Compare the success or failure of each algorithm, focusing on their convergence speed, reward maximization, and overall performance.

2. **Evaluation Metrics**:
   - Plot graphs for:
     - **Rewards**: The rewards earned by the agent.
     - **Loss (for DQN)**: The error in predicting future rewards.
     - **Epsilon Decay**: The exploration-exploitation trade-off.

3. **Hyperparameter Tuning**:
   - Train the DQN model with at least three different sets of hyperparameters and report the results in a table format.
   - Analyze the impact of these hyperparameters on the performance of the models.

4. **Testing**:
   - Test functions are provided in the code, and all final weights are saved in the respective files for reproducibility.

## Hyperparameters

Below is the table for the hyperparameters used in the **DQN** and **SARSA** algorithms based on the three stages of experimentation:

| **Stage**       | **Learning Rate** | **Discount Factor** | **Update Frequency** |
|-----------------|-------------------|---------------------|----------------------|
| **DQN (Stage 1)** | 2.3e-2            | 0.9                 | 8                    |
| **DQN (Stage 2)** | 2e-2              | 0.98                | 5                    |
| **DQN (Stage 3)** | 1e-3              | 0.96                | 15                   |
| **SARSA (Stage 1)** | 2.5e-2          | 0.95                | 10                   |
| **SARSA (Stage 2)** | 1e-2            | 0.99                | 7                    |
| **SARSA (Stage 3)** | 5e-3            | 0.92                | 12                   |

Learning Rate	2.3e-2	2e-2	1e-3	2.3e-2
Discount Factor	0.9	0.98	0.96	0.93
Update Frequency	8	5	15	10

## Experimentation

### Part 1: Comparing DQN and SARSA
1. **Convergence**:  
   DQN consistently outperforms SARSA by converging faster and achieving higher rewards. SARSA, being an **on-policy** algorithm, is less effective in utilizing experience compared to **off-policy** DQN.
   
   - **DQN**: Uses experiences multiple times (replay buffer) and selects actions based on a max Q-value approach, leading to better performance.
   - **SARSA**: Follows the policy directly, which results in slower convergence and noisier performance.

2. **Results Summary**:
   - DQN converges to optimal rewards, while SARSA struggles with noisy updates.

**Reward Comparison Plot**:
![Reward Comparison](path-to-your-plot/reward_comparison.png)

**Loss Comparison (DQN only)**:
![Loss Comparison](path-to-your-plot/loss_comparison.png)

### Part 2: Boltzmann vs Epsilon-Greedy Exploration
1. **Boltzmann Exploration**:  
   Instead of using epsilon-greedy for exploration, Boltzmann exploration was implemented. In this strategy, temperature parameters control the randomness of action selection.

2. **Parameters**:
   - **Temperature**: Controls exploration intensity.
   - **Decay Rate**: Determines how fast the temperature reduces.

3. **Results**:
   - Boltzmann temperature control led to faster convergence compared to epsilon-greedy.
   - Hyperparameter tuning further accelerated convergence, with optimal parameters leading to early stopping at around 1500 episodes.

**Boltzmann vs Epsilon-Greedy Exploration Plot**:
![Exploration Comparison](path-to-your-plot/exploration_comparison.png)

### Part 3: Hyperparameter Tuning

| Parameter               | Run 1   | Run 2  | Run 3   | Optimal |
|-------------------------|---------|--------|---------|---------|
| **Learning Rate**        | 2.3e-2  | 2e-2   | 1e-3    | 2.3e-2  |
| **Discount Factor**      | 0.9     | 0.98   | 0.96    | 0.93    |
| **Update Frequency**     | 8       | 5      | 15      | 10      |

**Hyperparameter Tuning Results**:
![Hyperparameter Tuning Plot](path-to-your-plot/hyperparameter_tuning.png)

## Conclusion

This project demonstrates the advantages of **DQN** over **SARSA** in reinforcement learning tasks, particularly when dealing with environments that require efficient exploration and experience replay. Boltzmann exploration proved more effective than epsilon-greedy, especially when tuned correctly.

## Installation

To run the code:
1. Install the required libraries:
```
pip install gymnasium
```
2. Clone the repository:
```
git clone https://github.com/navidadkhah/CartPole-V1
cd CartPole-V1
```
3. Run the training scripts for DQN and SARSA.

## License 
This project is under the MIT License, and I’d be thrilled if you use and improve my work!
