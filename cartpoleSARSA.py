import pickle
import pygame
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

seed = 2024

class SARSA:
    def __init__(self, hyperparams):
        # for model saving
        self.RL_load_path = hyperparams["RL_load_path"]
        self.save_path = hyperparams["save_path"]
        self.save_interval = hyperparams["save_interval"]

        # set the hyper params of environment
        self.env = hyperparams["env"]
        self.observation_space = hyperparams["observation_space"]
        self.action_space = hyperparams["action_space"]

        # set the hyper params
        self.bin_size = hyperparams["bin_size"]
        self.Learning_rate = hyperparams["learning_rate"]
        self.discount_factor = hyperparams["discount_factor"]
        self.max_episodes = hyperparams["max_episodes"]
        self.max_step = hyperparams["max_step"]
        self.render = hyperparams["render"]
        self.epsilon = hyperparams["epsilon"]
        self.epsilon_decay = hyperparams["epsilon_decay"]
        self.epsilon_min = hyperparams["epsilon_min"]

        # set the bins for convert continuous to discrete
        self.bins = [
            np.linspace(-4.8, 4.8, bin_size),
            np.linspace(-30, 30, bin_size),
            np.linspace(-0.418, 0.418, bin_size),
            np.linspace(-30, 30, bin_size)
        ]

        # create Q table
        self.create_table()

    def create_table(self):
        columns = [self.bin_size] * self.observation_space.shape[0]
        rows = [self.action_space.n]
        self.Q_table = np.zeros(columns + rows)

    def discretize_state(self, state):
        discrete = []
        discrete.append(np.digitize(state[0], self.bins[0]) - 1)
        discrete.append(np.digitize(state[1], self.bins[1]) - 1)
        discrete.append(np.digitize(state[2], self.bins[2]) - 1)
        discrete.append(np.digitize(state[3], self.bins[3]) - 1)
        return tuple(discrete)

    def select_action(self, state):
        # Exploration: epsilon-greedy
        if np.random.random() < self.epsilon:
            return self.action_space.sample()
        # Exploitation: the action is selected based on the Q-values.
        return np.argmax(self.Q_table[state])

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def calculate_loss(self, state, next_state, action, next_action, reward, total_loss):
        td_error = reward + self.discount_factor * self.Q_table[next_state + (next_action,)] - self.Q_table[
            state + (action,)]
        self.Q_table[
            state + (action,)] += self.Learning_rate * td_error
        total_loss += td_error ** 2
        return total_loss

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.Q_table, f)

    def train(self):
        reward_history = []
        loss_history = []
        epsilon_history = []
        for episode in range(1, self.max_episodes + 1):
            state, _ = self.env.reset(seed=seed)
            state = self.discretize_state(state)
            action = self.select_action(state)
            episode_reward = 0
            total_loss = 0
            iterations = 0
            done = False

            while not done and iterations != self.max_step:
                next_state_raw, reward, done, truncation, _ = self.env.step(action)
                next_state = self.discretize_state(next_state_raw)
                next_action = self.select_action(next_state)

                total_loss = self.calculate_loss(state, next_state, action, next_action, reward, total_loss)

                state = next_state
                action = next_action
                episode_reward += reward
                iterations += iterations

            reward_history.append(episode_reward)
            epsilon_history.append(self.epsilon)
            loss_history.append(total_loss)
            self.update_epsilon()
            if episode % self.save_interval == 0:
                self.save(self.save_path + '_' + f'{episode}' + '.pth')
                if episode != self.max_episodes:
                    self.plot_training(episode, reward_history, loss_history, epsilon_history)
                print('\n~~~~~~Interval Save: Model saved.\n')

            result = (f"Episode: {episode}, "
                      f"Raw Reward: {episode_reward:.2f}, ")
            print(result)

    def test(self, max_episodes):
        """
        Reinforcement learning policy evaluation.
        """

        # Load the weights of the test_network
        with open(self.save_path, 'rb') as f:
            self.Q_table = pickle.load(f)

        total_rewards = []
        for episode in range(1, self.max_episodes + 1):
            state, _ = self.env.reset(seed=seed)
            state = self.discretize_state(state)
            episode_reward = 0
            iterations = 0
            done = False

            while not done and iterations != self.max_step:
                action = np.argmax(self.Q_table[state])
                next_state_raw, reward, done, truncation, _ = self.env.step(action)
                next_state = self.discretize_state(next_state_raw)

                state = next_state
                episode_reward += reward

                iterations += iterations

            total_rewards.append(episode_reward)
            print(f"Test Episode: {episode}, Reward: {episode_reward}")

    def plot_training(self, episode, reward_history, loss_history, epsilon_history):
        # Calculate the Simple Moving Average (SMA) with a window size of 50
        sma = np.convolve(reward_history, np.ones(50) / 50, mode='valid')

        plt.figure()
        plt.title("Rewards")
        plt.plot(reward_history, label='Raw Reward', color='#F6CE3B', alpha=1)
        plt.plot(sma, label='SMA 50', color='#385DAA')
        plt.xlabel("Episode")
        plt.ylabel("Rewards")
        plt.legend()

        # Only save as file if last episode
        if episode == self.max_episodes:
            plt.savefig('./cartPole/SARSA/reward_plot.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.grid(True)
        plt.show()
        plt.clf()
        plt.close()

        plt.figure()
        plt.title("Loss")
        plt.plot(loss_history, label='Loss', color='#CB291A', alpha=1)
        plt.xlabel("Episode")
        plt.ylabel("Loss")

        # Only save as file if last episode
        if episode == self.max_episodes:
            plt.savefig('./cartPole/SARSA/Loss_plot.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.grid(True)
        plt.show()

        plt.figure()
        plt.title("epsilon")
        plt.plot(epsilon_history, label='epsilon', color='#0000ff', alpha=1)
        plt.xlabel("Episode")
        plt.ylabel("epsilon")

        # Only save as file if last episode
        if episode == self.max_episodes:
            plt.savefig('./cartPole/SARSA/Temperature.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    # Parameters:
    train_mode = True
    render = not train_mode
    env = gym.make('CartPole-v1', render_mode="human" if render else None)
    bin_size = 100

    RL_hyperparams = {
        "RL_load_path": f'./cartPole/SARSA/final_weights' + '_' + '5000' + '.pth',
        "save_path": f'./cartPole/SARSA/final_weights',
        "save_interval": 2000,
        "env": env,
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "render": render,
        "bin_size": 100,
        "learning_rate": 0.25,
        "discount_factor": 0.995,
        "max_episodes": 8000 if train_mode else 5,
        "max_step": 500,
        "epsilon": 0.98 if train_mode else -1,
        "epsilon_min": 0.2,
        "epsilon_decay": 0.98,
    }

    if train_mode:
        SARSA(RL_hyperparams).train()
    else:
        SARSA(RL_hyperparams).test(RL_hyperparams["max_episodes"])
