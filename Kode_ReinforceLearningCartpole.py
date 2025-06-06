# %% Import Library & Setup Environment
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os

environment_name = "CartPole-v1"
env = gym.make(environment_name, render_mode="human")

log_path = os.path.join('Training', 'Logs')
save_path = os.path.join('Training', 'Saved Models')

# %% Testing Random Agent
episodes = 10
for episode in range(1, episodes + 1):
    # Update for Gymnasium's reset
    state = env.reset()
    done = False
    score = 0 
    
    while not done:
        env.render()
        action = env.action_space.sample()
        # Update for Gymnasium's step
        n_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated  # Episode ends if either condition is True
        score += reward
    
    print(f'Episode: {episode}, Score: {score}')
env.close()

# %% PPO Model dan Training
env = gym.make(environment_name, render_mode="human")
env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose = 1, tensorboard_log=log_path)
model.learn(total_timesteps=500)


#%% Untuk menyimpan model yang sudah di train
PPO_path = os.path.join('Training', 'Models', 'PPO_model_10000steps')
model.save(PPO_path)

#%% Delete model
del model
# model = PPO.load('PPO_model', env=env)

#%% Load Model
env = DummyVecEnv([lambda: gym.make('CartPole-v1', render_mode="human")])
model = PPO.load('Training/Models/PPO_model_1000steps', env=env)

#%% Evaluasi Model PPO
evaluate_policy(model, env, n_eval_episodes=10, render=True)
env.close()


# %% Manual Testing PPO
episode = 5
for i in range(1, episode + 1):
    score = 0
    env = gym.make('CartPole-v1', render_mode="human")
    obs, _ = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        score += reward
        # Kalau CartPole sudah jatuh / selesai
        if terminated or truncated:
            print(f'Episode {i} done, info:{info}, score: {score}')
            break

    env.close()
    
# %% TensorBoard Path
training_log_path = os.path.join('Logs', 'PPO_10')
# !tensorboard --logdir={training_log_path}
