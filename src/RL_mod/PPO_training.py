"""Main file to train the PPO agent on the Plasma Pong environment."""

from stable_baselines3 import PPO

from gym_env import PlasmaPongEnv

env = PlasmaPongEnv(render_mode="human")
model = PPO("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
# model.save("ppo_plasma_pong")
input("Press Enter to close...")  # Keeps the script alive until you press Enter
env.close()
