"""Main file to train the PPO agent on the Plasma Pong environment."""

from stable_baselines3 import PPO

try:
    from .gym_env import PlasmaPongEnv
except ImportError:
    from gym_env import PlasmaPongEnv

env = PlasmaPongEnv(render_mode="human", observation_mode="state")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
model.save("ppo_plasma_pong_state")
# Keeps the script alive until you press Enter
input("Press Enter to close...")
env.close()
