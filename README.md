fluid_table_tennis
===============

My first trial on a RL problem for the game Fluid Plasma Pong. 
I'm using the HTML replication of the Plasma Pong made by http://anirudhjoshi.github.com/fluid_table_tennis.

This work is currently in progress.
Using PPO, I train and compare different policy model : CnnPolicy (pixed based), MlpPolicy (game state based), random (baseline)

=============================
To train the model
Start local server from src/ : python -m http.server 8000 --bind 127.0.0.1
Run ppo_training.py
