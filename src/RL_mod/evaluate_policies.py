"""Compare a trained PPO policy with a random policy in Plasma Pong."""

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

try:
    from .gym_env import PlasmaPongEnv
except ImportError:
    from gym_env import PlasmaPongEnv


def run_policy(env: PlasmaPongEnv, policy: PPO | None, episodes: int) -> dict[str, float]:
    """Run a policy and return averages over complete evaluation episodes."""
    rewards = []
    player_lives = []
    opponent_lives = []
    collisions = []
    wins = 0

    for episode in range(episodes):
        observation, _ = env.reset(seed=episode)
        episode_reward = 0.0
        metrics = {"player_life": 5, "opponent_life": 5, "player_collisions": 0}
        terminated = truncated = False

        while not (terminated or truncated):
            if policy is None:
                action = env.action_space.sample()
            else:
                action, _ = policy.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            metrics = info

        rewards.append(episode_reward)
        player_lives.append(metrics["player_life"])
        opponent_lives.append(metrics["opponent_life"])
        collisions.append(metrics["player_collisions"])
        if metrics["opponent_life"] == 0 and metrics["player_life"] > 0:
            wins += 1

    return {
        "average episode reward": float(np.mean(rewards)),
        "average player life remaining": float(np.mean(player_lives)),
        "average opponent lives remaining": float(np.mean(opponent_lives)),
        "average player-ball collisions": float(np.mean(collisions)),
        "number of wins": wins,
    }


def print_results(name: str, results: dict[str, float]) -> None:
    print(f"\n{name} ({int(results['number of wins'])} wins)")
    print(f"  average episode reward: {results['average episode reward']:.3f}")
    print(f"  average player life remaining: {results['average player life remaining']:.3f}")
    print(f"  average opponent lives remaining: {results['average opponent lives remaining']:.3f}")
    print(f"  average player-ball collisions: {results['average player-ball collisions']:.3f}")
    print(f"  number of wins: {int(results['number of wins'])}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ppo_plasma_pong_state", help="Saved PPO model path")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    env = PlasmaPongEnv(render_mode="human", observation_mode="state")
    try:
        print_results("Random policy", run_policy(env, None, args.episodes))
        model_path = Path(args.model)
        if model_path.with_suffix(".zip").exists() or model_path.exists():
            model = PPO.load(args.model, env=env)
            print_results("Trained PPO policy", run_policy(env, model, args.episodes))
        else:
            print(f"\nTrained PPO policy skipped: {args.model!r} was not found.")
    finally:
        env.close()


if __name__ == "__main__":
    main()