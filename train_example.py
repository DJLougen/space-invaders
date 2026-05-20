"""
Space Invaders RL Training Example
Simple DQN training loop using stable-baselines3.
"""
import argparse
import sys

from space_invaders_gym import SpaceInvadersEnv


def train_sb3(total_timesteps: int = 100_000, render: bool = False):
    """Train with stable-baselines3 DQN."""
    try:
        from stable_baselines3 import DQN
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError:
        print("stable-baselines3 not installed. Run: pip install stable-baselines3[extra]")
        sys.exit(1)

    render_mode = "human" if render else None

    def make_env():
        return SpaceInvadersEnv(render_mode=render_mode, frame_skip=4)

    env = DummyVecEnv([make_env])

    model = DQN(
        "CnnPolicy" if False else "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=10_000,
        learning_starts=1_000,
        batch_size=32,
        gamma=0.99,
        target_update_interval=500,
        train_freq=4,
        gradient_steps=1,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        verbose=1,
    )

    print(f"Training DQN for {total_timesteps} steps...")
    model.learn(total_timesteps=total_timesteps)
    model.save("dqn_space_invaders")
    print("Model saved to dqn_space_invaders.zip")

    # Test the trained model
    print("\nTesting trained model...")
    obs, _ = env.reset()
    total_reward = 0
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward[0]
        if terminated[0] or truncated[0]:
            break

    print(f"Test episode reward: {total_reward:.2f}, score: {info[0]['score']}")
    env.close()


def train_random(total_steps: int = 1000, render: bool = False):
    """Baseline: random agent for comparison."""
    render_mode = "human" if render else "rgb_array"
    env = SpaceInvadersEnv(render_mode=render_mode, frame_skip=4)

    obs, info = env.reset()
    total_reward = 0
    episodes = 0
    best_score = 0

    for step in range(total_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if render:
            env.render()

        if terminated or truncated:
            episodes += 1
            if info["score"] > best_score:
                best_score = info["score"]
            print(
                f"Episode {episodes}: score={info['score']}, "
                f"reward={total_reward:.1f}, best={best_score}"
            )
            obs, info = env.reset()
            total_reward = 0

    print(f"\nRandom baseline: {episodes} episodes, best score: {best_score}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Space Invaders agent")
    parser.add_argument(
        "--method",
        choices=["dqn", "random"],
        default="random",
        help="Training method (default: random)",
    )
    parser.add_argument(
        "--steps", type=int, default=5000, help="Total training steps (default: 5000)"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render the game during training"
    )
    args = parser.parse_args()

    if args.method == "dqn":
        train_sb3(total_timesteps=args.steps, render=args.render)
    else:
        train_random(total_steps=args.steps, render=args.render)
