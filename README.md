# Space Invaders

A classic Space Invaders game with both a browser version and a Gymnasium RL environment for training AI agents.

**[Play in browser](https://djlougen.github.io/space-invaders/)**

## Features

- 5 rows of invaders across 3 types (10/20/30 pts)
- Invaders speed up as their numbers thin out
- Mystery UFO with random bonus (50–300 pts)
- 4 destructible shields that degrade from both sides' fire
- Particle effects on kills
- Progressive difficulty across levels
- Shields carry over between levels

## Gymnasium RL Environment

Train reinforcement learning agents with a Gymnasium-compatible environment.

### Installation

```bash
pip install -r requirements.txt
```

For training with stable-baselines3:

```bash
pip install stable-baselines3[extra]
```

### Quick Start

```python
from space_invaders_gym import SpaceInvadersEnv

env = SpaceInvadersEnv(render_mode="rgb_array")
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # Random agent
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### Action Space

Discrete(6):

| Action | Meaning |
|--------|---------|
| 0 | NOOP |
| 1 | LEFT |
| 2 | RIGHT |
| 3 | SHOOT |
| 4 | LEFT + SHOOT |
| 5 | RIGHT + SHOOT |

### Observation Space

Box(0, 255, (120, 160, 3), dtype=uint8) — RGB pixel frame at 120×160 resolution.

### Rewards

- +0.1 × points scored (killing invaders, UFO bonus)
- -10.0 for losing a life
- -50.0 for game over

### Training Example

```bash
# Random baseline
python train_example.py --method random --steps 5000

# DQN with stable-baselines3
python train_example.py --method dqn --steps 100000 --render
```

### Human Mode Rendering

Watch the agent play in real-time:

```python
env = SpaceInvadersEnv(render_mode="human")
```

Requires pygame: `pip install pygame`

## Browser Game Controls

| Key | Action |
|-----|--------|
| `← →` or `A D` | Move ship |
| `Space` or `↑` | Shoot |
| `P` | Pause |
| `Enter` | Restart (after game over) |

## Running the browser game locally

Open `index.html` in any browser. No server needed.

## License

MIT
