"""
Space Invaders Gym Environment
Gymnasium-compatible RL environment with pure numpy rendering.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import random


@dataclass
class Entity:
    x: float
    y: float
    w: float
    h: float
    alive: bool = True


@dataclass
class Bullet:
    x: float
    y: float
    w: float
    h: float
    vy: float


@dataclass
class Invader(Entity):
    inv_type: int = 1  # 1, 2, or 3
    points: int = 10
    frame: int = 0


@dataclass
class ShieldBlock:
    x: float
    y: float
    w: float
    h: float
    hp: int = 3


@dataclass
class UFO:
    x: float
    y: float
    vx: float
    alive: bool = True


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    color: Tuple[int, int, int]
    size: int


class SpaceInvadersEnv(gym.Env):
    """Space Invaders reinforcement learning environment."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        obs_width: int = 160,
        obs_height: int = 120,
        frame_skip: int = 4,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.obs_width = obs_width
        self.obs_height = obs_height
        self.frame_skip = frame_skip

        # Game dimensions (internal)
        self.W = 800
        self.H = 600

        # Observation: RGB image
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(obs_height, obs_width, 3), dtype=np.uint8
        )

        # Actions: NOOP, LEFT, RIGHT, SHOOT, LEFT+SHOOT, RIGHT+SHOOT
        self.action_space = spaces.Discrete(6)

        # Game state
        self.player = None
        self.invaders = []
        self.player_bullets = []
        self.enemy_bullets = []
        self.shields = []
        self.particles = []
        self.ufo = None
        self.ufo_timer = 0.0

        self.score = 0
        self.lives = 3
        self.level = 1
        self.game_over = False

        self.invader_dir = 1
        self.invader_speed = 30.0
        self.invader_shoot_chance = 0.002
        self.need_drop = False

        self.shoot_cooldown = 0.0
        self.frame_count = 0

        # Rendering
        self._screen = None
        self._clock = None

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.score = 0
        self.lives = 3
        self.level = 1
        self.game_over = False

        self.player = Entity(x=self.W / 2, y=self.H - 50, w=40, h=20)
        self._create_invaders()
        self._create_shields()
        self.player_bullets = []
        self.enemy_bullets = []
        self.particles = []
        self.ufo = None
        self.ufo_timer = 0.0

        self.invader_dir = 1
        self.invader_speed = 30.0
        self.invader_shoot_chance = 0.002
        self.need_drop = False
        self.shoot_cooldown = 0.0
        self.frame_count = 0

        obs = self._get_obs()
        info = {"score": self.score, "lives": self.lives, "level": self.level}
        return obs, info

    def step(self, action: int):
        reward = 0.0
        dt = 1.0 / 30.0  # Fixed timestep

        for _ in range(self.frame_skip):
            if self.game_over:
                break

            old_score = self.score
            self.frame_count += 1
            old_lives = self.lives

            self._update_player(action, dt)
            self._update_bullets(dt)
            self._update_invaders(dt)
            self._update_ufo(dt)
            self._update_particles(dt)

            # Reward shaping
            reward += (self.score - old_score) * 0.1  # Points
            if self.lives < old_lives:
                reward -= 10.0  # Death penalty
            if self.game_over:
                reward -= 50.0  # Game over penalty

        obs = self._get_obs()
        info = {"score": self.score, "lives": self.lives, "level": self.level}

        terminated = self.game_over
        truncated = False

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_frame()
        return None

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self._screen = None
            self._clock = None

    # --- Private methods ---

    def _create_invaders(self):
        self.invaders = []
        rows, cols = 5, 11
        start_x, start_y = 80, 60
        gap_x, gap_y = 50, 40

        types = [3, 2, 2, 1, 1]
        points = [30, 20, 20, 10, 10]

        for r in range(rows):
            for c in range(cols):
                self.invaders.append(
                    Invader(
                        x=start_x + c * gap_x,
                        y=start_y + r * gap_y,
                        w=30,
                        h=20,
                        inv_type=types[r],
                        points=points[r],
                        frame=0,
                    )
                )

    def _create_shields(self):
        self.shields = []
        shield_positions = [130, 290, 450, 610]
        pattern = [
            "  XXXXXX  ",
            " XXXXXXXX ",
            "XXXXXXXXXX",
            "XXXXXXXXXX",
            "XXXXXXXXXX",
            "XXX    XXX",
        ]

        for sx in shield_positions:
            blocks = []
            for r, row in enumerate(pattern):
                for c, ch in enumerate(row):
                    if ch == "X":
                        blocks.append(
                            ShieldBlock(x=sx + c * 5, y=self.H - 140 + r * 5, w=5, h=5, hp=3)
                        )
            self.shields.append(blocks)

    def _update_player(self, action: int, dt: float):
        speed = 300.0

        # Decode action
        move_left = action in (1, 4)
        move_right = action in (2, 5)
        shoot = action in (3, 4, 5)

        if move_left:
            self.player.x -= speed * dt
        if move_right:
            self.player.x += speed * dt

        self.player.x = max(self.player.w / 2 + 10, min(self.W - self.player.w / 2 - 10, self.player.x))

        # Shooting
        self.shoot_cooldown -= dt
        if shoot and self.shoot_cooldown <= 0 and len(self.player_bullets) < 3:
            self.player_bullets.append(
                Bullet(x=self.player.x - 2, y=self.player.y - 15, w=4, h=12, vy=-500)
            )
            self.shoot_cooldown = 0.25

    def _update_bullets(self, dt: float):
        # Player bullets
        for i in range(len(self.player_bullets) - 1, -1, -1):
            b = self.player_bullets[i]
            b.y += b.vy * dt

            if b.y + b.h < 0:
                self.player_bullets.pop(i)
                continue

            # Hit invaders
            hit = False
            for inv in self.invaders:
                if not inv.alive:
                    continue
                if self._rect_collide(b.x, b.y, b.w, b.h, inv.x - inv.w/2, inv.y - inv.h/2, inv.w, inv.h):
                    inv.alive = False
                    self.score += inv.points
                    self._spawn_particles(inv.x, inv.y, self._invader_color(inv), 8)
                    hit = True
                    break

            if hit:
                self.player_bullets.pop(i)
                continue

            # Hit UFO
            if self.ufo and self.ufo.alive:
                if self._rect_collide(b.x, b.y, b.w, b.h, self.ufo.x - 25, self.ufo.y - 10, 50, 20):
                    pts = random.choice([50, 100, 150, 200, 300])
                    self.score += pts
                    self._spawn_particles(self.ufo.x, self.ufo.y, (255, 0, 0), 15)
                    self.ufo.alive = False
                    hit = True

            if hit:
                self.player_bullets.pop(i)
                continue

            # Hit shields
            if self._bullet_hits_shields(b, i, self.player_bullets):
                continue

        # Enemy bullets
        for i in range(len(self.enemy_bullets) - 1, -1, -1):
            b = self.enemy_bullets[i]
            b.y += b.vy * dt

            if b.y > self.H:
                self.enemy_bullets.pop(i)
                continue

            # Hit player
            if self._rect_collide(
                b.x, b.y, b.w, b.h,
                self.player.x - self.player.w/2, self.player.y - self.player.h/2,
                self.player.w, self.player.h
            ):
                self.lives -= 1
                self._spawn_particles(self.player.x, self.player.y, (0, 255, 0), 10)
                self.enemy_bullets.pop(i)
                if self.lives <= 0:
                    self.game_over = True
                else:
                    self.player.x = self.W / 2
                continue

            # Hit shields
            if self._bullet_hits_shields(b, i, self.enemy_bullets):
                continue

    def _update_invaders(self, dt: float):
        alive = [inv for inv in self.invaders if inv.alive]
        if not alive:
            self._next_level()
            return

        # Speed scales with fewer invaders
        speed_mult = 1.0 + (55 - len(alive)) * 0.04
        current_speed = self.invader_speed * speed_mult

        if self.need_drop:
            for inv in alive:
                inv.y += 20
            self.invader_dir *= -1
            self.need_drop = False
        else:
            hit_edge = False
            for inv in alive:
                inv.x += current_speed * self.invader_dir * dt
                if inv.x + inv.w/2 >= self.W - 10 or inv.x - inv.w/2 <= 10:
                    hit_edge = True
            if hit_edge:
                self.need_drop = True

        # Animation
        frame = (self.frame_count // 15) % 2
        for inv in alive:
            inv.frame = frame

        # Shooting
        for inv in alive:
            if random.random() < self.invader_shoot_chance * dt * 60:
                # Only bottom invaders shoot
                col = [i for i in self.invaders if i.alive and abs(i.x - inv.x) < 5]
                bottom = max(col, key=lambda i: i.y)
                if bottom is inv:
                    self.enemy_bullets.append(
                        Bullet(x=inv.x - 2, y=inv.y + inv.h/2, w=4, h=10, vy=200 + self.level * 20)
                    )

        # Check if invaders reached player
        for inv in alive:
            if inv.y + inv.h/2 >= self.player.y - 30:
                self.game_over = True
                break

    def _update_ufo(self, dt: float):
        self.ufo_timer += dt
        UFO_INTERVAL = 15.0

        if self.ufo is None or not self.ufo.alive:
            if self.ufo_timer >= UFO_INTERVAL:
                self.ufo_timer = 0.0
                direction = random.choice([-1, 1])
                self.ufo = UFO(
                    x=self.W + 30 if direction == -1 else -30,
                    y=30,
                    vx=direction * 120,
                )
        elif self.ufo and self.ufo.alive:
            self.ufo.x += self.ufo.vx * dt
            if self.ufo.x < -40 or self.ufo.x > self.W + 40:
                self.ufo = None

    def _update_particles(self, dt: float):
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.life -= dt
            if p.life <= 0:
                self.particles.pop(i)

    def _next_level(self):
        self.level += 1
        self.player_bullets = []
        self.enemy_bullets = []
        self.ufo = None
        self.ufo_timer = 0.0
        self._create_invaders()
        self.invader_speed = 30.0 + self.level * 10
        self.invader_shoot_chance = 0.002 + self.level * 0.001

    def _bullet_hits_shields(self, bullet: Bullet, idx: int, bullet_list: List[Bullet]) -> bool:
        for shield in self.shields:
            for j in range(len(shield) - 1, -1, -1):
                s = shield[j]
                if self._rect_collide(bullet.x, bullet.y, bullet.w, bullet.h, s.x, s.y, s.w, s.h):
                    s.hp -= 1
                    if s.hp <= 0:
                        shield.pop(j)
                    bullet_list.pop(idx)
                    return True
        return False

    def _spawn_particles(self, x: float, y: float, color: Tuple[int, int, int], count: int):
        for _ in range(count):
            self.particles.append(
                Particle(
                    x=x, y=y,
                    vx=(random.random() - 0.5) * 200,
                    vy=(random.random() - 0.5) * 200,
                    life=0.5 + random.random() * 0.5,
                    color=color,
                    size=2,
                )
            )

    @staticmethod
    def _rect_collide(ax: float, ay: float, aw: float, ah: float,
                      bx: float, by: float, bw: float, bh: float) -> bool:
        return ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by

    def _invader_color(self, inv: Invader) -> Tuple[int, int, int]:
        if inv.inv_type == 1:
            return (0, 255, 255)
        elif inv.inv_type == 2:
            return (255, 255, 0)
        else:
            return (255, 0, 255)

    def _get_obs(self) -> np.ndarray:
        return self._render_frame()

    def _render_frame(self) -> np.ndarray:
        # Create full-resolution frame
        frame = np.zeros((self.H, self.W, 3), dtype=np.uint8)

        # Draw ground
        frame[self.H - 20:self.H - 18, :, :] = [0, 255, 0]

        # Draw shields
        for shield in self.shields:
            for s in shield:
                x, y, w, h = int(s.x), int(s.y), int(s.w), int(s.h)
                alpha = s.hp / 3
                color = np.array([0, int(255 * alpha), 0], dtype=np.uint8)
                frame[y:y+h, x:x+w] = color

        # Draw invaders
        for inv in self.invaders:
            if inv.alive:
                color = self._invader_color(inv)
                x, y = int(inv.x - inv.w/2), int(inv.y - inv.h/2)
                w, h = int(inv.w), int(inv.h)
                frame[y:y+h, x:x+w] = color

        # Draw player
        if not self.game_over:
            px, py = int(self.player.x - self.player.w/2), int(self.player.y - self.player.h/2)
            pw, ph = int(self.player.w), int(self.player.h)
            frame[py:py+ph, px:px+pw] = [0, 255, 0]

        # Draw bullets
        for b in self.player_bullets:
            bx, by = int(b.x), int(b.y)
            bw, bh = int(b.w), int(b.h)
            frame[by:by+bh, bx:bx+bw] = [255, 255, 255]

        for b in self.enemy_bullets:
            bx, by = int(b.x), int(b.y)
            bw, bh = int(b.w), int(b.h)
            frame[by:by+bh, bx:bx+bw] = [255, 68, 68]

        # Draw UFO
        if self.ufo and self.ufo.alive:
            ux, uy = int(self.ufo.x - 25), int(self.ufo.y - 10)
            uw, uh = 50, 20
            frame[uy:uy+uh, ux:ux+uw] = [255, 0, 0]

        # Draw particles
        for p in self.particles:
            px, py = int(p.x), int(p.y)
            if 0 <= px < self.W and 0 <= py < self.H:
                frame[py:py+p.size, px:px+p.size] = p.color

        # Downscale to observation size
        from PIL import Image
        img = Image.fromarray(frame)
        img = img.resize((self.obs_width, self.obs_height), Image.Resampling.NEAREST)
        return np.array(img)

    def _render_human(self):
        import pygame

        if self._screen is None:
            pygame.init()
            pygame.display.init()
            self._screen = pygame.display.set_mode((self.W, self.H))
            pygame.display.set_caption("Space Invaders")
            self._clock = pygame.time.Clock()

        frame = self._render_frame()

        # Upscale to full resolution
        from PIL import Image
        img = Image.fromarray(frame)
        img = img.resize((self.W, self.H), Image.Resampling.NEAREST)
        frame_full = np.array(img)

        # Draw HUD
        surface = pygame.surfarray.make_surface(frame_full.transpose(1, 0, 2))
        self._screen.blit(surface, (0, 0))

        # Text overlay
        font = pygame.font.SysFont("Courier New", 18)
        score_text = font.render(f"SCORE: {self.score}", True, (255, 255, 255))
        level_text = font.render(f"LEVEL {self.level}", True, (255, 255, 255))
        lives_text = font.render(f"LIVES: {self.lives}", True, (255, 255, 255))

        self._screen.blit(score_text, (20, 5))
        self._screen.blit(level_text, (self.W // 2 - 40, 5))
        self._screen.blit(lives_text, (self.W - 150, 5))

        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])


# Convenience function for testing
def test_env():
    """Quick smoke test."""
    env = SpaceInvadersEnv(render_mode="rgb_array")
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Observation dtype: {obs.dtype}")
    print(f"Action space: {env.action_space}")

    total_reward = 0
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    print(f"Ran {step + 1} steps, total reward: {total_reward:.2f}")
    print(f"Final score: {info['score']}, lives: {info['lives']}")
    env.close()


if __name__ == "__main__":
    test_env()
