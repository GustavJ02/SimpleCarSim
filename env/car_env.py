import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import GameConfig
from car import Car
from track import make_track_from_config
from sensors import raycast_distances


class CarEnv(gym.Env):
    metadata = {"render_modes": ["human", "none"], "render_fps": 60}

    def __init__(self, cfg: GameConfig | None = None, render_mode: str = "none"):
        super().__init__()
        self.cfg = cfg or GameConfig()
        self.render_mode = render_mode

        self.track = make_track_from_config(self.cfg.track)
        self.car = Car(self.cfg.car, self.cfg.spawn)

        # Fixed sim step for RL
        self.dt = 1.0 / 30.0
        self.max_steps = 2000
        self.step_count = 0

        # Sensors
        self.n_rays = 9
        self.ray_fov_deg = 180
        self.ray_max_dist = 350

        # Action: [steer, throttle, brake]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation: rays + speed + sin(heading) + cos(heading)
        # speed normalized to [-1, 1] by dividing by max_speed
        obs_dim = self.n_rays + 3
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # Rendering (optional)
        self._screen = None
        self._clock = None

    def _get_obs(self) -> np.ndarray:
        s = self.car.state
        rays = raycast_distances(
            self.track, s.x, s.y, s.heading,
            n_rays=self.n_rays,
            fov_deg=self.ray_fov_deg,
            max_dist=self.ray_max_dist,
            step=6,
        )
        speed_norm = float(s.speed / self.cfg.car.max_speed)
        # clamp to [-1, 1]
        speed_norm = max(-1.0, min(1.0, speed_norm))

        obs = np.concatenate([
            rays,                                  # [0,1]
            np.array([speed_norm,
                      math.sin(s.heading),
                      math.cos(s.heading)], dtype=np.float32)
        ]).astype(np.float32)

        # Convert rays from [0,1] to [-1,1] to match observation_space bounds (optional but convenient)
        obs[:self.n_rays] = obs[:self.n_rays] * 2.0 - 1.0
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.car.reset()
        self.step_count = 0
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        self.step_count += 1

        steer = float(action[0])
        throttle = float(action[1])
        brake = float(action[2])

        # Apply to your existing Car.step(inputs=...)
        # keys can be None; your Car.step ignores keys when inputs is provided
        self.car.step(self.dt, keys=None, inputs={
            "steer": steer,
            "throttle": throttle,
            "brake": brake
        })

        # Termination
        off_track = not self.track.on_track(self.car.position())
        terminated = bool(off_track)
        truncated = bool(self.step_count >= self.max_steps)

        # Reward baseline:
        # Encourage forward motion in the direction of heading, punish leaving track.
        # This is a temporary proxy until you add track progress/waypoints.
        forward_reward = max(0.0, self.car.state.speed) * self.dt * 0.01
        reward = forward_reward
        if off_track:
            reward -= 5.0

        obs = self._get_obs()
        info = {"off_track": off_track}

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        # Optional: you can reuse your existing rendering approach from game.py.
        # Keep it minimal: only needed for debugging, not for training speed.
        import pygame

        if self._screen is None:
            pygame.init()
            self._screen = pygame.display.set_mode((self.cfg.screen.width, self.cfg.screen.height))
            pygame.display.set_caption("CarEnv Render")
            self._clock = pygame.time.Clock()

        # Handle quit events so window stays responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pass

        self._screen.fill(self.cfg.colors.bg)
        self.track.draw(
            self._screen,
            bg_color=self.cfg.colors.bg,
            track_fill=self.cfg.colors.track_fill,
            track_edge=self.cfg.colors.track_edge,
        )
        self.car.draw(
            self._screen,
            car_color=self.cfg.colors.car,
            heading_color=self.cfg.colors.heading_line,
        )

        import pygame as _pg
        _pg.display.flip()
        self._clock.tick(self.cfg.screen.fps)

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.quit()
            self._screen = None
            self._clock = None
