from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import math
import pygame


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass
class CarState:
    x: float
    y: float
    heading: float  # radians
    speed: float    # pixels/sec


class Car:
    def __init__(self, car_cfg, spawn_cfg):
        self.cfg = car_cfg
        self.spawn = spawn_cfg
        self.state = CarState(spawn_cfg.x, spawn_cfg.y, spawn_cfg.heading_rad, 0.0)

    def reset(self) -> None:
        self.state.x = self.spawn.x
        self.state.y = self.spawn.y
        self.state.heading = self.spawn.heading_rad
        self.state.speed = 0.0

    def step(self, dt: float, keys=None, inputs : dict = None) -> None:
        # Inputs (WASD)
        if inputs is None:
            if keys is None:
                raise ValueError("keys must be provided when inputs is None")
            throttle = 1.0 if keys[pygame.K_w] else 0.0
            brake = 1.0 if keys[pygame.K_s] else 0.0
            steer_left = 1.0 if keys[pygame.K_a] else 0.0
            steer_right = 1.0 if keys[pygame.K_d] else 0.0
            steer = steer_right - steer_left  # -1..+1
        else:
            throttle = float(clamp(inputs.get("throttle", 0.0), 0.0, 1.0))
            brake = float(clamp(inputs.get("brake", 0.0), 0.0, 1.0))
            steer = float(clamp(inputs.get("steer", 0.0), -1.0, 1.0))

        # Longitudinal dynamics
        if throttle > 0:
            self.state.speed += self.cfg.accel * throttle * dt
        if brake > 0:
            self.state.speed -= self.cfg.brake * brake * dt

        # Friction towards 0
        if self.state.speed > 0:
            self.state.speed -= self.cfg.friction * dt
            if self.state.speed < 0:
                self.state.speed = 0
        elif self.state.speed < 0:
            self.state.speed += self.cfg.friction * dt
            if self.state.speed > 0:
                self.state.speed = 0

        # Clamp speed
        self.state.speed = clamp(
            self.state.speed,
            -self.cfg.max_speed * self.cfg.reverse_speed_factor,
            self.cfg.max_speed,
        )

        # Steering scales with speed magnitude (avoids turning in place)
        speed_mag = abs(self.state.speed)
        steer_scale = clamp(speed_mag * self.cfg.steer_speed_factor, 0.0, 1.0)

        # Reverse steering: yaw response flips when moving backwards
        direction = 1.0 if self.state.speed >= 0 else -1.0

        self.state.heading += (steer * direction) * self.cfg.steer_rate * steer_scale * dt


        # Integrate position
        vx = math.cos(self.state.heading) * self.state.speed
        vy = math.sin(self.state.heading) * self.state.speed
        self.state.x += vx * dt
        self.state.y += vy * dt

    def position(self) -> Tuple[float, float]:
        return (self.state.x, self.state.y)

    def draw(self, surface: pygame.Surface, *, car_color, heading_color) -> None:
        w = self.cfg.width
        h = self.cfg.height

        corners = [
            (-w / 2, -h / 2),
            ( w / 2, -h / 2),
            ( w / 2,  h / 2),
            (-w / 2,  h / 2),
        ]

        cos_h = math.cos(self.state.heading)
        sin_h = math.sin(self.state.heading)

        pts = []
        for lx, ly in corners:
            rx = lx * cos_h - ly * sin_h
            ry = lx * sin_h + ly * cos_h
            pts.append((self.state.x + rx, self.state.y + ry))

        pygame.draw.polygon(surface, car_color, pts)

        nose = (
            self.state.x + cos_h * self.cfg.nose_length,
            self.state.y + sin_h * self.cfg.nose_length,
        )
        pygame.draw.line(surface, heading_color, (self.state.x, self.state.y), nose, 2)
