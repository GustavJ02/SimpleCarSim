from dataclasses import dataclass
from typing import Tuple

from math import pi

Color = Tuple[int, int, int]


@dataclass(frozen=True)
class ScreenConfig:
    width: int = 1100
    height: int = 700
    fps: int = 60
    title: str = "Manual Car (WASD) - Modular Starter"


@dataclass(frozen=True)
class ColorConfig:
    bg: Color = (15, 15, 20)
    track_fill: Color = (60, 60, 70)
    track_edge: Color = (220, 220, 230)
    car: Color = (0, 185, 231)
    hud: Color = (230, 230, 230)
    heading_line: Color = (255, 255, 255)


@dataclass(frozen=True)
class CarConfig:
    # Units: pixels, seconds, radians
    max_speed: float = 420.0
    accel: float = 600.0
    brake: float = 900.0
    friction: float = 420.0
    steer_rate: float = 2.6                 # rad/s at full steer
    steer_speed_factor: float = 0.004       # steering scales with |speed|
    reverse_speed_factor: float = 0.35      # reverse max speed = max_speed * factor

    width: float = 44.0
    height: float = 22.0
    nose_length: float = 28.0


@dataclass(frozen=True)
class TrackConfig:
    # Rounded-rect "oval-ish" track: outer boundary minus inner boundary
    outer_rect: Tuple[int, int, int, int] = (140, 80, 820, 540)   # x, y, w, h
    inner_rect: Tuple[int, int, int, int] = (260, 180, 580, 340)
    corner_radius: int = 170

    edge_width: int = 3


@dataclass(frozen=True)
class SpawnConfig:
    x: float = 200.0
    y: float = 350.0
    heading_rad: float = pi / 2.0  # facing down


@dataclass(frozen=True)
class GameConfig:
    screen: ScreenConfig = ScreenConfig()
    colors: ColorConfig = ColorConfig()
    car: CarConfig = CarConfig()
    track: TrackConfig = TrackConfig()
    spawn: SpawnConfig = SpawnConfig()
