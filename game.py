import math
import pygame
import time

from config import GameConfig
from car import Car
from track import make_track_from_config
from sensors import raycast_endpoints


class Game:
    def __init__(self, cfg: GameConfig):
        self.cfg = cfg

        pygame.init()
        self.screen = pygame.display.set_mode((cfg.screen.width, cfg.screen.height))
        pygame.display.set_caption(cfg.screen.title)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)

        self.track = make_track_from_config(cfg.track)
        self.car = Car(cfg.car, cfg.spawn)

        self.running = False
        self.crashed = False
        self.done = False
        self.travelled_distance = 0.0
        self.lap_start_time = time.time()
        self.last_position = (self.car.state.x, self.car.state.y)

    def reset(self):
        self.car.reset()
        self.crashed = False
        self.done = False
        self.travelled_distance = 0.0
        self.lap_start_time = time.time()
        self.last_position = (self.car.state.x, self.car.state.y)

    def get_observation(self):
        # Ray distances
        ray_endpoints = raycast_endpoints(
            self.track,
            self.car.state.x,
            self.car.state.y,
            self.car.state.heading,
            n_rays=self.cfg.rays.n_rays,
            fov_deg=self.cfg.rays.fov_deg,
            max_dist=self.cfg.rays.max_dist,
            step=self.cfg.rays.step,
        )
        ray_distances = [
            math.hypot(end[0] - start[0], end[1] - start[1])
            for start, end in ray_endpoints
        ]
        lap_time = time.time() - self.lap_start_time
        return {
            "speed": self.car.state.speed,
            "ray_distances": ray_distances,
            "position": (self.car.state.x, self.car.state.y),
            "heading": self.car.state.heading,
            "travelled_distance": self.travelled_distance,
            "lap_time": lap_time,
        }

    def step(self, input_fn=None):
        dt = self.clock.tick(self.cfg.screen.fps) / 1000.0
        events = {"quit": False, "reset": False, "crash": False}

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                events["quit"] = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            events["quit"] = True
        if keys[pygame.K_r]:
            events["reset"] = True

        if input_fn is not None:
            inputs = input_fn(self.get_observation())
            self.car.step(dt, inputs=inputs)
        else:
            self.car.step(dt, keys)

        # Crash detection (off-track)
        if not self.track.on_track(self.car.position()):
            events["crash"] = True
            self.crashed = True
            # Do not reset immediately

        # Update travelled distance
        current_position = self.car.position()
        dx = current_position[0] - self.last_position[0]
        dy = current_position[1] - self.last_position[1]
        self.travelled_distance += math.hypot(dx, dy)
        self.last_position = current_position

        # Render
        self.screen.fill(self.cfg.colors.bg)
        self.track.draw(
            self.screen,
            bg_color=self.cfg.colors.bg,
            track_fill=self.cfg.colors.track_fill,
            track_edge=self.cfg.colors.track_edge,
        )
        self.car.draw(
            self.screen,
            car_color=self.cfg.colors.car,
            heading_color=self.cfg.colors.heading_line,
        )

        # Draw rays
        ray_endpoints = raycast_endpoints(
            self.track,
            self.car.state.x,
            self.car.state.y,
            self.car.state.heading,
            n_rays=self.cfg.rays.n_rays,
            fov_deg=self.cfg.rays.fov_deg,
            max_dist=self.cfg.rays.max_dist,
            step=self.cfg.rays.step,
        )
        for start, end in ray_endpoints:
            pygame.draw.line(self.screen, (255, 255, 0), start, end, 2)

        # HUD
        heading_deg = (math.degrees(self.car.state.heading) % 360.0)
        hud = f"speed={self.car.state.speed:7.1f}  heading={heading_deg:6.1f}°  (WASD drive, R reset, ESC quit)"
        self.screen.blit(self.font.render(hud, True, self.cfg.colors.hud), (20, 20))

        pygame.display.flip()

        obs = self.get_observation()
        obs["crashed"] = self.crashed
        return obs, events

    def run(self, input_obj=None):
        self.reset()
        self.running = True
        while self.running:
            obs, events = self.step(input_obj.get_inputs)
            input_obj.feed_back(events, obs)
            if events["quit"]:
                self.running = False
            if events["reset"]:
                self.reset()
            # RL loop can check obs["crashed"] and decide when to reset

        pygame.quit()

