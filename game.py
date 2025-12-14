import math
import pygame

from config import GameConfig
from car import Car
from track import make_track_from_config


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

    def run(self):
        self.running = True
        while self.running:
            dt = self.clock.tick(self.cfg.screen.fps) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]:
                self.running = False
            if keys[pygame.K_r]:
                self.car.reset()
            self.car.step(dt, keys)

            # Off-track reset (center-point check for now)
            if not self.track.on_track(self.car.position()):
                self.car.reset()

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

            # HUD
            heading_deg = (math.degrees(self.car.state.heading) % 360.0)
            hud = f"speed={self.car.state.speed:7.1f}  heading={heading_deg:6.1f}°  (WASD drive, R reset, ESC quit)"
            self.screen.blit(self.font.render(hud, True, self.cfg.colors.hud), (20, 20))

            pygame.display.flip()

        pygame.quit()


def main():
    game = Game(GameConfig())
    game.run()

if __name__ == "__main__":
    main()
