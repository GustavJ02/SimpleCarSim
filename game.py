import math
import pygame

from config import GameConfig
from car import Car
from track import make_track_from_config


def main():
    cfg = GameConfig()

    pygame.init()
    screen = pygame.display.set_mode((cfg.screen.width, cfg.screen.height))
    pygame.display.set_caption(cfg.screen.title)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    track = make_track_from_config(cfg.track)
    car = Car(cfg.car, cfg.spawn)

    running = True
    while running:
        dt = clock.tick(cfg.screen.fps) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            running = False
        if keys[pygame.K_r]:
            car.reset()

        car.step(dt, keys)

        # Off-track reset (center-point check for now)
        if not track.on_track(car.position()):
            car.reset()

        # Render
        screen.fill(cfg.colors.bg)
        track.draw(
            screen,
            bg_color=cfg.colors.bg,
            track_fill=cfg.colors.track_fill,
            track_edge=cfg.colors.track_edge,
        )
        car.draw(
            screen,
            car_color=cfg.colors.car,
            heading_color=cfg.colors.heading_line,
        )

        # HUD
        heading_deg = (math.degrees(car.state.heading) % 360.0)
        hud = f"speed={car.state.speed:7.1f}  heading={heading_deg:6.1f}°  (WASD drive, R reset, ESC quit)"
        screen.blit(font.render(hud, True, cfg.colors.hud), (20, 20))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
