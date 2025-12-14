from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import pygame


def draw_rounded_rect(surface: pygame.Surface, rect: pygame.Rect, color, radius: int, width: int = 0) -> None:
    pygame.draw.rect(surface, color, rect, width=width, border_radius=radius)


def point_in_rounded_rect(pt: Tuple[float, float], rect: pygame.Rect, radius: int) -> bool:
    """True if point is inside a filled rounded rect."""
    x, y = pt

    if not rect.collidepoint(x, y):
        return False

    inner = rect.inflate(-2 * radius, -2 * radius)
    if inner.width < 0 or inner.height < 0:
        return True  # radius too large; treat as rect

    if inner.collidepoint(x, y):
        return True

    cx_left = rect.left + radius
    cx_right = rect.right - radius
    cy_top = rect.top + radius
    cy_bottom = rect.bottom - radius

    if x < inner.left and y < inner.top:
        cx, cy = cx_left, cy_top
    elif x > inner.right and y < inner.top:
        cx, cy = cx_right, cy_top
    elif x < inner.left and y > inner.bottom:
        cx, cy = cx_left, cy_bottom
    elif x > inner.right and y > inner.bottom:
        cx, cy = cx_right, cy_bottom
    else:
        return True

    dx = x - cx
    dy = y - cy
    return (dx * dx + dy * dy) <= radius * radius


@dataclass(frozen=True)
class RoundedRectTrack:
    outer_rect: pygame.Rect
    inner_rect: pygame.Rect
    corner_radius: int
    edge_width: int

    def on_track(self, pt: Tuple[float, float]) -> bool:
        inside_outer = point_in_rounded_rect(pt, self.outer_rect, self.corner_radius)
        inside_inner = point_in_rounded_rect(pt, self.inner_rect, self.corner_radius)
        return inside_outer and not inside_inner

    def draw(self, surface: pygame.Surface, *, bg_color, track_fill, track_edge) -> None:
        # Fill outer, carve inner hole with bg
        draw_rounded_rect(surface, self.outer_rect, track_fill, self.corner_radius, width=0)
        draw_rounded_rect(surface, self.inner_rect, bg_color, self.corner_radius, width=0)

        # Edges
        draw_rounded_rect(surface, self.outer_rect, track_edge, self.corner_radius, width=self.edge_width)
        draw_rounded_rect(surface, self.inner_rect, track_edge, self.corner_radius, width=self.edge_width)


def make_track_from_config(track_cfg) -> RoundedRectTrack:
    outer = pygame.Rect(*track_cfg.outer_rect)
    inner = pygame.Rect(*track_cfg.inner_rect)
    return RoundedRectTrack(
        outer_rect=outer,
        inner_rect=inner,
        corner_radius=track_cfg.corner_radius,
        edge_width=track_cfg.edge_width,
    )
