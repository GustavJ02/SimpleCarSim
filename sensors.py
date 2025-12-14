import math
import numpy as np


def raycast_distances(track, x, y, heading, *,
                      n_rays=9,
                      fov_deg=180,
                      max_dist=350,
                      step=6):
    """
    Returns: np.array shape (n_rays,) with normalized distances in [0,1].
    """
    angles = np.linspace(-math.radians(fov_deg) / 2, math.radians(fov_deg) / 2, n_rays)
    out = np.zeros((n_rays,), dtype=np.float32)

    for i, a in enumerate(angles):
        ang = heading + a
        dx = math.cos(ang)
        dy = math.sin(ang)

        dist = 0.0
        hit = False
        while dist < max_dist:
            px = x + dx * dist
            py = y + dy * dist
            if not track.on_track((px, py)):
                hit = True
                break
            dist += step

        if not hit:
            dist = max_dist

        out[i] = dist / max_dist

    return out
