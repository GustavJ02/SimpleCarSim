import math
import numpy as np


def raycast_endpoints(track, x, y, heading, *,
                     n_rays=9,
                     fov_deg=180,
                     max_dist=350,
                     step=6):
    """
    Returns: list of (start, end) tuples for each ray.
    """
    import math
    import numpy as np
    angles = np.linspace(-math.radians(fov_deg) / 2, math.radians(fov_deg) / 2, n_rays)
    endpoints = []
    for a in angles:
        ang = heading + a
        dx = math.cos(ang)
        dy = math.sin(ang)
        dist = 0.0
        while dist < max_dist:
            px = x + dx * dist
            py = y + dy * dist
            if not track.on_track((px, py)):
                break
            dist += step
        if dist >= max_dist:
            dist = max_dist
        endpoints.append(((x, y), (x + dx * dist, y + dy * dist)))
    return endpoints
