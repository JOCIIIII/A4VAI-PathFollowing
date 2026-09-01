############################################################
#
#   - Name : waypoint.py
#
#                                 - KAIST FDCL, 2026.03.11
#
############################################################

import math
import numpy as np


WAYPOINT_TYPE_NAME = {
    0: "external",
    1: "straight_line",
    2: "rectangle",
    3: "zigzag",
    4: "circle",
    5: "figure8",
    6: "altitude_change",
    7: "spiral",
    8: "lissajous",
    9: "test_course",
}


def build_waypoints(
    wp_type: int,
    waypoint_x=None,
    waypoint_y=None,
    waypoint_z=None,
) -> np.ndarray:
    """
    Return waypoint array with shape (N, 3), in NED frame.

    Parameters
    ----------
    wp_type : int
        0: external waypoint input
        1: straight line
        2: rectangle
        3: zigzag
        4: circle
        5: figure-8
        6: altitude change
        7: spiral
        8: lissajous
        9: test course
    waypoint_x, waypoint_y, waypoint_z :
        Used only when wp_type == 0
    """
    if wp_type == 0:
        return _build_external_waypoints(waypoint_x, waypoint_y, waypoint_z)

    if wp_type == 1:
        return _build_straight_line()

    if wp_type == 2:
        return _build_rectangle()

    if wp_type == 3:
        return _build_zigzag()

    if wp_type == 4:
        return _build_circle()

    if wp_type == 5:
        return _build_figure8()

    if wp_type == 6:
        return _build_altitude_change()

    if wp_type == 7:
        return _build_spiral()

    if wp_type == 8:
        return _build_lissajous()

    if wp_type == 9:
        return _build_test_course()

    raise RuntimeError(
        f"[FATAL] unsupported wp_type={int(wp_type)}. "
        f"Supported: {sorted(WAYPOINT_TYPE_NAME.keys())}"
    )


def insert_waypoint(
    waypoints: np.ndarray,
    waypoint_index: int,
    waypoint_position,
) -> np.ndarray:
    waypoint_position = np.asarray(waypoint_position, dtype=float).reshape(1, 3)
    return np.insert(waypoints, waypoint_index, waypoint_position, axis=0)


# =====================================================
# internal waypoint builders
# =====================================================
def _build_external_waypoints(waypoint_x, waypoint_y, waypoint_z) -> np.ndarray:
    if waypoint_x is None or waypoint_y is None or waypoint_z is None:
        raise RuntimeError(
            "[FATAL] external waypoint input is missing. "
            "waypoint_x/waypoint_y/waypoint_z are all required."
        )

    x = np.asarray(waypoint_x, dtype=float).reshape(-1)
    y = np.asarray(waypoint_y, dtype=float).reshape(-1)
    z = np.asarray(waypoint_z, dtype=float).reshape(-1)

    if not (len(x) == len(y) == len(z)):
        raise RuntimeError(
            "[FATAL] external waypoint lengths must match: "
            f"len(x)={len(x)}, len(y)={len(y)}, len(z)={len(z)}"
        )

    n = len(x)
    if n < 2:
        raise RuntimeError("[FATAL] external waypoint count must be >= 2.")

    wps = np.zeros((n, 3), dtype=float)
    wps[:, 0] = x[:n]
    wps[:, 1] = y[:n]
    wps[:, 2] = -z[:n]   # old code convention: input altitude -> NED z
    return wps


def _build_straight_line() -> np.ndarray:
    d = 100.0
    h1 = 10.0
    wp0 = 5.0

    return np.array(
        [
            [0.0, 0.0, -h1],
            [-wp0, 0.0, -h1],
            [-d + wp0, 0.0, -h1],
            [-d, 0.0, -h1],
        ],
        dtype=float,
    )


def _build_rectangle() -> np.ndarray:
    d = 40.0
    wp0 = 0.0
    h1 = 10.0
    h2 = 10.0

    return np.array(
        [
            [0.0, 0.0, -h1],
            [-wp0, wp0, -h1],
            [-wp0 - d, wp0, -h2],
            [-wp0 - d, wp0 + d, -h1],
            [-wp0, wp0 + d, -h2],
            [-wp0, wp0, -h1],
            [-0.5 * wp0, 0.5 * wp0, -h1],
            [0.0, 0.0, -h1],
        ],
        dtype=float,
    )


def _build_zigzag() -> np.ndarray:
    d1 = 60.0
    d2 = 20.0
    wp0 = 5.0
    h = 10.0

    return np.array(
        [
            [0.0, 0.0, -h],
            [-wp0, wp0, -h],
            [-wp0 - d2, wp0 + d1, -h],
            [-wp0 - 2.0 * d2, wp0, -h],
            [-wp0 - 3.0 * d2, wp0 + d1, -h],
            [-wp0 - 4.0 * d2, wp0, -h],
            [-wp0 - 4.0 * d2 - wp0, wp0, -h],
            [-wp0 - 4.0 * d2 - 2.0 * wp0, wp0, -h],
        ],
        dtype=float,
    )


def _build_circle() -> np.ndarray:
    n_cycle = 1
    radius = 20.0
    num_points = n_cycle * 20
    h = 10.0

    angles = n_cycle * 2.0 * math.pi * (np.arange(num_points) + 1.0) / num_points

    wps = -h * np.ones((num_points + 1, 3), dtype=float)
    wps[0, 0] = 0.0
    wps[0, 1] = 0.0
    wps[1:, 0] = radius * np.sin(angles)
    wps[1:, 1] = -radius * np.cos(angles) + radius
    return wps


def _build_figure8() -> np.ndarray:
    wp0 = 5.0
    radius = 16.0
    num_points = 40
    h = 10.0

    n = num_points // 4
    ang_half = math.pi * np.linspace(0.0, 1.0, n)
    ang_full = np.concatenate([ang_half, math.pi + ang_half])

    wps = -h * np.ones((num_points + 3, 3), dtype=float)

    wps[0, :] = [0.0, 0.0, -h]

    wps[1:n + 1, 0] = -radius + radius * np.cos(math.pi + ang_half[::-1]) - wp0
    wps[1:n + 1, 1] = radius * np.sin(math.pi + ang_half[::-1])

    wps[n + 1:3 * n + 1, 0] = -3.0 * radius + radius * np.cos(ang_full) - wp0
    wps[n + 1:3 * n + 1, 1] = radius * np.sin(ang_full)

    wps[3 * n + 1:4 * n + 1, 0] = -radius + radius * np.cos(ang_half[::-1]) - wp0
    wps[3 * n + 1:4 * n + 1, 1] = radius * np.sin(ang_half[::-1])

    wps[4 * n + 1, :] = [-wp0, 0.0, -h]
    wps[4 * n + 2, :] = [0.0, 0.0, -h]

    return wps


def _build_altitude_change() -> np.ndarray:
    d = 10.0
    h1 = 10.0
    h2 = 20.0
    wp0 = 5.0

    return np.array(
        [
            [0.0, 0.0, -h1],
            [-d, 0.0, -h1],
            [-2.0 * d, 0.0, -h2],
            [-3.0 * d, 0.0, -h2],
            [-4.0 * d, 0.0, -h1],
            [-5.0 * d + wp0, 0.0, -h1],
            [-5.0 * d, 0.0, -h1],
        ],
        dtype=float,
    )


def _build_spiral() -> np.ndarray:
    n_turns = 2
    points_per_turn = 20
    radius = 40.0
    h_start = 10.0
    h_step = 20.0

    num_points = n_turns * points_per_turn
    theta = np.linspace(0.0, 2.0 * math.pi * n_turns, num_points)

    wps = np.zeros((num_points, 3), dtype=float)
    wps[:, 0] = radius * np.cos(theta)
    wps[:, 1] = radius * np.sin(theta)
    wps[:, 2] = -(h_start + h_step * theta / (2.0 * math.pi))

    wps = np.vstack(([0.0, 0.0, -h_start], wps))
    wps = np.vstack((wps, [0.0, 0.0, -(h_start + h_step * n_turns)]))
    wps = np.vstack((wps, [0.0, 0.0, -h_start]))
    return wps


def _build_lissajous() -> np.ndarray:
    a = 30.0
    b = 20.0
    num_points = 40
    h = 10.0

    t = np.linspace(0.0, 2.0 * math.pi, num_points, endpoint=False)

    x = a * np.sin(t)
    y = b * np.sin(2.0 * t)
    z = -h * np.ones_like(t)

    wps = np.vstack([x, y, z]).T

    start_point = np.array([[0.0, 0.0, -h]], dtype=float)
    wps = np.concatenate([start_point, wps, start_point], axis=0)
    return wps


def _build_test_course() -> np.ndarray:
    d1 = 15.0
    d2 = 25.0
    d3 = 75.0
    h = 5.0

    return np.array(
        [
            [0.0, 0.0, -h],
            [0.0, d3, -h],
            [-d1, d3, -h],
            [-d1, -d2, -h],
            [-2.0 * d1, -d2, -h],
            [-2.0 * d1, d3, -h],
            [-3.0 * d1, d3, -h],
            [-3.0 * d1, -d2, -h],
            [-4.0 * d1, -d2, -h],
            [-4.0 * d1, d3, -h],
            [-5.0 * d1, d3, -h],
            [-5.0 * d1, -d2, -h],
            [-6.0 * d1, -d2, -h],
            [-6.0 * d1, d3, -h],
            [-7.0 * d1, d3, -h],
            [-7.0 * d1, -d2, -h],
            [-8.0 * d1, -d2, -h],
            [-8.0 * d1, d3, -h],
            [-9.0 * d1, d3, -h],
            [-9.0 * d1, -d2, -h],
        ],
        dtype=float,
    )
