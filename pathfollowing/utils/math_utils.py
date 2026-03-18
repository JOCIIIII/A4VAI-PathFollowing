############################################################
#
#   - Name : math_utils.py
#
#                                 - KAIST FDCL, 2026.03.11
#
############################################################

from __future__ import annotations

import math
import numpy as np


def wrap_pi(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def safe_norm(v: np.ndarray, eps: float = 1e-9) -> float:
    return max(float(np.linalg.norm(v)), eps)


def normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    return v / safe_norm(v, eps)


def quat_from_euler(roll: float, pitch: float, yaw: float) -> list[float]:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return [qw, qx, qy, qz]


def euler_from_quat(q0: float, q1: float, q2: float, q3: float) -> tuple[float, float, float]:
    # q = [w, x, y, z]
    sinr_cosp = 2.0 * (q0 * q1 + q2 * q3)
    cosr_cosp = 1.0 - 2.0 * (q1 * q1 + q2 * q2)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (q0 * q2 - q3 * q1)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (q0 * q3 + q1 * q2)
    cosy_cosp = 1.0 - 2.0 * (q2 * q2 + q3 * q3)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def azim_elev_from_vec3(vec: np.ndarray) -> tuple[float, float]:
    azim = math.atan2(float(vec[1]), float(vec[0]))
    elev = math.atan2(-float(vec[2]), math.sqrt(float(vec[0] * vec[0] + vec[1] * vec[1])))
    return azim, elev


def dcm_from_euler321(ang_euler321: np.ndarray) -> np.ndarray:
    phi, the, psi = float(ang_euler321[0]), float(ang_euler321[1]), float(ang_euler321[2])
    spsi = math.sin(psi)
    cpsi = math.cos(psi)
    sthe = math.sin(the)
    cthe = math.cos(the)
    sphi = math.sin(phi)
    cphi = math.cos(phi)

    dcm = np.zeros((3, 3), dtype=float)
    dcm[0, 0] = cpsi * cthe
    dcm[1, 0] = cpsi * sthe * sphi - spsi * cphi
    dcm[2, 0] = cpsi * sthe * cphi + spsi * sphi

    dcm[0, 1] = spsi * cthe
    dcm[1, 1] = spsi * sthe * sphi + cpsi * cphi
    dcm[2, 1] = spsi * sthe * cphi - cpsi * sphi

    dcm[0, 2] = -sthe
    dcm[1, 2] = cthe * sphi
    dcm[2, 2] = cthe * cphi
    return dcm
