############################################################
#
#   - Name : gpr.py
#
#                                 - KAIST FDCL, 2026.03.11
#
############################################################

from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np


@dataclass
class GPRConfig:
    use_gpr_forecast: bool
    dt_gpr: float
    dt_gpr_opt_mul: int
    forecast_steps: int
    eps: float

    measurement_matrix: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0], dtype=np.float64)
    )
    meas_noise_var_x: float = 1.0e-6
    meas_noise_var_y: float = 1.0e-6

    hyp_l_init: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 1.0], dtype=np.float64)
    )
    hyp_q_init: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 1.0], dtype=np.float64)
    )
    hyp_n: int = 500

    hyp_l_min: float = 1.0e-2
    hyp_l_max: float = 0.5
    hyp_q_min: float = 1.0e-2
    hyp_q_max: float = 0.5

    lpf_cutoff_hz: float = 5.0

    @classmethod
    def from_dict(cls, cfg: dict) -> "GPRConfig":
        if "gpr" not in cfg:
            raise RuntimeError("[FATAL] missing required section: gpr")
        if "numerics" not in cfg or "eps" not in cfg["numerics"]:
            raise RuntimeError("[FATAL] missing required numerics.eps")
        gpr_cfg = cfg["gpr"]
        eps = float(cfg["numerics"]["eps"])
        if not np.isfinite(eps) or eps <= 0.0:
            raise RuntimeError("[FATAL] numerics.eps must be finite and > 0.")

        required_keys = (
            "use_gpr_forecast",
            "dt_gpr",
            "dt_gpr_opt_mul",
            "forecast_steps",
            "measurement_matrix",
            "meas_noise_var_x",
            "meas_noise_var_y",
            "hyp_l_init",
            "hyp_q_init",
            "hyp_n",
            "hyp_l_min",
            "hyp_l_max",
            "hyp_q_min",
            "hyp_q_max",
            "lpf_cutoff_hz",
        )
        missing = [k for k in required_keys if k not in gpr_cfg]
        if missing:
            missing_list = ", ".join(missing)
            raise RuntimeError(f"[FATAL] missing required gpr keys: {missing_list}")

        use_gpr_forecast = gpr_cfg["use_gpr_forecast"]
        if not isinstance(use_gpr_forecast, bool):
            raise RuntimeError("[FATAL] gpr.use_gpr_forecast must be boolean (true/false).")

        measurement_matrix = np.asarray(
            gpr_cfg["measurement_matrix"],
            dtype=np.float64,
        ).reshape(-1)
        if measurement_matrix.size != 2:
            raise RuntimeError("[FATAL] gpr.measurement_matrix must have 2 elements.")

        hyp_l_init = np.asarray(
            gpr_cfg["hyp_l_init"],
            dtype=np.float64,
        ).reshape(-1)
        hyp_q_init = np.asarray(
            gpr_cfg["hyp_q_init"],
            dtype=np.float64,
        ).reshape(-1)
        if hyp_l_init.size < 2 or hyp_q_init.size < 2:
            raise RuntimeError("[FATAL] gpr.hyp_l_init and gpr.hyp_q_init need at least 2 elements.")

        return cls(
            use_gpr_forecast=use_gpr_forecast,
            dt_gpr=float(gpr_cfg["dt_gpr"]),
            dt_gpr_opt_mul=int(gpr_cfg["dt_gpr_opt_mul"]),
            forecast_steps=int(gpr_cfg["forecast_steps"]),
            eps=eps,
            measurement_matrix=measurement_matrix[:2].copy(),
            meas_noise_var_x=float(gpr_cfg["meas_noise_var_x"]),
            meas_noise_var_y=float(gpr_cfg["meas_noise_var_y"]),
            hyp_l_init=hyp_l_init[:2].copy(),
            hyp_q_init=hyp_q_init[:2].copy(),
            hyp_n=int(gpr_cfg["hyp_n"]),
            hyp_l_min=float(gpr_cfg["hyp_l_min"]),
            hyp_l_max=float(gpr_cfg["hyp_l_max"]),
            hyp_q_min=float(gpr_cfg["hyp_q_min"]),
            hyp_q_max=float(gpr_cfg["hyp_q_max"]),
            lpf_cutoff_hz=float(gpr_cfg["lpf_cutoff_hz"]),
        )


class GPRCore:
    """
    Legacy-inspired GPR block used by MPPI.

    Model per axis:
      x = [disturbance, disturbance_rate]^T
      x(k+1) = A(l) x(k) + w(q)
      y(k)   = H x(k) + v
    """

    def __init__(self, cfg: GPRConfig):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.last_time = 0.0

        self.H = self.cfg.measurement_matrix.reshape(1, 2).astype(np.float64)
        self.R_x = float(max(self.cfg.meas_noise_var_x, self.cfg.eps))
        self.R_y = float(max(self.cfg.meas_noise_var_y, self.cfg.eps))

        self.hyp_l = np.array(
            [
                self._clamp_l(float(self.cfg.hyp_l_init[0])),
                self._clamp_l(float(self.cfg.hyp_l_init[1])),
            ],
            dtype=np.float64,
        )
        self.hyp_q = np.array(
            [
                self._clamp_q(float(self.cfg.hyp_q_init[0])),
                self._clamp_q(float(self.cfg.hyp_q_init[1])),
            ],
            dtype=np.float64,
        )

        self.A_x, self.Q_x = self._discrete_model(self.hyp_l[0], self.hyp_q[0])
        self.A_y, self.Q_y = self._discrete_model(self.hyp_l[1], self.hyp_q[1])

        self.m_x = np.zeros((2, 1), dtype=np.float64)
        self.P_x = np.zeros((2, 2), dtype=np.float64)
        self.m_y = np.zeros((2, 1), dtype=np.float64)
        self.P_y = np.zeros((2, 2), dtype=np.float64)

        self._init_lpf()

        self.training_data_x: list[np.ndarray] = []
        self.training_data_y: list[np.ndarray] = []
        self._opt_cycle = 0

    def _init_lpf(self) -> None:
        dt = max(float(self.cfg.dt_gpr), self.cfg.eps)
        wc = 2.0 * math.pi * max(float(self.cfg.lpf_cutoff_hz), self.cfg.eps)
        alpha = math.exp(-wc * dt)

        # y[k] = num*x[k-1] - den*y[k-1] with den < 0.
        self._num_lpf = 1.0 - alpha
        self._den_lpf = -alpha

        self._m_x_lpf = np.zeros((2, 1), dtype=np.float64)
        self._m_y_lpf = np.zeros((2, 1), dtype=np.float64)
        self._m_x_prev = np.zeros((2, 1), dtype=np.float64)
        self._m_y_prev = np.zeros((2, 1), dtype=np.float64)
        self._m_x_lpf_prev = np.zeros((2, 1), dtype=np.float64)
        self._m_y_lpf_prev = np.zeros((2, 1), dtype=np.float64)

    def _clamp_l(self, value: float) -> float:
        return float(np.clip(value, self.cfg.hyp_l_min, self.cfg.hyp_l_max))

    def _clamp_q(self, value: float) -> float:
        return float(np.clip(value, self.cfg.hyp_q_min, self.cfg.hyp_q_max))

    def _discrete_model(self, hyp_l: float, hyp_q: float) -> tuple[np.ndarray, np.ndarray]:
        dt = max(float(self.cfg.dt_gpr), self.cfg.eps)
        l = float(max(hyp_l, self.cfg.eps))
        q = float(max(hyp_q, self.cfg.eps))

        A = np.array(
            [
                [1.0, dt],
                [-(l * l) * dt, 1.0 - 2.0 * l * dt],
            ],
            dtype=np.float64,
        )

        q11 = (1.0 / 3.0) * (dt ** 3)
        q12 = 0.5 * (dt ** 2) - (2.0 * l / 3.0) * (dt ** 3)
        q22 = dt - 2.0 * l * (dt ** 2) + (4.0 / 3.0) * (l ** 2) * (dt ** 3)

        Q = q * np.array(
            [
                [q11, q12],
                [q12, q22],
            ],
            dtype=np.float64,
        )
        Q = 0.5 * (Q + Q.T)
        return A, Q

    def _kf_update(
        self,
        mp: np.ndarray,
        Pp: np.ndarray,
        measurement: float,
        R: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        v = float(measurement - (self.H @ mp)[0, 0])
        S = float((self.H @ Pp @ self.H.T)[0, 0] + R)
        if S < self.cfg.eps:
            S = self.cfg.eps
        K = (Pp @ self.H.T) / S

        m_next = mp + K * v
        P_next = Pp - (K @ K.T) * S
        P_next = 0.5 * (P_next + P_next.T)
        return m_next, P_next

    def _update_lpf_and_training(self, mp_x: np.ndarray, mp_y: np.ndarray) -> None:
        self._m_x_lpf = self._num_lpf * self._m_x_prev - self._den_lpf * self._m_x_lpf_prev
        self._m_x_prev = mp_x
        self._m_x_lpf_prev = self._m_x_lpf

        self._m_y_lpf = self._num_lpf * self._m_y_prev - self._den_lpf * self._m_y_lpf_prev
        self._m_y_prev = mp_y
        self._m_y_lpf_prev = self._m_y_lpf

        self.training_data_x.append(self._m_x_lpf.copy())
        self.training_data_y.append(self._m_y_lpf.copy())

        max_len = max(int(self.cfg.hyp_n), 2)
        if len(self.training_data_x) > max_len:
            self.training_data_x.pop(0)
            self.training_data_y.pop(0)

    def update(self, sim_time: float, measurement: np.ndarray) -> None:
        z = np.asarray(measurement, dtype=np.float64).reshape(-1)
        if z.size < 2:
            return

        self.last_time = float(sim_time)

        mp_x = self.A_x @ self.m_x
        Pp_x = self.A_x @ self.P_x @ self.A_x.T + self.Q_x
        self.m_x, self.P_x = self._kf_update(mp_x, Pp_x, float(z[0]), self.R_x)

        mp_y = self.A_y @ self.m_y
        Pp_y = self.A_y @ self.P_y @ self.A_y.T + self.Q_y
        self.m_y, self.P_y = self._kf_update(mp_y, Pp_y, float(z[1]), self.R_y)

        self._update_lpf_and_training(mp_x, mp_y)

    def _optimize_axis(self, axis: int) -> None:
        data = self.training_data_x if axis == 0 else self.training_data_y
        if len(data) < 2:
            return

        temp1 = np.zeros((2, 2), dtype=np.float64)
        temp2 = np.zeros((2, 2), dtype=np.float64)
        temp3 = np.zeros((2, 2), dtype=np.float64)

        for j in range(len(data) - 1):
            xk = np.asarray(data[j], dtype=np.float64).reshape(2, 1)
            xk1 = np.asarray(data[j + 1], dtype=np.float64).reshape(2, 1)
            temp1 += xk1 @ xk.T
            temp2 += xk @ xk.T

        if np.max(np.abs(temp1)) < 1.0 or np.max(np.abs(temp2)) < 1.0:
            return

        try:
            A_ml = temp1 @ np.linalg.inv(temp2 + np.eye(2, dtype=np.float64) * self.cfg.eps)
        except np.linalg.LinAlgError:
            return

        dt = max(float(self.cfg.dt_gpr), self.cfg.eps)
        l_opt = self._clamp_l((1.0 - float(A_ml[1, 1])) * 0.5 / dt)
        A_tmp, _ = self._discrete_model(l_opt, max(self.hyp_q[axis], self.cfg.eps))

        for j in range(len(data) - 1):
            xk = np.asarray(data[j], dtype=np.float64).reshape(2, 1)
            xk1 = np.asarray(data[j + 1], dtype=np.float64).reshape(2, 1)
            residual = xk1 - A_tmp @ xk
            temp3 += residual @ residual.T

        S_ml = temp3 / max(len(data) - 1, 1)
        denom = dt - 2.0 * l_opt * (dt ** 2) + (4.0 / 3.0) * (l_opt ** 2) * (dt ** 3)
        denom = max(denom, self.cfg.eps)
        q_opt = self._clamp_q(float(S_ml[1, 1]) / denom)

        A_new, Q_new = self._discrete_model(l_opt, q_opt)
        self.hyp_l[axis] = l_opt
        self.hyp_q[axis] = q_opt

        if axis == 0:
            self.A_x = A_new
            self.Q_x = Q_new
        else:
            self.A_y = A_new
            self.Q_y = Q_new

    def optimize_hyperparams(self) -> None:
        self._optimize_axis(self._opt_cycle)
        self._opt_cycle = (self._opt_cycle + 1) % 2

    def forecast(self, steps: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        if steps is None:
            n = max(int(self.cfg.forecast_steps), 1)
        else:
            n = max(int(steps), 1)

        mean = np.zeros((n, 3), dtype=np.float64)
        var = np.zeros((n, 1), dtype=np.float64)

        me_x = self.m_x.copy()
        Pe_x = self.P_x.copy()
        me_y = self.m_y.copy()
        Pe_y = self.P_y.copy()

        for i in range(n):
            me_x = self.A_x @ me_x
            Pe_x = self.A_x @ Pe_x @ self.A_x.T + self.Q_x
            me_y = self.A_y @ me_y
            Pe_y = self.A_y @ Pe_y @ self.A_y.T + self.Q_y

            mean[i, 0] = float(me_x[0, 0])
            mean[i, 1] = float(me_y[0, 0])
            mean[i, 2] = 0.0
            var[i, 0] = float(max(Pe_x[0, 0], 0.0))

        return mean, var
