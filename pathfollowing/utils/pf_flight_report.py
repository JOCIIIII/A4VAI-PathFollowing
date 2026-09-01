############################################################
#
#   - Name : pf_flight_report.py
#
#                                 - KAIST FDCL, 2026.03.11
#
############################################################

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "pf_outputs"
DEFAULT_HOST_LOG_DIR = Path.home() / "Documents" / "A4VAI-SITL" / "ROS2" / "logs"
DEFAULT_CONTAINER_LOG_DIR = Path("/home/user/workspace/ros2/logs")
LEGACY_HIDDEN_DATA_DIR = Path(__file__).resolve().parent / ".pf_data"


def _to_float(value: Any) -> float:
    if value is None:
        return float("nan")
    text = str(value).strip()
    if text == "":
        return float("nan")
    try:
        return float(text)
    except (TypeError, ValueError):
        return float("nan")


def _load_csv_numeric(csv_path: Path) -> dict[str, np.ndarray]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"[FATAL] CSV has no header: {csv_path}")

        buffer: dict[str, list[float]] = {name: [] for name in reader.fieldnames}
        row_count = 0
        for row in reader:
            row_count += 1
            for name in reader.fieldnames:
                buffer[name].append(_to_float(row.get(name)))

    if row_count == 0:
        raise RuntimeError(f"[FATAL] CSV has no data rows: {csv_path}")

    return {k: np.asarray(v, dtype=float) for k, v in buffer.items()}


def _get_col(data: dict[str, np.ndarray], name: str, default: float = float("nan")) -> np.ndarray:
    if name in data:
        return data[name]
    length = 0
    if data:
        length = len(next(iter(data.values())))
    return np.full(length, default, dtype=float)


def _latest_csv(log_dir: Path) -> Path:
    csv_files = list(log_dir.glob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"[FATAL] no CSV files found in: {log_dir}")
    return max(csv_files, key=lambda p: p.stat().st_mtime)


def _candidate_log_dirs(primary_log_dir: Path) -> list[Path]:
    candidates: list[Path] = []

    def _append_unique(path: Path) -> None:
        for existing in candidates:
            try:
                if existing.resolve() == path.resolve():
                    return
            except Exception:
                if str(existing) == str(path):
                    return
        candidates.append(path)

    _append_unique(primary_log_dir)
    _append_unique(DEFAULT_HOST_LOG_DIR)
    _append_unique(DEFAULT_CONTAINER_LOG_DIR)
    _append_unique(LEGACY_HIDDEN_DATA_DIR)
    return candidates


def _default_log_dir() -> Path:
    env_path = os.environ.get("PF_LOG_DIR", "").strip()
    if env_path:
        return Path(env_path).expanduser()
    return DEFAULT_HOST_LOG_DIR


def _default_ingest_dir() -> Path:
    env_path = os.environ.get("PF_INGEST_DIR", "").strip()
    if env_path:
        return Path(env_path).expanduser()
    env_data_path = os.environ.get("PF_DATA_DIR", "").strip()
    if env_data_path:
        return Path(env_data_path).expanduser()
    return DEFAULT_DATA_DIR


def _resolve_csv_arg(csv_arg: str, ingest_dir: Path) -> Path:
    direct_path = Path(csv_arg).expanduser()
    if direct_path.exists():
        return direct_path

    # Convenience: `--csv 2026-...csv` resolves inside pf_outputs by default.
    if direct_path.parent == Path("."):
        in_ingest = ingest_dir / direct_path.name
        if in_ingest.exists():
            return in_ingest

    raise RuntimeError(f"[FATAL] CSV not found: {direct_path}")


def _select_csv_path(csv_arg: str | None, log_dir: Path, ingest_dir: Path, copy_latest: bool) -> Path:
    if csv_arg is not None:
        return _resolve_csv_arg(csv_arg, ingest_dir)

    src_csv: Path | None = None
    for candidate_dir in _candidate_log_dirs(log_dir):
        try:
            src_csv = _latest_csv(candidate_dir)
            if candidate_dir != log_dir:
                print(f"[INFO] No CSV in {log_dir}; using fallback dir: {candidate_dir}")
            break
        except RuntimeError:
            continue

    if src_csv is None:
        candidate_text = ", ".join(str(p) for p in _candidate_log_dirs(log_dir))
        raise RuntimeError(f"[FATAL] no CSV files found in candidate dirs: {candidate_text}")

    if not copy_latest:
        return src_csv

    ingest_dir.mkdir(parents=True, exist_ok=True)
    dst_csv = ingest_dir / src_csv.name
    try:
        if src_csv.resolve() == dst_csv.resolve():
            return src_csv
    except Exception:
        pass
    shutil.copy2(src_csv, dst_csv)
    print(f"[INFO] Copied latest CSV: {src_csv} -> {dst_csv}")
    return dst_csv


def _load_sim_params(sim_yaml_path: Path) -> dict[str, Any]:
    if not sim_yaml_path.exists():
        return {}

    with sim_yaml_path.open("r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    if not isinstance(doc, dict):
        return {}

    node_cfg = doc.get("node_pathfollowing")
    if not isinstance(node_cfg, dict):
        return {}
    ros_params = node_cfg.get("ros__parameters", {})
    if not isinstance(ros_params, dict):
        return {}
    return ros_params


def _format_unique_ints(values: np.ndarray) -> str:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return "N/A"
    unique_ints = np.unique(finite_values.astype(int))
    return ",".join(str(v) for v in unique_ints.tolist())


def _median_finite(values: np.ndarray) -> float | None:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return None
    return float(np.nanmedian(finite_values))


def _fmt_scalar(value: Any, nd: int = 3) -> str:
    try:
        number = float(value)
        if math.isfinite(number):
            return f"{number:.{nd}f}"
    except (TypeError, ValueError):
        pass
    return str(value)


def _path_following_cfg(sim_params: dict[str, Any]) -> dict[str, Any]:
    pf_cfg = sim_params.get("path_following", {})
    if isinstance(pf_cfg, dict):
        return pf_cfg
    return {}


def _meta_lines(
    csv_path: Path,
    data: dict[str, np.ndarray],
    sim_params: dict[str, Any],
) -> list[str]:
    guid_cfg_log = _format_unique_ints(_get_col(data, "guid_type_cfg"))
    guid_used_log = _format_unique_ints(_get_col(data, "guid_type_used"))
    wp_heading_last = _get_col(data, "wp_idx_heading")
    wp_passed_last = _get_col(data, "wp_idx_passed")

    pf_cfg = _path_following_cfg(sim_params)

    desired_speed_log = _get_col(data, "desired_speed")
    desired_speed_value = float(np.nanmedian(desired_speed_log)) if np.isfinite(desired_speed_log).any() else float("nan")

    vehicle_type = sim_params.get("vehicle_type", "N/A")
    wp_type = sim_params.get("wp_type", "N/A")
    guid_type_init = sim_params.get("guid_type", "N/A")
    desired_speed_cfg: float | Any = pf_cfg.get("desired_speed", "N/A")
    lookahead: float | Any = pf_cfg.get("lookahead_distance", "N/A")
    wp_switch: float | Any = pf_cfg.get("waypoint_switch_distance", "N/A")

    vehicle_type_log = _median_finite(_get_col(data, "vehicle_type_cfg"))
    if vehicle_type_log is not None:
        vehicle_type = int(round(vehicle_type_log))

    wp_type_log = _median_finite(_get_col(data, "wp_type_cfg"))
    if wp_type_log is not None:
        wp_type = int(round(wp_type_log))

    guid_type_init_log = _median_finite(_get_col(data, "guid_type_init"))
    if guid_type_init_log is not None:
        guid_type_init = int(round(guid_type_init_log))

    desired_speed_cfg_log = _median_finite(_get_col(data, "desired_speed_cfg"))
    if desired_speed_cfg_log is not None:
        desired_speed_cfg = desired_speed_cfg_log

    lookahead_log = _median_finite(_get_col(data, "lookahead_distance_cfg"))
    if lookahead_log is not None:
        lookahead = lookahead_log

    wp_switch_log = _median_finite(_get_col(data, "waypoint_switch_distance_cfg"))
    if wp_switch_log is not None:
        wp_switch = wp_switch_log

    heading_last = int(wp_heading_last[-1]) if wp_heading_last.size > 0 and math.isfinite(wp_heading_last[-1]) else -1
    passed_last = int(wp_passed_last[-1]) if wp_passed_last.size > 0 and math.isfinite(wp_passed_last[-1]) else -1

    return [
        f"log: {csv_path.name}",
        (
            f"cfg: vehicle={vehicle_type}, guid_init={guid_type_init}, wp={wp_type} | "
            f"log: guid_cfg={guid_cfg_log}, guid_used={guid_used_log}"
        ),
        (
            f"speed[cfg/log]={_fmt_scalar(desired_speed_cfg)}/{_fmt_scalar(desired_speed_value)} | "
            f"lookahead={_fmt_scalar(lookahead)} | wp_switch={_fmt_scalar(wp_switch)} | "
            f"wp_idx[h/p]={heading_last}/{passed_last}"
        ),
    ]


def _apply_figure_header(fig: plt.Figure, title: str, meta: list[str]) -> None:
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.992)
    fig.text(
        0.012,
        0.957,
        "\n".join(meta),
        ha="left",
        va="top",
        fontsize=8.2,
        family="monospace",
        linespacing=1.2,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.80", alpha=0.95),
    )


def _save_figure(fig: plt.Figure, save_dir: Path, stem: str, suffix: str, dpi: int) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{stem}_{suffix}.png"
    fig.savefig(out_path, dpi=dpi)
    return out_path


def _copy_csv_to_dir(csv_path: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_csv = dst_dir / csv_path.name
    try:
        if csv_path.resolve() == dst_csv.resolve():
            return dst_csv
    except Exception:
        pass
    shutil.copy2(csv_path, dst_csv)
    print(f"[INFO] Copied CSV to plot dir: {csv_path} -> {dst_csv}")
    return dst_csv


def _slice_data(data: dict[str, np.ndarray], start_idx: int, end_idx: int) -> dict[str, np.ndarray]:
    return {name: values[start_idx:end_idx] for name, values in data.items()}


def _time_for_index(t: np.ndarray, index: int) -> float | None:
    if t.size <= index:
        return None
    value = float(t[index])
    if math.isfinite(value):
        return value
    return None


def _index_at_or_after_time(t: np.ndarray, target_time: float, fallback_idx: int) -> int:
    valid = np.flatnonzero(np.isfinite(t) & (t >= target_time))
    if valid.size == 0:
        return fallback_idx
    return int(valid[0])


def _index_at_or_before_time(t: np.ndarray, target_time: float, fallback_idx: int) -> int:
    valid = np.flatnonzero(np.isfinite(t) & (t <= target_time))
    if valid.size == 0:
        return fallback_idx
    return int(valid[-1])


def _find_arm_disarm_window(
    data: dict[str, np.ndarray],
    pre_takeoff_s: float,
    post_landing_s: float,
) -> tuple[int, int, str] | None:
    source = ""
    is_armed = _get_col(data, "vehicle_is_armed")
    if is_armed.size > 0 and np.isfinite(is_armed).any():
        source = "vehicle_is_armed"
        valid = np.isfinite(is_armed)
        armed_mask = valid & (is_armed >= 0.5)
    else:
        arming_state = _get_col(data, "vehicle_arming_state")
        if arming_state.size == 0 or not np.isfinite(arming_state).any():
            return None
        source = "vehicle_arming_state"
        valid = np.isfinite(arming_state)
        armed_mask = np.zeros_like(arming_state, dtype=bool)
        armed_mask[valid] = np.rint(arming_state[valid]).astype(int) == 2

    armed_indices = np.flatnonzero(armed_mask)
    if armed_indices.size == 0:
        return None

    raw_start_idx = int(armed_indices[0])
    disarmed_after_start = np.flatnonzero(
        valid
        & (~armed_mask)
        & (np.arange(armed_mask.size) > raw_start_idx)
    )
    if disarmed_after_start.size > 0:
        raw_end_idx = int(disarmed_after_start[0])
        end_label = "disarm"
    else:
        raw_end_idx = int(armed_indices[-1])
        end_label = "last_armed"

    t = _get_col(data, "time")
    start_idx = raw_start_idx
    end_idx = raw_end_idx

    start_time = _time_for_index(t, raw_start_idx)
    if start_time is not None and pre_takeoff_s > 0.0:
        start_idx = _index_at_or_after_time(t, start_time - pre_takeoff_s, raw_start_idx)

    end_time = _time_for_index(t, raw_end_idx)
    if end_time is not None and post_landing_s > 0.0:
        end_idx = _index_at_or_before_time(t, end_time + post_landing_s, raw_end_idx)

    end_idx = min(end_idx + 1, armed_mask.size)
    if start_idx >= end_idx:
        return None

    window_start_time = _time_for_index(t, start_idx)
    window_end_time = _time_for_index(t, end_idx - 1)
    raw_start_time = _time_for_index(t, raw_start_idx)
    raw_end_time = _time_for_index(t, raw_end_idx)
    detail = (
        "window: arm_disarm "
        f"rows={start_idx}:{end_idx} "
        f"t={_fmt_scalar(window_start_time)}..{_fmt_scalar(window_end_time)}s "
        f"arm_t={_fmt_scalar(raw_start_time)}s "
        f"{end_label}_t={_fmt_scalar(raw_end_time)}s "
        f"source={source}"
    )
    return start_idx, end_idx, detail


def _find_altitude_takeoff_landing_window(
    data: dict[str, np.ndarray],
    takeoff_altitude_m: float,
    landing_altitude_m: float,
    pre_takeoff_s: float,
    post_landing_s: float,
) -> tuple[int, int, str] | None:
    pos_z = _get_col(data, "pos_z")
    if pos_z.size == 0 or not np.isfinite(pos_z).any():
        return None

    altitude = -pos_z
    finite_altitude = np.isfinite(altitude)
    takeoff_candidates = np.flatnonzero(finite_altitude & (altitude >= takeoff_altitude_m))
    if takeoff_candidates.size == 0:
        return None

    raw_start_idx = int(takeoff_candidates[0])
    landing_candidates = np.flatnonzero(
        finite_altitude
        & (altitude >= landing_altitude_m)
        & (np.arange(altitude.size) >= raw_start_idx)
    )
    if landing_candidates.size == 0:
        return None
    raw_end_idx = int(landing_candidates[-1])

    t = _get_col(data, "time")
    start_idx = raw_start_idx
    end_idx = raw_end_idx

    start_time = _time_for_index(t, raw_start_idx)
    if start_time is not None and pre_takeoff_s > 0.0:
        start_idx = _index_at_or_after_time(t, start_time - pre_takeoff_s, raw_start_idx)

    end_time = _time_for_index(t, raw_end_idx)
    if end_time is not None and post_landing_s > 0.0:
        end_idx = _index_at_or_before_time(t, end_time + post_landing_s, raw_end_idx)

    end_idx = min(end_idx + 1, altitude.size)
    if start_idx >= end_idx:
        return None

    window_start_time = _time_for_index(t, start_idx)
    window_end_time = _time_for_index(t, end_idx - 1)
    raw_start_time = _time_for_index(t, raw_start_idx)
    raw_end_time = _time_for_index(t, raw_end_idx)
    detail = (
        "window: takeoff_landing_altitude "
        f"rows={start_idx}:{end_idx} "
        f"t={_fmt_scalar(window_start_time)}..{_fmt_scalar(window_end_time)}s "
        f"threshold_t={_fmt_scalar(raw_start_time)}..{_fmt_scalar(raw_end_time)}s "
        f"alt={_fmt_scalar(takeoff_altitude_m)}/{_fmt_scalar(landing_altitude_m)}m"
    )
    return start_idx, end_idx, detail


def _find_takeoff_landing_window(
    data: dict[str, np.ndarray],
    takeoff_altitude_m: float,
    landing_altitude_m: float,
    pre_takeoff_s: float,
    post_landing_s: float,
) -> tuple[int, int, str] | None:
    window = _find_arm_disarm_window(data, pre_takeoff_s, post_landing_s)
    if window is not None:
        return window

    return _find_altitude_takeoff_landing_window(
        data=data,
        takeoff_altitude_m=takeoff_altitude_m,
        landing_altitude_m=landing_altitude_m,
        pre_takeoff_s=pre_takeoff_s,
        post_landing_s=post_landing_s,
    )


def _first_true_interval(mask: np.ndarray) -> tuple[int, int] | None:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return None

    start_idx = int(indices[0])
    end_idx = start_idx
    for index in indices[1:]:
        index = int(index)
        if index != end_idx + 1:
            break
        end_idx = index
    return start_idx, end_idx


def _finite_rounded_isin(values: np.ndarray, targets: tuple[int, ...]) -> np.ndarray:
    finite = np.isfinite(values)
    rounded = np.zeros(values.shape, dtype=int)
    rounded[finite] = np.rint(values[finite]).astype(int)
    return finite & np.isin(rounded, targets)


def _find_mppi_interval(data: dict[str, np.ndarray]) -> tuple[int, int, float, float, str] | None:
    t = _get_col(data, "time")
    if t.size == 0 or not np.isfinite(t).any():
        return None

    guid_type_used = _get_col(data, "guid_type_used")
    mppi_solve_time = _get_col(data, "mppi_solve_time")
    vehicle_nav_state = _get_col(data, "vehicle_nav_state")
    pf_done = _get_col(data, "pf_done")

    valid_time = np.isfinite(t)
    mppi_guidance = _finite_rounded_isin(guid_type_used, (1, 2))
    mppi_output = np.isfinite(mppi_solve_time) & (mppi_solve_time > 0.0)

    if np.isfinite(vehicle_nav_state).any():
        offboard = _finite_rounded_isin(vehicle_nav_state, (14,))
    else:
        offboard = np.ones(t.size, dtype=bool)

    if np.isfinite(pf_done).any():
        not_done = np.isfinite(pf_done) & (pf_done < 0.5)
    else:
        not_done = np.ones(t.size, dtype=bool)

    masks = [
        ("offboard_mppi", valid_time & mppi_guidance & mppi_output & offboard & not_done),
        ("mppi_guidance", valid_time & mppi_guidance & mppi_output & not_done),
        ("mppi_output", valid_time & mppi_output),
    ]
    for source, mask in masks:
        interval = _first_true_interval(mask)
        if interval is None:
            continue
        start_idx, end_idx = interval
        start_time = _time_for_index(t, start_idx)
        end_time = _time_for_index(t, end_idx)
        if start_time is None or end_time is None:
            continue
        return start_idx, end_idx, start_time, end_time, source

    return None


def _find_mppi_window(data: dict[str, np.ndarray]) -> tuple[int, int, str] | None:
    interval = _find_mppi_interval(data)
    if interval is None:
        return None

    start_idx, end_idx, start_time, end_time, source = interval
    end_idx = min(end_idx + 1, len(_get_col(data, "time")))
    if start_idx >= end_idx:
        return None

    detail = (
        "window: mppi "
        f"rows={start_idx}:{end_idx} "
        f"t={_fmt_scalar(start_time)}..{_fmt_scalar(end_time)}s "
        f"source={source}"
    )
    return start_idx, end_idx, detail


def _mask_outside_time_window(
    values: np.ndarray,
    t: np.ndarray,
    interval: tuple[int, int, float, float, str] | None,
) -> np.ndarray:
    masked = values.astype(float).copy()
    if interval is None:
        return masked

    _, _, start_time, end_time, _ = interval
    inside = np.isfinite(t) & (t >= start_time) & (t <= end_time)
    masked[~inside] = np.nan
    return masked


def _speed_reference_for_plot(
    data: dict[str, np.ndarray],
    t: np.ndarray,
    mppi_interval: tuple[int, int, float, float, str] | None,
) -> tuple[np.ndarray, str]:
    desired_speed = _get_col(data, "desired_speed")
    desired_speed_target = _get_col(data, "desired_speed_target")
    mppi_ref = desired_speed if np.isfinite(desired_speed).any() else desired_speed_target

    guid_type_used = _get_col(data, "guid_type_used")
    if not np.isfinite(guid_type_used).any():
        return _mask_outside_time_window(mppi_ref, t, mppi_interval), "desired_speed"

    ref = np.full(t.shape, np.nan, dtype=float)

    mppi_mask = _finite_rounded_isin(guid_type_used, (1, 2))
    if np.isfinite(mppi_ref).any():
        ref[mppi_mask] = mppi_ref[mppi_mask]

    if np.isfinite(ref).any():
        if mppi_interval is not None:
            ref = _mask_outside_time_window(ref, t, mppi_interval)
        return ref, "desired_speed"

    return ref, "desired_speed"


MPPI_START_COLOR = "#00e676"
MPPI_END_COLOR = "#ff1744"
MPPI_MARKER_LINEWIDTH = 2.0
MPPI_MARKER_ALPHA = 0.95


def _add_mppi_time_markers(
    ax,
    interval: tuple[int, int, float, float, str] | None,
    *,
    label: bool = False,
) -> None:
    if interval is None:
        return

    _, _, start_time, end_time, _ = interval
    start_label = f"MPPI start {start_time:.1f}s" if label else "_nolegend_"
    end_label = f"MPPI end {end_time:.1f}s" if label else "_nolegend_"
    ax.axvline(
        start_time,
        color=MPPI_START_COLOR,
        linestyle=":",
        linewidth=MPPI_MARKER_LINEWIDTH,
        alpha=MPPI_MARKER_ALPHA,
        label=start_label,
    )
    ax.axvline(
        end_time,
        color=MPPI_END_COLOR,
        linestyle=":",
        linewidth=MPPI_MARKER_LINEWIDTH,
        alpha=MPPI_MARKER_ALPHA,
        label=end_label,
    )


def _plot_trajectory_error(
    t: np.ndarray,
    data: dict[str, np.ndarray],
    meta: list[str],
    mppi_interval: tuple[int, int, float, float, str] | None,
) -> plt.Figure:
    fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=False)
    _apply_figure_header(fig, "PathFollowing: Trajectory & Error", meta)

    pos_x = _get_col(data, "pos_x")
    pos_z = _get_col(data, "pos_z")
    vt_x = _get_col(data, "vt_x")
    vt_z = _get_col(data, "vt_z")
    cp_x = _get_col(data, "closest_x")
    cp_z = _get_col(data, "closest_z")

    speed_error = _get_col(data, "speed_error")
    cte = _get_col(data, "cross_track_error")
    dist_heading = _get_col(data, "dist_to_heading_wp")
    speed_ref, speed_ref_label = _speed_reference_for_plot(
        data=data,
        t=t,
        mppi_interval=mppi_interval,
    )

    vel_x = _get_col(data, "vel_x")
    vel_y = _get_col(data, "vel_y")
    vel_z = _get_col(data, "vel_z")
    speed_mag = np.sqrt(vel_x * vel_x + vel_y * vel_y + vel_z * vel_z)
    if np.isfinite(speed_ref).any():
        speed_error = np.abs(speed_ref - speed_mag)
    else:
        speed_error = np.full(t.shape, np.nan, dtype=float)

    ax = axs[0, 0]
    ax.plot(pos_x, -pos_z, label="vehicle", linewidth=1.6)
    if np.isfinite(vt_x).any() and np.isfinite(vt_z).any():
        ax.plot(vt_x, -vt_z, label="virtual_target", linewidth=1.0, alpha=0.8)
    if np.isfinite(cp_x).any() and np.isfinite(cp_z).any():
        ax.plot(cp_x, -cp_z, label="closest_path_point", linewidth=1.0, alpha=0.8)
    if mppi_interval is not None:
        start_idx, end_idx, start_time, end_time, _ = mppi_interval
        if start_idx < pos_x.size and end_idx < pos_x.size:
            ax.scatter(
                pos_x[start_idx],
                -pos_z[start_idx],
                s=52,
                marker="o",
                color=MPPI_START_COLOR,
                label="_nolegend_",
            )
            ax.scatter(
                pos_x[end_idx],
                -pos_z[end_idx],
                s=64,
                marker="x",
                color=MPPI_END_COLOR,
                linewidths=MPPI_MARKER_LINEWIDTH,
                label="_nolegend_",
            )
    ax.set_title("XZ Trajectory")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("up [m]")
    ax.axis("equal")
    ax.grid(True)
    ax.legend(loc="best")

    ax = axs[0, 1]
    cte_line, = ax.plot(t, cte, label="cross_track_error", color="tab:blue")
    lines = [cte_line]
    if np.isfinite(dist_heading).any():
        ax_dist = ax.twinx()
        dist_heading_line, = ax_dist.plot(
            t,
            dist_heading,
            label="dist_to_heading_wp",
            color="tab:green",
            alpha=0.75,
        )
        ax_dist.set_ylabel("distance [m]")
        ax_dist.tick_params(axis="y", labelcolor="tab:green")
        lines.append(dist_heading_line)
    _add_mppi_time_markers(ax, mppi_interval)
    ax.set_title("Path Error")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("cross track error [m]")
    ax.grid(True)
    ax.legend(lines, [line.get_label() for line in lines], loc="best")

    ax = axs[1, 0]
    ax.plot(t, speed_mag, label="|vel|")
    if np.isfinite(speed_ref).any():
        ax.plot(t, speed_ref, label=speed_ref_label, linewidth=1.5)
    _add_mppi_time_markers(ax, mppi_interval)
    ax.set_title("Speed Tracking")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("m/s")
    ax.grid(True)
    ax.legend(loc="best")

    ax = axs[1, 1]
    if np.isfinite(speed_error).any():
        ax.plot(t, speed_error, label="speed_error", color="tab:orange")
    _add_mppi_time_markers(ax, mppi_interval)
    ax.set_title("Speed Error")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("m/s")
    ax.set_ylim(0.0, 3.0)
    ax.grid(True)
    ax.legend(loc="best")

    fig.tight_layout(rect=(0.02, 0.03, 0.99, 0.88))
    return fig


def _plot_attitude_thrust(
    t: np.ndarray,
    data: dict[str, np.ndarray],
    meta: list[str],
    mppi_interval: tuple[int, int, float, float, str] | None,
) -> plt.Figure:
    fig, axs = plt.subplots(4, 1, figsize=(14, 9), sharex=True)
    _apply_figure_header(fig, "PathFollowing: Attitude & Thrust", meta)

    eul_roll = _get_col(data, "eul_roll")
    eul_pitch = _get_col(data, "eul_pitch")
    eul_yaw = _get_col(data, "eul_yaw")
    cmd_roll = _get_col(data, "cmd_roll")
    cmd_pitch = _get_col(data, "cmd_pitch")
    cmd_yaw = _get_col(data, "cmd_yaw")
    thrust_norm = _get_col(data, "thrust_norm_cmd")
    thrust_total = _get_col(data, "thrust_total_cmd")

    axs[0].plot(t, np.degrees(np.unwrap(eul_roll)), label="roll")
    axs[0].plot(t, np.degrees(np.unwrap(cmd_roll)), label="roll_cmd")
    _add_mppi_time_markers(axs[0], mppi_interval, label=True)
    axs[0].set_ylabel("deg")
    axs[0].set_title("Roll")
    axs[0].grid(True)
    axs[0].legend(loc="best")

    axs[1].plot(t, np.degrees(np.unwrap(eul_pitch)), label="pitch")
    axs[1].plot(t, np.degrees(np.unwrap(cmd_pitch)), label="pitch_cmd")
    _add_mppi_time_markers(axs[1], mppi_interval)
    axs[1].set_ylabel("deg")
    axs[1].set_title("Pitch")
    axs[1].grid(True)
    axs[1].legend(loc="best")

    axs[2].plot(t, np.degrees(np.unwrap(eul_yaw)), label="yaw")
    axs[2].plot(t, np.degrees(np.unwrap(cmd_yaw)), label="yaw_cmd")
    _add_mppi_time_markers(axs[2], mppi_interval)
    axs[2].set_ylabel("deg")
    axs[2].set_title("Yaw")
    axs[2].grid(True)
    axs[2].legend(loc="best")

    axs[3].plot(t, thrust_norm, label="thrust_norm_cmd")
    axs[3].plot(t, thrust_total, label="thrust_total_cmd")
    _add_mppi_time_markers(axs[3], mppi_interval)
    axs[3].set_ylabel("thrust")
    axs[3].set_xlabel("time [s]")
    axs[3].set_title("Thrust Commands")
    axs[3].grid(True)
    axs[3].legend(loc="best")

    fig.tight_layout(rect=(0.02, 0.03, 0.99, 0.88))
    return fig


def _plot_mppi_ndo(
    t: np.ndarray,
    data: dict[str, np.ndarray],
    meta: list[str],
    mppi_interval: tuple[int, int, float, float, str] | None,
) -> plt.Figure:
    fig, axs = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    _apply_figure_header(fig, "PathFollowing: MPPI & NDO", meta)

    mppi_u0 = _get_col(data, "mppi_u0")
    mppi_u1 = _get_col(data, "mppi_u1")
    mppi_solve_time = _get_col(data, "mppi_solve_time")
    mppi_reset_flag = _get_col(data, "mppi_reset_flag")
    ndo_x = _get_col(data, "ndo_x")
    ndo_y = _get_col(data, "ndo_y")
    ndo_z = _get_col(data, "ndo_z")

    axs[0].plot(t, mppi_u0, label="mppi_u0")
    axs[0].plot(t, mppi_u1, label="mppi_u1")
    _add_mppi_time_markers(axs[0], mppi_interval, label=True)
    ax0b = axs[0].twinx()
    ax0b.step(t, mppi_reset_flag, label="mppi_reset_flag", color="tab:red", alpha=0.4, where="post")
    axs[0].set_ylabel("control")
    ax0b.set_ylabel("reset_flag")
    axs[0].set_title("MPPI Control Inputs")
    axs[0].grid(True)
    axs[0].legend(loc="upper left")
    ax0b.legend(loc="upper right")

    mppi_ms = 1000.0 * mppi_solve_time
    axs[1].plot(t, mppi_ms, label="mppi_solve_time [ms]")
    _add_mppi_time_markers(axs[1], mppi_interval)
    finite_ms = mppi_ms[np.isfinite(mppi_ms)]
    if finite_ms.size > 0:
        mean_ms = float(np.mean(finite_ms))
        axs[1].axhline(mean_ms, color="tab:red", linestyle="--", linewidth=1.0, label=f"mean={mean_ms:.2f} ms")
    axs[1].set_ylabel("ms")
    axs[1].set_title("MPPI Solve Time")
    axs[1].grid(True)
    axs[1].legend(loc="best")

    axs[2].plot(t, ndo_x, label="ndo_x")
    axs[2].plot(t, ndo_y, label="ndo_y")
    axs[2].plot(t, ndo_z, label="ndo_z")
    _add_mppi_time_markers(axs[2], mppi_interval)
    axs[2].set_ylabel("m/s^2")
    axs[2].set_xlabel("time [s]")
    axs[2].set_title("NDO Disturbance Estimate")
    axs[2].grid(True)
    axs[2].legend(loc="best")

    fig.tight_layout(rect=(0.02, 0.03, 0.99, 0.88))
    return fig


def _plot_flags_waypoints(
    t: np.ndarray,
    data: dict[str, np.ndarray],
    meta: list[str],
    mppi_interval: tuple[int, int, float, float, str] | None,
) -> plt.Figure:
    fig, axs = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    _apply_figure_header(fig, "PathFollowing: Runtime Flags & Waypoint Progress", meta)

    guid_type_cfg = _get_col(data, "guid_type_cfg")
    guid_type_used = _get_col(data, "guid_type_used")
    wp_idx_heading = _get_col(data, "wp_idx_heading")
    wp_idx_passed = _get_col(data, "wp_idx_passed")
    vehicle_nav_state = _get_col(data, "vehicle_nav_state")

    interrupt_active = _get_col(data, "interrupt_active")
    interrupt_prev = _get_col(data, "interrupt_prev")
    stop_flag = _get_col(data, "stop_flag")
    pf_done = _get_col(data, "pf_done")
    plot_complete = _get_col(data, "plot_complete")
    vehicle_is_armed = _get_col(data, "vehicle_is_armed")

    axs[0].step(t, guid_type_cfg, where="post", label="guid_type_cfg")
    axs[0].step(t, guid_type_used, where="post", label="guid_type_used")
    axs[0].step(t, wp_idx_heading, where="post", label="wp_idx_heading")
    axs[0].step(t, wp_idx_passed, where="post", label="wp_idx_passed")
    if np.isfinite(vehicle_nav_state).any():
        axs[0].step(t, vehicle_nav_state, where="post", label="vehicle_nav_state")
    _add_mppi_time_markers(axs[0], mppi_interval, label=True)
    axs[0].set_ylabel("index / type")
    axs[0].set_title("Guidance Type and Waypoint Index")
    axs[0].grid(True)
    axs[0].legend(loc="best")

    axs[1].step(t, interrupt_active, where="post", label="interrupt_active")
    axs[1].step(t, interrupt_prev, where="post", label="interrupt_prev")
    axs[1].step(t, stop_flag, where="post", label="stop_flag")
    axs[1].step(t, pf_done, where="post", label="pf_done")
    axs[1].step(t, plot_complete, where="post", label="plot_complete")
    if np.isfinite(vehicle_is_armed).any():
        axs[1].step(t, vehicle_is_armed, where="post", label="vehicle_is_armed")
    _add_mppi_time_markers(axs[1], mppi_interval)
    axs[1].set_ylabel("flag")
    axs[1].set_xlabel("time [s]")
    axs[1].set_title("Runtime Flags")
    axs[1].grid(True)
    axs[1].legend(loc="best")

    fig.tight_layout(rect=(0.02, 0.03, 0.99, 0.88))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot PathFollowing logger CSV into multiple figures. "
            "If --csv is omitted, latest CSV in --log-dir is copied to --ingest-dir and plotted."
        )
    )
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV file. Default: latest in --log-dir")
    parser.add_argument(
        "--log-dir",
        type=str,
        default=str(_default_log_dir()),
        help=(
            "Directory containing logger CSV files. "
            "Default: $PF_LOG_DIR or ~/Documents/A4VAI-SITL/ROS2/logs"
        ),
    )
    parser.add_argument(
        "--ingest-dir",
        type=str,
        default=str(_default_ingest_dir()),
        help=(
            "Directory to copy latest CSV before plotting. "
            "Default: $PF_INGEST_DIR or $PF_DATA_DIR or <repo>/pathfollowing/utils/pf_outputs"
        ),
    )
    parser.add_argument(
        "--no-copy-latest",
        action="store_true",
        help="When --csv is omitted, use latest CSV in-place (do not copy).",
    )
    parser.add_argument(
        "--sim-config",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "config" / "sim.yaml"),
        help="Path to sim.yaml for major tuning metadata annotation.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save plots. Default: <csv_dir>",
    )
    parser.add_argument("--dpi", type=int, default=160, help="PNG DPI.")
    parser.add_argument("--show", action="store_true", help="Show figures interactively.")
    parser.add_argument(
        "--flight-window",
        choices=("full", "takeoff-landing", "mppi"),
        default="full",
        help=(
            "Plot full CSV, trim to arm-through-disarm, or trim to the first detected MPPI interval."
        ),
    )
    parser.add_argument(
        "--takeoff-altitude-m",
        type=float,
        default=0.5,
        help="Fallback altitude threshold used to detect takeoff for old logs.",
    )
    parser.add_argument(
        "--landing-altitude-m",
        type=float,
        default=0.5,
        help="Fallback altitude threshold used to detect landing for old logs.",
    )
    parser.add_argument(
        "--pre-takeoff-s",
        type=float,
        default=0.0,
        help="Seconds to include before detected takeoff.",
    )
    parser.add_argument(
        "--post-landing-s",
        type=float,
        default=1.0,
        help="Seconds to include after detected landing threshold crossing.",
    )
    args = parser.parse_args()

    csv_path = _select_csv_path(
        csv_arg=args.csv,
        log_dir=Path(args.log_dir).expanduser(),
        ingest_dir=Path(args.ingest_dir).expanduser(),
        copy_latest=(not bool(args.no_copy_latest)),
    )
    if not csv_path.exists():
        raise RuntimeError(f"[FATAL] CSV not found: {csv_path}")

    save_dir = Path(args.save_dir).expanduser() if args.save_dir else csv_path.parent
    _copy_csv_to_dir(csv_path, save_dir)
    sim_params = _load_sim_params(Path(args.sim_config).expanduser())

    data = _load_csv_numeric(csv_path)
    window_detail: str | None = None
    window_suffix: str | None = None
    if args.flight_window == "takeoff-landing":
        window = _find_takeoff_landing_window(
            data=data,
            takeoff_altitude_m=float(args.takeoff_altitude_m),
            landing_altitude_m=float(args.landing_altitude_m),
            pre_takeoff_s=max(0.0, float(args.pre_takeoff_s)),
            post_landing_s=max(0.0, float(args.post_landing_s)),
        )
        if window is None:
            print("[WARN] Could not detect takeoff-landing window. Plotting full CSV.")
        else:
            start_idx, end_idx, window_detail = window
            data = _slice_data(data, start_idx, end_idx)
            window_suffix = "takeoff_landing"
            print(f"[INFO] Applied {window_detail}")
    elif args.flight_window == "mppi":
        window = _find_mppi_window(data)
        if window is None:
            print("[WARN] Could not detect MPPI window. Plotting full CSV.")
        else:
            start_idx, end_idx, window_detail = window
            data = _slice_data(data, start_idx, end_idx)
            window_suffix = "mppi"
            print(f"[INFO] Applied {window_detail}")

    t = _get_col(data, "time")
    if not np.isfinite(t).any():
        n = len(next(iter(data.values())))
        t = np.arange(n, dtype=float)

    meta = _meta_lines(csv_path, data, sim_params)
    if window_detail is not None:
        meta.append(window_detail)

    mppi_interval = _find_mppi_interval(data)
    if mppi_interval is not None:
        _, _, mppi_start, mppi_end, mppi_source = mppi_interval
        mppi_detail = (
            "mppi: "
            f"t={_fmt_scalar(mppi_start)}..{_fmt_scalar(mppi_end)}s "
            f"source={mppi_source}"
        )
        meta.append(mppi_detail)
        print(f"[INFO] Applied {mppi_detail}")

    fig1 = _plot_trajectory_error(t, data, meta, mppi_interval)
    fig2 = _plot_attitude_thrust(t, data, meta, mppi_interval)
    fig3 = _plot_mppi_ndo(t, data, meta, mppi_interval)
    fig4 = _plot_flags_waypoints(t, data, meta, mppi_interval)

    stem = csv_path.stem
    if window_suffix is not None:
        stem = f"{stem}_{window_suffix}"
    out1 = _save_figure(fig1, save_dir, stem, "fig1_traj_error", args.dpi)
    out2 = _save_figure(fig2, save_dir, stem, "fig2_att_thrust", args.dpi)
    out3 = _save_figure(fig3, save_dir, stem, "fig3_mppi_ndo", args.dpi)
    out4 = _save_figure(fig4, save_dir, stem, "fig4_flags_wp", args.dpi)

    print(f"[INFO] CSV: {csv_path}")
    print(f"[INFO] saved: {out1}")
    print(f"[INFO] saved: {out2}")
    print(f"[INFO] saved: {out3}")
    print(f"[INFO] saved: {out4}")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
