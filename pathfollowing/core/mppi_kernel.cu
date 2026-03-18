////////////////////////////////////////////////////////////
//
//   - Name : mppi_kernel.cu
//
//                                 - KAIST FDCL, 2026.03.11
//
////////////////////////////////////////////////////////////

#include <math.h>
#include <float.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Real N_WP is injected dynamically from core_mppi.py before PyCUDA compilation.
#ifndef N_WP
#define N_WP 1
#endif

#ifndef EPS_USER
#define EPS_USER 1.0e-9f
#endif

#define EPS_NORM EPS_USER
#define EPS_DIV EPS_USER
#define EPS_DEN EPS_USER
#define MIN_SEGMENT_LENGTH_M EPS_USER
#define MIN_TIME_CONSTANT_S EPS_USER
#define MIN_TAU_THROTTLE_S EPS_USER

__device__ inline int i_min_(int a, int b) { return (a < b) ? a : b; }
__device__ inline int i_max_(int a, int b) { return (a > b) ? a : b; }
__device__ inline int i_clamp_(int x, int lo, int hi) { return i_min_(i_max_(x, lo), hi); }

__device__ inline float clamp_(float x, float lo, float hi)
{
    return fminf(fmaxf(x, lo), hi);
}

__device__ inline float wrap_pi_(float x)
{
    while (x > M_PI) x -= 2.0 * M_PI;
    while (x < -M_PI) x += 2.0 * M_PI;
    return x;
}

__device__ inline float norm3_(const float x[3])
{
    return sqrtf(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
}

__device__ inline float norm_xy_(const float x[3])
{
    return sqrtf(x[0] * x[0] + x[1] * x[1]);
}

__device__ inline float dot3_(const float a[3], const float b[3])
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

__device__ inline void sub3_(const float a[3], const float b[3], float out[3])
{
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = a[2] - b[2];
}

__device__ inline void cross3_(const float a[3], const float b[3], float out[3])
{
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

__device__ inline void copy3_(const float a[3], float out[3])
{
    out[0] = a[0];
    out[1] = a[1];
    out[2] = a[2];
}

__device__ inline void normalize3_(float v[3])
{
    float n = norm3_(v);
    if (n < EPS_NORM) {
        v[0] = 1.0;
        v[1] = 0.0;
        v[2] = 0.0;
        return;
    }
    v[0] /= n;
    v[1] /= n;
    v[2] /= n;
}

__device__ inline void normalize_xy_(float v[3])
{
    float n = sqrtf(v[0] * v[0] + v[1] * v[1]);
    if (n < EPS_NORM) {
        v[0] = 1.0;
        v[1] = 0.0;
        v[2] = 0.0;
        return;
    }
    v[0] /= n;
    v[1] /= n;
    v[2] = 0.0;
}

__device__ inline void azim_elev_from_vec3_(const float vec[3], float* azim, float* elev)
{
    *azim = atan2f(vec[1], vec[0]);
    *elev = atan2f(-vec[2], sqrtf(vec[0] * vec[0] + vec[1] * vec[1]));
}

__device__ inline void dcm_from_euler321_(const float euler[3], float dcm[3][3])
{
    float phi = euler[0];
    float the = euler[1];
    float psi = euler[2];

    float spsi = sinf(psi);
    float cpsi = cosf(psi);
    float sthe = sinf(the);
    float cthe = cosf(the);
    float sphi = sinf(phi);
    float cphi = cosf(phi);

    dcm[0][0] = cpsi * cthe;
    dcm[1][0] = cpsi * sthe * sphi - spsi * cphi;
    dcm[2][0] = cpsi * sthe * cphi + spsi * sphi;

    dcm[0][1] = spsi * cthe;
    dcm[1][1] = spsi * sthe * sphi + cpsi * cphi;
    dcm[2][1] = spsi * sthe * cphi - cpsi * sphi;

    dcm[0][2] = -sthe;
    dcm[1][2] = cthe * sphi;
    dcm[2][2] = cthe * cphi;
}

__device__ inline void transpose3x3_(const float a[3][3], float out[3][3])
{
    out[0][0] = a[0][0];
    out[0][1] = a[1][0];
    out[0][2] = a[2][0];

    out[1][0] = a[0][1];
    out[1][1] = a[1][1];
    out[1][2] = a[2][1];

    out[2][0] = a[0][2];
    out[2][1] = a[1][2];
    out[2][2] = a[2][2];
}

__device__ inline void matvec3_(const float a[3][3], const float x[3], float out[3])
{
    out[0] = a[0][0] * x[0] + a[0][1] * x[1] + a[0][2] * x[2];
    out[1] = a[1][0] * x[0] + a[1][1] * x[1] + a[1][2] * x[2];
    out[2] = a[2][0] * x[0] + a[2][1] * x[1] + a[2][2] * x[2];
}

__device__ inline void get_wp_(const float* wps, int idx, float out[3])
{
    out[0] = wps[3 * idx + 0];
    out[1] = wps[3 * idx + 1];
    out[2] = wps[3 * idx + 2];
}

__device__ void distance_to_path_(
    const float* wps,
    int n_wp,
    int wp_idx_heading,
    const float pos[3],
    float closest_on_path[3],
    int* wp_idx_passed,
    float* dist_to_path
)
{
    if (n_wp <= 1) {
        get_wp_(wps, 0, closest_on_path);
        float rel[3];
        sub3_(closest_on_path, pos, rel);
        *dist_to_path = norm3_(rel);
        *wp_idx_passed = 0;
        return;
    }

    float dist = FLT_MAX;
    bool updated = false;
    int passed_prev = *wp_idx_passed;
    int heading = i_clamp_(wp_idx_heading, 1, n_wp - 1);

    for (int i_wp = heading; i_wp > i_max_(passed_prev - 1, 0); --i_wp) {
        float wp1[3], wp2[3];
        get_wp_(wps, i_wp - 1, wp1);
        get_wp_(wps, i_wp, wp2);

        float rw1w2[3];
        sub3_(wp2, wp1, rw1w2);
        float mag_rw1w2 = norm3_(rw1w2);

        float rw1q[3];
        sub3_(pos, wp1, rw1q);

        float mag_w1p = dot3_(rw1w2, rw1q) / fmaxf(mag_rw1w2, MIN_SEGMENT_LENGTH_M);
        mag_w1p = clamp_(mag_w1p, 0.0, mag_rw1w2);

        float p_closest[3] = {
            wp1[0] + mag_w1p * rw1w2[0] / fmaxf(mag_rw1w2, MIN_SEGMENT_LENGTH_M),
            wp1[1] + mag_w1p * rw1w2[1] / fmaxf(mag_rw1w2, MIN_SEGMENT_LENGTH_M),
            wp1[2] + mag_w1p * rw1w2[2] / fmaxf(mag_rw1w2, MIN_SEGMENT_LENGTH_M),
        };

        float rel[3];
        sub3_(p_closest, pos, rel);
        float mag_rqp = norm3_(rel);

        if (dist > mag_rqp) {
            dist = mag_rqp;
            copy3_(p_closest, closest_on_path);
            *wp_idx_passed = i_max_(i_wp - 1, passed_prev);
            updated = true;
        } else if (i_wp == (passed_prev + 1) && !updated) {
            dist = mag_rqp;
            copy3_(p_closest, closest_on_path);
            *wp_idx_passed = passed_prev;
        }
    }

    *dist_to_path = dist;
}

__device__ void check_waypoint_(
    const float* wps,
    int n_wp,
    int* wp_idx_heading,
    const float pos[3],
    float distance_change_wp
)
{
    int idx = i_clamp_(*wp_idx_heading, 0, n_wp - 1);
    float wp_h[3];
    get_wp_(wps, idx, wp_h);

    float rel[3];
    sub3_(wp_h, pos, rel);
    float mag = norm3_(rel);

    if (mag < distance_change_wp) {
        *wp_idx_heading = i_min_(idx + 1, n_wp - 1);
    }
}

__device__ void vtp_decision_(
    const float* wps,
    int n_wp,
    float dist_to_path,
    float virtual_target_distance,
    const float closest_on_path[3],
    int wp_idx_passed,
    float vt[3]
)
{
    if (dist_to_path >= virtual_target_distance) {
        copy3_(closest_on_path, vt);
        return;
    }

    float total_len = dist_to_path;
    float p1[3];
    copy3_(closest_on_path, p1);

    for (int i_wp = wp_idx_passed + 1; i_wp < n_wp; ++i_wp) {
        float p2[3];
        get_wp_(wps, i_wp, p2);

        float rp1p2[3];
        sub3_(p2, p1, rp1p2);
        float mag = norm3_(rp1p2);

        if (total_len + mag > virtual_target_distance) {
            float mag_rp1t = virtual_target_distance - total_len;
            vt[0] = p1[0] + mag_rp1t * rp1p2[0] / fmaxf(mag, MIN_SEGMENT_LENGTH_M);
            vt[1] = p1[1] + mag_rp1t * rp1p2[1] / fmaxf(mag, MIN_SEGMENT_LENGTH_M);
            vt[2] = p1[2] + mag_rp1t * rp1p2[2] / fmaxf(mag, MIN_SEGMENT_LENGTH_M);
            return;
        }

        copy3_(p2, p1);
        total_len += mag;
        if (i_wp == n_wp - 1) {
            copy3_(p2, vt);
            return;
        }
    }

    copy3_(closest_on_path, vt);
}

__device__ void guidance_modules_(
    int guid_type,
    int wp_idx_passed,
    int wp_idx_heading,
    int n_wp,
    const float vt[3],
    const float pos[3],
    const float vel[3],
    const float acc[3],
    float guid_eta,
    float transition_speed_threshold_mps,
    float position_kp,
    float position_kv_multiplier,
    float min_guid_eta,
    float gl_vertical_speed_gain,
    const float mppi_ctrl_input[2],
    float acc_cmd[3]
)
{
    (void)acc;

    // starting/terminal phase
    if (wp_idx_passed < 1 && norm3_(vel) <= transition_speed_threshold_mps) {
        guid_type = 0;
    }
    if (wp_idx_heading >= (n_wp - 1)) {
        guid_type = 0;
    }

    if (guid_type == 0) {
        float pos_error_ned[3] = {vt[0] - pos[0], vt[1] - pos[1], vt[2] - pos[2]};
        float kp_pos = position_kp;
        float vel_cmd_ned[3] = {
            kp_pos * pos_error_ned[0],
            kp_pos * pos_error_ned[1],
            kp_pos * pos_error_ned[2]
        };
        float vel_error_ned[3] = {
            vel_cmd_ned[0] - vel[0],
            vel_cmd_ned[1] - vel[1],
            vel_cmd_ned[2] - vel[2]
        };
        float kp_vel = position_kv_multiplier * kp_pos;
        acc_cmd[0] = kp_vel * vel_error_ned[0];
        acc_cmd[1] = kp_vel * vel_error_ned[1];
        acc_cmd[2] = kp_vel * vel_error_ned[2];
        return;
    }

    guid_eta = fmaxf(mppi_ctrl_input[1], min_guid_eta);
    float speed_mag = norm3_(vel);

    float fpa_azim = 0.0, fpa_elev = 0.0;
    azim_elev_from_vec3_(vel, &fpa_azim, &fpa_elev);
    float fpa_euler[3] = {0.0, fpa_elev, fpa_azim};

    float dcm_ned_to_wind[3][3];
    dcm_from_euler321_(fpa_euler, dcm_ned_to_wind);

    float acc_cmd_wind[3] = {0.0, 0.0, 0.0};
    acc_cmd_wind[0] = mppi_ctrl_input[0];

    float vec_to_target_ned[3] = {vt[0] - pos[0], vt[1] - pos[1], vt[2] - pos[2]};
    float vec_to_target_wind[3];
    matvec3_(dcm_ned_to_wind, vec_to_target_ned, vec_to_target_wind);

    float err_azim = 0.0, err_elev = 0.0;
    azim_elev_from_vec3_(vec_to_target_wind, &err_azim, &err_elev);

    acc_cmd_wind[1] = guid_eta * speed_mag * sinf(err_azim);
    acc_cmd_wind[2] = -gl_vertical_speed_gain * speed_mag * sinf(err_elev);

    float dcm_wind_to_ned[3][3];
    transpose3x3_(dcm_ned_to_wind, dcm_wind_to_ned);
    matvec3_(dcm_wind_to_ned, acc_cmd_wind, acc_cmd);
    // acc_cmd[2] = -guid_eta * speed_mag * sinf(err_elev);
}

__device__ void convert_acc_cmd_to_thrust_and_att_(
    const float cI_B[3][3],
    const float acc_cmd[3],
    float mass,
    float t_max,
    const float* wps,
    int n_wp,
    int wp_idx_heading,
    const float pos[3],
    const float att_ang[3],
    float del_psi_cmd_limit,
    float max_tilt_rad,
    float* t_cmd,
    float att_cmd[3]
)
{
    float acc_cmd_mag = norm3_(acc_cmd);

    float acc_cmd_body[3];
    matvec3_(cI_B, acc_cmd, acc_cmd_body);
    *t_cmd = fminf(fabsf(acc_cmd_body[2]) * mass, t_max);

    int heading_idx = i_clamp_(wp_idx_heading, 0, n_wp - 1);
    float wp_heading[3];
    get_wp_(wps, heading_idx, wp_heading);

    float vec_to_heading_wp[3];
    sub3_(wp_heading, pos, vec_to_heading_wp);

    float psi_des = 0.0, tmp = 0.0;
    if (heading_idx < n_wp - 1) {
        azim_elev_from_vec3_(vec_to_heading_wp, &psi_des, &tmp);
    } else {
        int wp_idx_passed = i_max_(heading_idx - 1, 0);
        float wp_passed[3];
        get_wp_(wps, wp_idx_passed, wp_passed);
        float wp12[3];
        sub3_(wp_heading, wp_passed, wp12);
        azim_elev_from_vec3_(wp12, &psi_des, &tmp);
    }

    float del_psi = wrap_pi_(psi_des - att_ang[2]);
    del_psi = clamp_(del_psi, -del_psi_cmd_limit, del_psi_cmd_limit);
    psi_des = att_ang[2] + del_psi;

    float yaw_euler[3] = {0.0, 0.0, psi_des};
    float dcm_yaw_only[3][3];
    dcm_from_euler321_(yaw_euler, dcm_yaw_only);

    float yaw_aligned_acc_cmd[3];
    matvec3_(dcm_yaw_only, acc_cmd, yaw_aligned_acc_cmd);

    if (acc_cmd_mag < EPS_DIV) {
        att_cmd[0] = 0.0;
        att_cmd[1] = 0.0;
        att_cmd[2] = psi_des;
        return;
    }

    att_cmd[0] = clamp_(asinf(yaw_aligned_acc_cmd[1] / acc_cmd_mag), -max_tilt_rad, max_tilt_rad);
    float sin_tilt_limit = sinf(max_tilt_rad);
    float sin_pitch = clamp_(
        -yaw_aligned_acc_cmd[0] / fmaxf(cosf(att_cmd[0]) * acc_cmd_mag, EPS_DIV),
        -sin_tilt_limit,
        sin_tilt_limit
    );
    att_cmd[1] = asinf(sin_pitch);
    att_cmd[2] = psi_des;
}

__device__ void attitude_controller_(
    const float att_cmd_in[3],
    const float att[3],
    const float wb[3],
    float tau_phi,
    float tau_the,
    float tau_psi,
    float wb_cmd[3]
)
{
    float att_cmd[3] = {att_cmd_in[0], att_cmd_in[1], att_cmd_in[2]};

    if (fabsf(att_cmd[2] - att[2]) > M_PI) {
        if (att_cmd[2] > att[2]) {
            att_cmd[2] -= 2.0 * M_PI;
        } else {
            att_cmd[2] += 2.0 * M_PI;
        }
    }

    float desired_dot_att[3] = {0.0, 0.0, 0.0};
    desired_dot_att[0] = (att_cmd[0] - att[0]) / fmaxf(tau_phi, EPS_DEN);
    desired_dot_att[1] = (att_cmd[1] - att[1]) / fmaxf(tau_the, EPS_DEN);
    desired_dot_att[2] = (att_cmd[2] - att[2]) / fmaxf(tau_psi, EPS_DEN);

    float wb_model[3] = {wb[0], wb[1], wb[2]};
    if ((wb_model[0] == 0.0) && (wb_model[1] == 0.0) && (wb_model[2] == 0.0)) {
        wb_model[0] = desired_dot_att[0] - desired_dot_att[2] * sinf(att[1]);
        wb_model[1] = desired_dot_att[1] * cosf(att[0]) + desired_dot_att[2] * sinf(att[0]) * cosf(att[1]);
        wb_model[2] = -desired_dot_att[1] * sinf(att[0]) + desired_dot_att[2] * cosf(att[0]) * cosf(att[1]);
    }

    float cthe = cosf(att[1]);
    float cthe_safe = cthe;
    if (fabsf(cthe_safe) < EPS_DEN) cthe_safe = (cthe_safe >= 0.0) ? EPS_DEN : -EPS_DEN;
    float sthe = 1.0 / cthe_safe;
    float tthe = tanf(att[1]);
    float sphi = sinf(att[0]);
    float cphi = cosf(att[0]);
    float tphi = tanf(att[0]);

    float p_trim = -wb_model[1] * sphi * tthe - wb_model[2] * cphi * tthe;
    wb_cmd[0] = desired_dot_att[0] + p_trim;

    float q_trim = wb_model[2] * tphi;
    float den_q = cphi;
    if (fabsf(den_q) < EPS_DEN) den_q = (den_q >= 0.0) ? EPS_DEN : -EPS_DEN;
    wb_cmd[1] = desired_dot_att[1] / den_q + q_trim;

    float r_trim = -wb_model[1] * tphi;
    float den_r = cphi * sthe;
    if (fabsf(den_r) < EPS_DEN) den_r = (den_r >= 0.0) ? EPS_DEN : -EPS_DEN;
    wb_cmd[2] = desired_dot_att[2] / den_r + r_trim;
}

__device__ void rate_controller_(
    float wb_cmd[3],
    const float wb[3],
    float tau_wb,
    float dt,
    float err_wb[3],
    float int_err_wb[3]
)
{
    float lim_wb = (2.0 / fmaxf(tau_wb, EPS_DEN)) * (M_PI / 180.0);
    float norm_wb_cmd = norm3_(wb_cmd);
    if (norm_wb_cmd > lim_wb) {
        wb_cmd[0] = wb_cmd[0] / norm_wb_cmd * lim_wb;
        wb_cmd[1] = wb_cmd[1] / norm_wb_cmd * lim_wb;
        wb_cmd[2] = wb_cmd[2] / norm_wb_cmd * lim_wb;
    }

    for (int i = 0; i < 3; ++i) {
        err_wb[i] = wb_cmd[i] - wb[i];
        int_err_wb[i] = int_err_wb[i] + err_wb[i] * dt;
    }
}

__device__ void dynamics_equations_(
    const float cI_B[3][3],
    const float cB_I[3][3],
    float t_cmd,
    float mass,
    const float disturbance_acc_ned[3],
    float gravity,
    const float zeta_wb[3],
    const float omega_wb[3],
    const float err_wb[3],
    const float int_err_wb[3],
    const float wb[3],
    const float vb[3],
    const float att[3],
    float dot_ri[3],
    float dot_vb[3],
    float dot_att[3],
    float dot_wb[3]
)
{
    float ab_thrust[3] = {0.0, 0.0, -t_cmd / fmaxf(mass, EPS_DEN)};
    float ab_disturbance[3];
    matvec3_(cI_B, disturbance_acc_ned, ab_disturbance);
    float ai_grav[3] = {0.0, 0.0, gravity};
    float ab_grav[3];
    matvec3_(cI_B, ai_grav, ab_grav);

    float ab_total[3] = {
        ab_thrust[0] + ab_disturbance[0] + ab_grav[0],
        ab_thrust[1] + ab_disturbance[1] + ab_grav[1],
        ab_thrust[2] + ab_disturbance[2] + ab_grav[2]
    };

    float wb_cross_vb[3];
    cross3_(wb, vb, wb_cross_vb);

    for (int i = 0; i < 3; ++i) {
        dot_vb[i] = -wb_cross_vb[i] + ab_total[i];
        dot_wb[i] = 2.0 * zeta_wb[i] * omega_wb[i] * err_wb[i] + omega_wb[i] * omega_wb[i] * int_err_wb[i];
    }

    matvec3_(cB_I, vb, dot_ri);

    float cthe = cosf(att[1]);
    float cthe_safe = cthe;
    if (fabsf(cthe_safe) < EPS_DEN) cthe_safe = (cthe_safe >= 0.0) ? EPS_DEN : -EPS_DEN;
    float sthe = 1.0 / cthe_safe;
    float tthe = tanf(att[1]);
    float sphi = sinf(att[0]);
    float cphi = cosf(att[0]);

    dot_att[0] = wb[0] + wb[1] * sphi * tthe + wb[2] * cphi * tthe;
    dot_att[1] = wb[1] * cphi - wb[2] * sphi;
    dot_att[2] = wb[1] * sphi * sthe + wb[2] * cphi * sthe;
}

__device__ void update_states_(
    const float dot_ri[3],
    const float dot_vb[3],
    const float dot_att[3],
    const float dot_wb[3],
    float dt,
    float ri[3],
    float vb[3],
    float att[3],
    float wb[3],
    float cI_B[3][3],
    float cB_I[3][3],
    float vi[3]
)
{
    for (int i = 0; i < 3; ++i) {
        ri[i] += dot_ri[i] * dt;
        vb[i] += dot_vb[i] * dt;
        att[i] = wrap_pi_(att[i] + dot_att[i] * dt);
        wb[i] += dot_wb[i] * dt;
    }

    dcm_from_euler321_(att, cI_B);
    transpose3x3_(cI_B, cB_I);
    matvec3_(cB_I, vb, vi);
}

__device__ void cost_function_1_(
    const float r[3],
    const float mppi_ctrl_input[2],
    float q0,
    float dist_to_path,
    float q1,
    const float att_ang[3],
    float weight_by_var,
    float dt,
    float cost_arr[3]
)
{
    float u_ru = r[0] * (mppi_ctrl_input[0] * mppi_ctrl_input[0]) +
                  r[1] * (mppi_ctrl_input[1] * mppi_ctrl_input[1]);

    float x0 = dist_to_path;
    float x0_q0_x0 = x0 * q0 * x0;

    float x1 = sqrtf(att_ang[0] * att_ang[0] + att_ang[1] * att_ang[1]);
    float x1_q1_x1 = x1 * q1 * x1;

    cost_arr[0] = u_ru * dt;
    cost_arr[1] = x0_q0_x0 * dt;
    cost_arr[2] = weight_by_var * x1_q1_x1 * dt;
}

__device__ void cost_function_2_(
    const float r[3],
    const float mppi_ctrl_input[2],
    const float q[3],
    float dist_to_path,
    float path_error_scale,
    float dt,
    const float delta_u[2],
    float mag_v,
    float ref_v,
    float gamma,
    const float att_ang[3],
    float weight_by_var,
    float cost_arr[3]
)
{
    float u0 = mppi_ctrl_input[0];
    float u1 = mppi_ctrl_input[1];
    float du0 = delta_u[0];
    float du1 = delta_u[1];

    float u_r_du = gamma * (((r[0] * (u0 - du0)) * du0) + ((r[1] * (u1 - du1)) * du1));

    float x0 = fminf(dist_to_path / fmaxf(path_error_scale, EPS_DIV), 1.0);
    float x0_q0_x0 = x0 * q[0] * x0;

    float x1 = fminf(fabsf(mag_v - ref_v) / fmaxf(ref_v, EPS_DIV), 1.0);
    float x1_q1_x1 = x1 * q[1] * x1;

    float x2 = sqrtf(att_ang[0] * att_ang[0] + att_ang[1] * att_ang[1]);
    float x2_q2_x2 = x2 * q[2] * x2;

    cost_arr[0] = u_r_du * dt;
    cost_arr[1] = (x0_q0_x0 + x1_q1_x1) * dt;
    cost_arr[2] = (weight_by_var * x2_q2_x2) * dt;
}

__device__ float terminal_cost_2_(
    const float p[3],
    const float vel_tf[3],
    float dist_to_path,
    float ref_v,
    float path_error_scale
)
{
    float mag_v = sqrtf(vel_tf[0] * vel_tf[0] + vel_tf[1] * vel_tf[1]);
    float x0 = fminf(dist_to_path / fmaxf(path_error_scale, EPS_DIV), 1.0);
    float x1 = fminf(fabsf(mag_v - ref_v) / fmaxf(ref_v, EPS_DIV), 1.0);

    return p[1] * x0 * x0 + p[2] * x1 * x1;
}

__device__ float terminal_cost_1_(
    float p1,
    const float* wps,
    int n_wp,
    int init_wp_idx_passed,
    int final_wp_idx_passed,
    const float init_closest[3],
    const float final_closest[3],
    float min_move_range,
    float total_time
)
{
    float init_remained = 0.0;
    for (int i_wp = init_wp_idx_passed; i_wp < n_wp - 1; ++i_wp) {
        float wp_a[3], wp_b[3], rel[3];
        get_wp_(wps, i_wp, wp_a);
        get_wp_(wps, i_wp + 1, wp_b);
        sub3_(wp_b, wp_a, rel);
        init_remained += norm3_(rel);
    }
    {
        float wp_init[3], rel[3];
        get_wp_(wps, i_clamp_(init_wp_idx_passed, 0, n_wp - 1), wp_init);
        sub3_(init_closest, wp_init, rel);
        init_remained -= norm3_(rel);
    }

    float final_remained = 0.0;
    for (int i_wp = final_wp_idx_passed; i_wp < n_wp - 1; ++i_wp) {
        float wp_a[3], wp_b[3], rel[3];
        get_wp_(wps, i_wp, wp_a);
        get_wp_(wps, i_wp + 1, wp_b);
        sub3_(wp_b, wp_a, rel);
        final_remained += norm3_(rel);
    }
    {
        float wp_final[3], rel[3];
        get_wp_(wps, i_clamp_(final_wp_idx_passed, 0, n_wp - 1), wp_final);
        sub3_(final_closest, wp_final, rel);
        final_remained -= norm3_(rel);
    }

    float move_range = init_remained - final_remained;
    return p1 * total_time / fmaxf(move_range, min_move_range);
}

extern "C" __global__ void mppi_rollout_kernel(
    float* arr_u0,
    float* arr_u1,
    float* arr_delta_u0,
    float* arr_delta_u1,
    float* arr_stk,
    float* arr_const,
    float* arr_update,
    float* arr_waypoints_ned,
    float* arr_disturbance_acc_est,
    float* arr_disturbance_acc_var
)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    int K = (int)arr_const[0];
    if (idx >= K) return;

    int N = (int)arr_const[1];
    float dt = arr_const[2];
    float gamma = arr_const[3];

    float R[3] = {arr_const[4], arr_const[5], arr_const[6]};
    float Q[3] = {arr_const[7], arr_const[8], arr_const[9]};
    float P[3] = {arr_const[10], arr_const[11], arr_const[12]};

    float mass = arr_const[13];
    float gravity = arr_const[14];
    float waypoint_switch_distance = arr_const[15];
    float desired_speed_const = arr_const[16];
    float lookahead_distance_const = arr_const[17];
    float guid_eta_const = arr_const[18];
    float del_psi_cmd_limit = arr_const[19];
    float tau_phi = fmaxf(arr_const[20], MIN_TIME_CONSTANT_S);
    float tau_the = fmaxf(arr_const[21], MIN_TIME_CONSTANT_S);
    float tau_psi = fmaxf(arr_const[22], MIN_TIME_CONSTANT_S);
    float cost_min_v_aligned = arr_const[23];
    float t_max = arr_const[25];
    float max_tilt_rad = arr_const[27];
    float transition_speed_threshold_mps = arr_const[29];
    float position_kp = arr_const[30];
    float position_kv_multiplier = arr_const[31];
    float min_guid_eta = arr_const[32];
    float gl_vertical_speed_gain = arr_const[33];
    float tau_wb = fmaxf(arr_const[34], MIN_TIME_CONSTANT_S);
    float tau_p = fmaxf(arr_const[35], MIN_TIME_CONSTANT_S);
    float tau_q = fmaxf(arr_const[36], MIN_TIME_CONSTANT_S);
    float tau_r = fmaxf(arr_const[37], MIN_TIME_CONSTANT_S);
    float alpha_p = fmaxf(arr_const[38], EPS_DEN);
    float alpha_q = fmaxf(arr_const[39], EPS_DEN);
    float alpha_r = fmaxf(arr_const[40], EPS_DEN);
    float tau_throttle = fmaxf(arr_const[41], MIN_TAU_THROTTLE_S);
    float disturbance_var_weight_gain = arr_const[42];
    float disturbance_var_clip = arr_const[43];
    float cost_path_error_scale_m = fmaxf(arr_const[44], EPS_DEN);

    int wp_idx_heading = (int)arr_update[0];
    int wp_idx_passed = (int)arr_update[1];
    int guid_type = (int)arr_update[2];

    float pos[3] = {arr_update[3], arr_update[4], arr_update[5]};
    float vel[3] = {arr_update[6], arr_update[7], arr_update[8]};
    float att[3] = {arr_update[9], arr_update[10], arr_update[11]};
    float t_cmd = arr_update[12];

    float desired_speed = arr_update[13];
    if (desired_speed <= 0.0) desired_speed = desired_speed_const;
    float virtual_target_distance = arr_update[14];
    if (virtual_target_distance <= 0.0) virtual_target_distance = lookahead_distance_const;

    int n_wp = N_WP;
    wp_idx_heading = i_clamp_(wp_idx_heading, 0, n_wp - 1);
    wp_idx_passed = i_clamp_(wp_idx_passed, 0, n_wp - 1);

    float total_cost = 0.0;

    float acc[3] = {0.0, 0.0, 0.0};
    float cI_B[3][3];
    float cB_I[3][3];
    dcm_from_euler321_(att, cI_B);
    transpose3x3_(cI_B, cB_I);
    float vb[3];
    matvec3_(cI_B, vel, vb);
    float wb[3] = {0.0, 0.0, 0.0};
    float err_wb[3] = {0.0, 0.0, 0.0};
    float int_err_wb[3] = {0.0, 0.0, 0.0};
    float zeta_wb[3] = {
        0.5f * sqrtf(alpha_p / tau_p),
        0.5f * sqrtf(alpha_q / tau_q),
        0.5f * sqrtf(alpha_r / tau_r),
    };
    float omega_wb[3] = {
        sqrtf(1.0f / (alpha_p * tau_p)),
        sqrtf(1.0f / (alpha_q * tau_q)),
        sqrtf(1.0f / (alpha_r * tau_r)),
    };
    float closest_on_path[3] = {0.0, 0.0, 0.0};
    float vt[3] = {0.0, 0.0, 0.0};
    float dist_to_path = 0.0;

    int init_wp_idx_passed = wp_idx_passed;
    float init_closest[3] = {0.0, 0.0, 0.0};
    bool init_set = false;

    int final_wp_idx_passed = wp_idx_passed;
    float final_closest[3] = {0.0, 0.0, 0.0};

    for (int i_n = 0; i_n < N; ++i_n) {
        float disturbance_acc_est[3] = {
            arr_disturbance_acc_est[3 * i_n + 0],
            arr_disturbance_acc_est[3 * i_n + 1],
            arr_disturbance_acc_est[3 * i_n + 2]
        };
        float disturbance_acc_var = arr_disturbance_acc_var[i_n];

        float delta_u[2] = {0.0, 0.0};
        float mppi_ctrl_input[2] = {0.0, 0.0};
        if (guid_type >= 1) {
            delta_u[0] = arr_delta_u0[idx + K * i_n];
            delta_u[1] = arr_delta_u1[idx + K * i_n];
            mppi_ctrl_input[0] = arr_u0[i_n] + delta_u[0];
            mppi_ctrl_input[1] = arr_u1[i_n] + delta_u[1];
        }

        distance_to_path_(arr_waypoints_ned, n_wp, wp_idx_heading, pos, closest_on_path, &wp_idx_passed, &dist_to_path);
        check_waypoint_(arr_waypoints_ned, n_wp, &wp_idx_heading, pos, waypoint_switch_distance);
        vtp_decision_(arr_waypoints_ned, n_wp, dist_to_path, virtual_target_distance, closest_on_path, wp_idx_passed, vt);

        if (!init_set) {
            init_wp_idx_passed = wp_idx_passed;
            copy3_(closest_on_path, init_closest);
            init_set = true;
        }

        float acc_cmd[3] = {0.0, 0.0, 0.0};
        guidance_modules_(
            guid_type,
            wp_idx_passed,
            wp_idx_heading,
            n_wp,
            vt,
            pos,
            vel,
            acc,
            guid_eta_const,
            transition_speed_threshold_mps,
            position_kp,
            position_kv_multiplier,
            min_guid_eta,
            gl_vertical_speed_gain,
            mppi_ctrl_input,
            acc_cmd
        );

        float disturbance_acc[3] = {
            disturbance_acc_est[0],
            disturbance_acc_est[1],
            disturbance_acc_est[2]
        };

        float acc_cmd_comp[3] = {
            acc_cmd[0] - disturbance_acc[0],
            acc_cmd[1] - disturbance_acc[1],
            acc_cmd[2] - disturbance_acc[2] - gravity
        };

        float att_cmd[3] = {0.0, 0.0, 0.0};
        float t_cmd_ref = t_cmd;
        convert_acc_cmd_to_thrust_and_att_(
            cI_B,
            acc_cmd_comp,
            mass,
            t_max,
            arr_waypoints_ned,
            n_wp,
            wp_idx_heading,
            pos,
            att,
            del_psi_cmd_limit,
            max_tilt_rad,
            &t_cmd_ref,
            att_cmd
        );

        float mag_v = norm_xy_(vel);
        float weight_by_var = disturbance_var_weight_gain * fminf(disturbance_acc_var, disturbance_var_clip);
        float cost_arr[3] = {0.0, 0.0, 0.0};

        if (guid_type == 2) {
            cost_function_2_(
                R,
                mppi_ctrl_input,
                Q,
                dist_to_path,
                cost_path_error_scale_m,
                dt,
                delta_u,
                mag_v,
                desired_speed,
                gamma,
                att,
                weight_by_var,
                cost_arr
            );
        } else {
            cost_function_1_(
                R,
                mppi_ctrl_input,
                Q[0],
                dist_to_path,
                Q[1],
                att,
                weight_by_var,
                dt,
                cost_arr
            );
        }

        total_cost += cost_arr[0] + cost_arr[1] + cost_arr[2];

        t_cmd = t_cmd + (t_cmd_ref - t_cmd) / tau_throttle * dt;
        t_cmd = clamp_(t_cmd, 0.0, t_max);

        float wb_cmd[3] = {0.0, 0.0, 0.0};
        attitude_controller_(
            att_cmd,
            att,
            wb,
            tau_phi,
            tau_the,
            tau_psi,
            wb_cmd
        );
        rate_controller_(
            wb_cmd,
            wb,
            tau_wb,
            dt,
            err_wb,
            int_err_wb
        );

        float dot_ri[3] = {0.0, 0.0, 0.0};
        float dot_vb[3] = {0.0, 0.0, 0.0};
        float dot_att[3] = {0.0, 0.0, 0.0};
        float dot_wb[3] = {0.0, 0.0, 0.0};
        float vel_prev[3] = {vel[0], vel[1], vel[2]};

        dynamics_equations_(
            cI_B,
            cB_I,
            t_cmd,
            mass,
            disturbance_acc,
            gravity,
            zeta_wb,
            omega_wb,
            err_wb,
            int_err_wb,
            wb,
            vb,
            att,
            dot_ri,
            dot_vb,
            dot_att,
            dot_wb
        );

        update_states_(
            dot_ri,
            dot_vb,
            dot_att,
            dot_wb,
            dt,
            pos,
            vb,
            att,
            wb,
            cI_B,
            cB_I,
            vel
        );

        acc[0] = (vel[0] - vel_prev[0]) / fmaxf(dt, EPS_DIV);
        acc[1] = (vel[1] - vel_prev[1]) / fmaxf(dt, EPS_DIV);
        acc[2] = (vel[2] - vel_prev[2]) / fmaxf(dt, EPS_DIV);

        final_wp_idx_passed = wp_idx_passed;
        copy3_(closest_on_path, final_closest);
    }

    float terminal_cost = 0.0;
    if (guid_type == 2) {
        terminal_cost = terminal_cost_2_(
            P,
            vel,
            dist_to_path,
            desired_speed,
            cost_path_error_scale_m
        );
    } else {
        float total_time = dt * N;
        float min_move_range = cost_min_v_aligned * N * dt;
        terminal_cost = terminal_cost_1_(
            P[0],
            arr_waypoints_ned,
            n_wp,
            init_wp_idx_passed,
            final_wp_idx_passed,
            init_closest,
            final_closest,
            min_move_range,
            total_time
        );
    }

    arr_stk[idx] = total_cost + terminal_cost;
}
