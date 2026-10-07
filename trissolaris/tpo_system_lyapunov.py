import numpy as np
from trissolaris.tpo_system_events import evaluate_events
from trissolaris.tpo_system_physics import (
    compute_planet_accelerations,
    compute_star_accelerations,
)


def compute_lyapunov_exponents(
    integrator,
    q_s0: np.ndarray,
    v_s0: np.ndarray,
    q_p0: np.ndarray,
    v_p0: np.ndarray,
    R_stars: np.ndarray,
    dt: float = 0.001,
    T_max: float = 100.0,
    d0: float = 1e-8,
    renorm_steps: int = 10,
    R_corte: float = 100.0,
) -> dict:
    """Calcula o Maior Expoente de Lyapunov (lambda_max) utilizando o método do vetor desviado de Benettin."""
    q_s = np.array(q_s0, dtype=np.float64)
    v_s = np.array(v_s0, dtype=np.float64)
    q_p = np.array(q_p0, dtype=np.float64)
    v_p = np.array(v_p0, dtype=np.float64)

    if q_p.ndim == 1:
        q_p = q_p[np.newaxis, :]
        v_p = v_p[np.newaxis, :]

    K = len(q_p)
    n_steps = int(np.ceil(T_max / dt))

    np.random.seed(42)
    random_dirs = np.random.normal(size=(K, 6))
    norms = np.linalg.norm(random_dirs, axis=1, keepdims=True)
    unit_dirs = random_dirs / norms

    q_p_pert = q_p + unit_dirs[:, :3] * d0
    v_p_pert = v_p + unit_dirs[:, 3:] * d0

    active_mask = np.ones(K, dtype=bool)
    t_sob = np.full(K, T_max, dtype=np.float64)
    events = ["Sobreviveu"] * K
    log_divergence_sum = np.zeros(K, dtype=np.float64)

    a_s = compute_star_accelerations(q_s, integrator.m_stars, integrator.eps)
    a_p = compute_planet_accelerations(
        q_p, q_s, integrator.m_stars, integrator.eps
    )
    a_p_pert = compute_planet_accelerations(
        q_p_pert, q_s, integrator.m_stars, integrator.eps
    )

    for step in range(n_steps):
        t_curr = step * dt

        if np.any(active_mask):
            has_event, event_types = evaluate_events(
                q_p[active_mask],
                v_p[active_mask],
                q_s,
                v_s,
                integrator.m_stars,
                R_stars,
                R_corte=R_corte,
            )

            if np.any(has_event):
                active_indices = np.where(active_mask)[0]
                for local_i, occurred in enumerate(has_event):
                    if occurred:
                        global_idx = active_indices[local_i]
                        active_mask[global_idx] = False
                        t_sob[global_idx] = t_curr
                        events[global_idx] = event_types[local_i]

        if not np.any(active_mask):
            break

        q_s_new, v_s_new, a_s_new, q_p_act_new, v_p_act_new, a_p_act_new = (
            integrator.step(
                q_s,
                v_s,
                a_s,
                q_p[active_mask],
                v_p[active_mask],
                a_p[active_mask],
                dt,
            )
        )

        _, _, _, q_p_pert_act_new, v_p_pert_act_new, a_p_pert_act_new = (
            integrator.step(
                q_s,
                v_s,
                a_s,
                q_p_pert[active_mask],
                v_p_pert[active_mask],
                a_p_pert[active_mask],
                dt,
            )
        )

        q_s, v_s, a_s = q_s_new, v_s_new, a_s_new
        q_p[active_mask] = q_p_act_new
        v_p[active_mask] = v_p_act_new
        a_p[active_mask] = a_p_act_new

        q_p_pert[active_mask] = q_p_pert_act_new
        v_p_pert[active_mask] = v_p_pert_act_new
        a_p_pert[active_mask] = a_p_pert_act_new

        if (step + 1) % renorm_steps == 0:
            active_indices = np.where(active_mask)[0]

            dq = q_p_pert[active_mask] - q_p[active_mask]
            dv = v_p_pert[active_mask] - v_p[active_mask]
            dw = np.hstack([dq, dv])

            d_t = np.linalg.norm(dw, axis=1)
            valid = d_t > 0.0

            if np.any(valid):
                valid_globals = active_indices[valid]
                log_divergence_sum[valid_globals] += np.log(d_t[valid] / d0)

                rescaled_dw = (dw[valid] / d_t[valid][:, np.newaxis]) * d0
                q_p_pert[valid_globals] = (
                    q_p[valid_globals] + rescaled_dw[:, :3]
                )
                v_p_pert[valid_globals] = (
                    v_p[valid_globals] + rescaled_dw[:, 3:]
                )

                a_p_pert[active_mask] = compute_planet_accelerations(
                    q_p_pert[active_mask],
                    q_s,
                    integrator.m_stars,
                    integrator.eps,
                )

    lyapunov_exponents = np.zeros(K, dtype=np.float64)
    for k in range(K):
        if t_sob[k] > 0:
            lyapunov_exponents[k] = log_divergence_sum[k] / t_sob[k]
        else:
            lyapunov_exponents[k] = np.nan

    return {
        "lyapunov_exponents": lyapunov_exponents,
        "t_sob": t_sob,
        "events": events,
    }