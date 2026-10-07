import numpy as np
from trissolaris.tpo_system_physics import specific_planet_energy


def check_collisions(
    q_p: np.ndarray, q_s: np.ndarray, R_stars: np.ndarray
) -> tuple[np.ndarray, list[str | None]]:
    if q_p.ndim == 1:
        q_p = q_p[np.newaxis, :]

    K = len(q_p)
    r_ki = q_p[:, np.newaxis, :] - q_s[np.newaxis, :, :]
    dist_ki = np.linalg.norm(r_ki, axis=2)

    hit_matrix = dist_ki <= R_stars[np.newaxis, :]
    collided = np.any(hit_matrix, axis=1)
    reasons = [None] * K

    if np.any(collided):
        for k in range(K):
            if collided[k]:
                star_idx = int(np.argmin(dist_ki[k]))
                reasons[k] = f"Colisão Estrela {star_idx}"

    return collided, reasons


def check_ejections(
    q_p: np.ndarray,
    v_p: np.ndarray,
    q_s: np.ndarray,
    v_s: np.ndarray,
    m_s: np.ndarray,
    R_corte: float = 100.0,
) -> tuple[np.ndarray, list[str | None]]:
    if q_p.ndim == 1:
        q_p = q_p[np.newaxis, :]
        v_p = v_p[np.newaxis, :]

    K = len(q_p)

    M_tot = np.sum(m_s)
    q_cm = np.sum(m_s[:, np.newaxis] * q_s, axis=0) / M_tot
    v_cm = np.sum(m_s[:, np.newaxis] * v_s, axis=0) / M_tot

    r_rel = q_p - q_cm[np.newaxis, :]
    v_rel = v_p - v_cm[np.newaxis, :]

    dist_cm = np.linalg.norm(r_rel, axis=1)
    v_rad_dot = np.sum(r_rel * v_rel, axis=1)

    E_spec = specific_planet_energy(q_p, v_p, q_s, m_s)

    ejected = (dist_cm > R_corte) & (E_spec >= 0.0) & (v_rad_dot > 0.0)
    reasons = [None] * K

    for k in range(K):
        if ejected[k]:
            reasons[k] = "Ejeção"

    return ejected, reasons


def evaluate_events(
    q_p: np.ndarray,
    v_p: np.ndarray,
    q_s: np.ndarray,
    v_s: np.ndarray,
    m_s: np.ndarray,
    R_stars: np.ndarray,
    R_corte: float = 100.0,
):
    collided, reasons_col = check_collisions(q_p, q_s, R_stars)
    ejected, reasons_ej = check_ejections(q_p, v_p, q_s, v_s, m_s, R_corte)

    has_event = collided | ejected
    event_types = [None] * len(q_p)

    for k in range(len(q_p)):
        if collided[k]:
            event_types[k] = reasons_col[k]
        elif ejected[k]:
            event_types[k] = reasons_ej[k]

    return has_event, event_types