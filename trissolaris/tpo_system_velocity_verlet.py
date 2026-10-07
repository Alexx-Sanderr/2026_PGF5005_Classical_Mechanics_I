import numpy as np
from trissolaris.tpo_system_events import evaluate_events
from trissolaris.tpo_system_physics import SystemPhysics


class VelocityVerletIntegrator:
    """Integrador numérico simplético Velocity Verlet com suporte a deteção de eventos e avaliação de conservação de energia."""

    def __init__(self, m_stars: np.ndarray, eps: float = 1e-4):
        self.physics = SystemPhysics(m_stars, eps=eps)
        self.m_stars = self.physics.m_stars
        self.eps = self.physics.eps

    def step(
        self,
        q_s: np.ndarray,
        v_s: np.ndarray,
        a_s: np.ndarray,
        q_p: np.ndarray,
        v_p: np.ndarray,
        a_p: np.ndarray,
        dt: float,
    ):
        """Avança o estado do sistema por um passo de tempo dt utilizando o esquema Velocity Verlet."""
        v_s_half = v_s + 0.5 * dt * a_s
        v_p_half = v_p + 0.5 * dt * a_p if len(q_p) > 0 else v_p

        q_s_new = q_s + dt * v_s_half
        q_p_new = q_p + dt * v_p_half if len(q_p) > 0 else q_p

        a_s_new = self.physics.compute_star_accelerations(q_s_new)
        a_p_new = (
            self.physics.compute_planet_accelerations(q_p_new, q_s_new)
            if len(q_p) > 0
            else a_p
        )

        v_s_new = v_s_half + 0.5 * dt * a_s_new
        v_p_new = v_p_half + 0.5 * dt * a_p_new if len(q_p) > 0 else v_p_half

        return q_s_new, v_s_new, a_s_new, q_p_new, v_p_new, a_p_new

    def simulate_orbits(
        self,
        q_s0: np.ndarray,
        v_s0: np.ndarray,
        q_p0: np.ndarray,
        v_p0: np.ndarray,
        dt: float = 0.001,
        T_max: float = 100.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simula a evolução orbital contínua das estrelas e planetas sem filtragem de eventos."""
        q_s = np.array(q_s0, dtype=np.float64)
        v_s = np.array(v_s0, dtype=np.float64)
        q_p = np.array(q_p0, dtype=np.float64)
        v_p = np.array(v_p0, dtype=np.float64)

        if q_p.ndim == 1:
            q_p = q_p[np.newaxis, :]
            v_p = v_p[np.newaxis, :]

        a_s = self.physics.compute_star_accelerations(q_s)
        a_p = self.physics.compute_planet_accelerations(q_p, q_s)

        n_steps = int(np.ceil(T_max / dt))
        q_s_list = []
        q_p_list = []

        for _ in range(n_steps):
            q_s_list.append(q_s.copy())
            q_p_list.append(q_p.copy())
            q_s, v_s, a_s, q_p, v_p, a_p = self.step(
                q_s, v_s, a_s, q_p, v_p, a_p, dt
            )

        return np.array(q_s_list), np.array(q_p_list)

    def simulate_with_events(
        self,
        q_s0: np.ndarray,
        v_s0: np.ndarray,
        q_p0: np.ndarray,
        v_p0: np.ndarray,
        R_stars: np.ndarray,
        T_max: float = 300.0,
        dt: float = 0.001,
        R_corte: float = 50.0,
        save_history: bool = True,
    ) -> dict:
        """Executa a simulação completa a passo fixo dt com monitoramento e remoção de partículas por evento."""
        q_s = np.array(q_s0, dtype=np.float64)
        v_s = np.array(v_s0, dtype=np.float64)
        q_p = np.array(q_p0, dtype=np.float64)
        v_p = np.array(v_p0, dtype=np.float64)

        if q_p.ndim == 1:
            q_p = q_p[np.newaxis, :]
            v_p = v_p[np.newaxis, :]

        K = len(q_p)
        active_mask = np.ones(K, dtype=bool)
        t_sob = np.full(K, T_max, dtype=np.float64)
        events = ["Sobreviveu"] * K

        t_list, q_s_list, q_p_list = [], [], []

        a_s = self.physics.compute_star_accelerations(q_s)
        a_p = self.physics.compute_planet_accelerations(q_p, q_s)

        E0_stars = self.physics.total_star_energy(q_s, v_s)
        max_energy_error = 0.0

        n_steps = int(np.ceil(T_max / dt))
        t_curr = 0.0

        for _ in range(n_steps):
            if t_curr >= T_max:
                break

            if save_history:
                t_list.append(t_curr)
                q_s_list.append(q_s.copy())
                q_p_step = np.full((K, 3), np.nan)
                q_p_step[active_mask] = q_p[active_mask]
                q_p_list.append(q_p_step)

            if np.any(active_mask):
                has_event, event_types = evaluate_events(
                    q_p[active_mask],
                    v_p[active_mask],
                    q_s,
                    v_s,
                    self.m_stars,
                    R_stars,
                    R_corte=R_corte,
                )

                if np.any(has_event):
                    active_indices = np.where(active_mask)[0]
                    for idx_local, event_occurred in enumerate(has_event):
                        if event_occurred:
                            idx_global = active_indices[idx_local]
                            active_mask[idx_global] = False
                            t_sob[idx_global] = t_curr
                            events[idx_global] = event_types[idx_local]

            q_p_active = q_p[active_mask]
            v_p_active = v_p[active_mask]
            a_p_active = a_p[active_mask] if len(a_p) > 0 else a_p

            q_s, v_s, a_s, q_p_next, v_p_next, a_p_next = self.step(
                q_s, v_s, a_s, q_p_active, v_p_active, a_p_active, dt
            )

            if np.any(active_mask):
                q_p[active_mask] = q_p_next
                v_p[active_mask] = v_p_next
                a_p[active_mask] = a_p_next

            E_curr = self.physics.total_star_energy(q_s, v_s)
            energy_err = np.abs((E_curr - E0_stars) / E0_stars)
            if energy_err > max_energy_error:
                max_energy_error = energy_err

            t_curr += dt

        return {
            "q_s_hist": np.array(q_s_list) if save_history else None,
            "q_p_hist": np.array(q_p_list) if save_history else None,
            "time_array": np.array(t_list),
            "t_sob": t_sob,
            "events": events,
            "max_energy_error": max_energy_error,
        }