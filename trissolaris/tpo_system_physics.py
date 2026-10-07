import numpy as np


class SystemPhysics:
    """Classe responsável pelo cálculo de acelerações, transformações de coordenadas e diagnósticos de energia do sistema N-corpos."""

    def __init__(self, m_stars: np.ndarray, eps: float = 1e-4):
        self.m_stars = np.array(m_stars, dtype=np.float64)
        self.eps = float(eps)

    def normalize_center_of_mass(
        self, q_s: np.ndarray, v_s: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Translada e transforma o sistema estelar para o referencial do Centro de Massa em repouso."""
        M_tot = np.sum(self.m_stars)
        q_cm = np.sum(self.m_stars[:, np.newaxis] * q_s, axis=0) / M_tot
        v_cm = np.sum(self.m_stars[:, np.newaxis] * v_s, axis=0) / M_tot

        q_s_norm = q_s - q_cm
        v_s_norm = v_s - v_cm

        return q_s_norm, v_s_norm

    def compute_star_accelerations(self, q_s: np.ndarray) -> np.ndarray:
        """Calcula as acelerações mútuas entre as estrelas utilizando o parâmetro de suavização (eps)."""
        N = len(self.m_stars)
        a_s = np.zeros_like(q_s)
        eps2 = self.eps**2

        for i in range(N):
            for j in range(N):
                if i != j:
                    r_ij = q_s[i] - q_s[j]
                    dist3 = (np.dot(r_ij, r_ij) + eps2) ** 1.5
                    a_s[i] -= self.m_stars[j] * r_ij / dist3

        return a_s

    def compute_planet_accelerations(
        self, q_p: np.ndarray, q_s: np.ndarray
    ) -> np.ndarray:
        """Calcula a aceleração gravitacional sofrida pelos planetas devido às estrelas."""
        if len(q_p) == 0:
            return np.empty((0, 3), dtype=np.float64)

        eps2 = self.eps**2
        r_ki = q_p[:, np.newaxis, :] - q_s[np.newaxis, :, :]
        dist2 = np.sum(r_ki**2, axis=2) + eps2
        dist3 = (dist2**1.5)[:, :, np.newaxis]

        accel_contributions = (
            -(self.m_stars[np.newaxis, :, np.newaxis] * r_ki) / dist3
        )
        a_p = np.sum(accel_contributions, axis=1)

        return a_p

    def total_star_energy(self, q_s: np.ndarray, v_s: np.ndarray) -> float:
        """Calcula a energia Hamiltoniana total (Cinética + Potencial Suavizada) do sistema estelar."""
        T = 0.5 * np.sum(self.m_stars[:, np.newaxis] * (v_s**2))
        V = 0.0
        N = len(self.m_stars)
        eps2 = self.eps**2

        for i in range(N):
            for j in range(i + 1, N):
                r_ij2 = np.sum((q_s[i] - q_s[j]) ** 2)
                V -= (self.m_stars[i] * self.m_stars[j]) / np.sqrt(r_ij2 + eps2)

        return float(T + V)

    def specific_planet_energy(
        self, q_p: np.ndarray, v_p: np.ndarray, q_s: np.ndarray
    ) -> np.ndarray:
        """Calcula a energia orbital específica para cada um dos planetas."""
        if len(q_p) == 0:
            return np.array([], dtype=np.float64)

        T_spec = 0.5 * np.sum(v_p**2, axis=1)
        r_ki2 = np.sum((q_p[:, np.newaxis, :] - q_s[np.newaxis, :, :]) ** 2, axis=2)
        V_spec = -np.sum(
            self.m_stars[np.newaxis, :] / np.sqrt(r_ki2 + self.eps**2), axis=1
        )

        return T_spec + V_spec


def normalize_center_of_mass(
    m_s: np.ndarray, q_s: np.ndarray, v_s: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    phys = SystemPhysics(m_s)
    return phys.normalize_center_of_mass(q_s, v_s)


def compute_star_accelerations(
    q_s: np.ndarray, m_s: np.ndarray, eps: float = 1e-4
) -> np.ndarray:
    phys = SystemPhysics(m_s, eps=eps)
    return phys.compute_star_accelerations(q_s)


def compute_planet_accelerations(
    q_p: np.ndarray, q_s: np.ndarray, m_s: np.ndarray, eps: float = 1e-4
) -> np.ndarray:
    phys = SystemPhysics(m_s, eps=eps)
    return phys.compute_planet_accelerations(q_p, q_s)


def total_star_energy(
    q_s: np.ndarray, v_s: np.ndarray, m_s: np.ndarray, eps: float = 1e-4
) -> float:
    phys = SystemPhysics(m_s, eps=eps)
    return phys.total_star_energy(q_s, v_s)


def specific_planet_energy(
    q_p: np.ndarray,
    v_p: np.ndarray,
    q_s: np.ndarray,
    m_s: np.ndarray,
    eps: float = 1e-4,
) -> np.ndarray:
    phys = SystemPhysics(m_s, eps=eps)
    return phys.specific_planet_energy(q_p, v_p, q_s)