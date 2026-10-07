from pathlib import Path
import sys
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from trissolaris.tpo_system_lyapunov import compute_lyapunov_exponents
from trissolaris.tpo_system_velocity_verlet import VelocityVerletIntegrator


def test_compute_lyapunov_exponents():
    m_stars = np.array([1.0, 1.0, 1.0])
    R_stars = np.array([0.1, 0.1, 0.1])

    q_s0 = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [-3.0, 0.0, 0.0]])
    v_s0 = np.array([[0.0, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, -0.3, 0.0]])

    q_p0 = np.array([[1.5, 0.0, 0.0]])
    v_p0 = np.array([[0.0, 0.8, 0.0]])

    integrator = VelocityVerletIntegrator(m_stars=m_stars, eps=1e-4)

    res = compute_lyapunov_exponents(
        integrator,
        q_s0,
        v_s0,
        q_p0,
        v_p0,
        R_stars=R_stars,
        dt=0.005,
        T_max=1.0,
        d0=1e-8,
        renorm_steps=5,
    )

    assert "lyapunov_exponents" in res
    assert len(res["lyapunov_exponents"]) == 1
    assert not np.isnan(res["lyapunov_exponents"][0])