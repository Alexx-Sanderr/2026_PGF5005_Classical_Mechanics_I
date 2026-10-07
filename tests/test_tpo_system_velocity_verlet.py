from pathlib import Path
import sys
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from trissolaris.tpo_system_velocity_verlet import VelocityVerletIntegrator


def test_integration_shape_and_execution():
    """Testa a dimensão das matrizes de histórico retornadas pela integração."""
    m_stars = np.array([1.0, 1.0, 1.0])
    q_s0 = np.array(
        [[1.0, 0.0, 0.0], [-0.5, 0.866, 0.0], [-0.5, -0.866, 0.0]]
    )
    v_s0 = np.array(
        [[0.0, 0.5, 0.0], [-0.433, -0.25, 0.0], [0.433, -0.25, 0.0]]
    )
    q_p0 = np.array([[2.5, 0.0, 0.0]])
    v_p0 = np.array([[0.0, 0.7, 0.0]])

    integrator = VelocityVerletIntegrator(m_stars=m_stars, eps=1e-5)

    dt = 0.01
    T_max = 1.0
    q_s_hist, q_p_hist = integrator.simulate_orbits(
        q_s0, v_s0, q_p0, v_p0, dt=dt, T_max=T_max
    )

    expected_steps = int(np.ceil(T_max / dt))
    assert q_s_hist.shape == (expected_steps, 3, 3)
    assert q_p_hist.shape == (expected_steps, 1, 3)
    assert not np.isnan(q_s_hist).any()
    assert not np.isnan(q_p_hist).any()


def test_multi_planet_vectorized_integration():
    """Garante que a integração vetorizada lida corretamente com K partículas simultâneas."""
    m_stars = np.array([1.0, 1.0, 1.0])
    q_s0 = np.array(
        [[1.0, 0.0, 0.0], [-0.5, 0.866, 0.0], [-0.5, -0.866, 0.0]]
    )
    v_s0 = np.array(
        [[0.0, 0.5, 0.0], [-0.433, -0.25, 0.0], [0.433, -0.25, 0.0]]
    )

    K = 25
    np.random.seed(42)

    q_p0 = np.random.uniform(-2, 2, (K, 3))
    v_p0 = np.random.uniform(-1, 1, (K, 3))

    integrator = VelocityVerletIntegrator(m_stars=m_stars, eps=1e-4)
    q_s_hist, q_p_hist = integrator.simulate_orbits(
        q_s0, v_s0, q_p0, v_p0, dt=0.01, T_max=0.5
    )

    assert q_s_hist.shape == (50, 3, 3)
    assert q_p_hist.shape == (50, K, 3)


def test_simulate_with_events_removal():
    """Testa se uma partícula que colide é removida e tem seu t_sob registrado corretamente."""
    m_stars = np.array([1.0, 1.0, 1.0])
    R_stars = np.array([0.5, 0.5, 0.5])

    q_s0 = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [-5.0, 0.0, 0.0]])
    v_s0 = np.array([[0.0, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, -0.2, 0.0]])

    q_p0 = np.array([[0.2, 0.0, 0.0], [2.5, 0.0, 0.0]])
    v_p0 = np.array([[0.0, 0.0, 0.0], [0.0, 0.6, 0.0]])

    integrator = VelocityVerletIntegrator(m_stars=m_stars, eps=1e-4)

    results = integrator.simulate_with_events(
        q_s0,
        v_s0,
        q_p0,
        v_p0,
        R_stars=R_stars,
        dt=0.01,
        T_max=1.0,
        R_corte=100.0,
    )

    t_sob = results["t_sob"]
    events = results["events"]
    q_p_hist = results["q_p_hist"]

    assert t_sob[0] == 0.0
    assert "Colisão Estrela 0" in events[0]
    assert np.isnan(q_p_hist[1, 0, :]).all()

    assert t_sob[1] == 1.0
    assert events[1] == "Sobreviveu"