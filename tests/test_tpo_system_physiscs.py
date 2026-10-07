from pathlib import Path
import sys
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from trissolaris.tpo_system_physics import SystemPhysics


def test_action_reaction_symmetry():
    """Valida a conservação do momento linear total (Força resultante interna nula)."""
    m_stars = np.array([1.0, 2.0, 0.5])
    q_s = np.array([[0.0, 0.0, 0.0], [2.0, 1.0, 0.0], [-1.0, 3.0, 0.0]])

    physics = SystemPhysics(m_stars, eps=1e-5)
    a_s = physics.compute_star_accelerations(q_s)

    total_force = np.sum(m_stars[:, np.newaxis] * a_s, axis=0)

    np.testing.assert_allclose(total_force, np.zeros(3), atol=1e-12)


def test_vectorized_planet_accelerations():
    """Garante que a aceleração vetorizada para K planetas produz os mesmos valores que a forma individual."""
    m_stars = np.array([1.0, 1.0, 1.0])
    q_s = np.array([[1.0, 0.0, 0.0], [-0.5, 0.866, 0.0], [-0.5, -0.866, 0.0]])

    q_p = np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])

    physics = SystemPhysics(m_stars, eps=1e-4)
    a_p_all = physics.compute_planet_accelerations(q_p, q_s)

    a_p_single = physics.compute_planet_accelerations(q_p[0:1], q_s)

    assert a_p_all.shape == (2, 3)
    np.testing.assert_allclose(a_p_all[0], a_p_single[0])


def test_specific_energy_distant_planet():
    """Um planeta infinitamente distante deve ter energia potencial ~ 0 e energia específica ~ T."""
    m_stars = np.array([1.0, 1.0, 1.0])
    q_s = np.array([[1.0, 0.0, 0.0], [-0.5, 0.866, 0.0], [-0.5, -0.866, 0.0]])

    q_p = np.array([[10000.0, 0.0, 0.0]])
    v_p = np.array([[2.0, 0.0, 0.0]])

    physics = SystemPhysics(m_stars)
    e_spec = physics.specific_planet_energy(q_p, v_p, q_s)
    expected_e = 0.5 * (2.0**2)

    np.testing.assert_allclose(e_spec[0], expected_e, rtol=1e-3)