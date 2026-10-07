from pathlib import Path
import sys
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from trissolaris.tpo_system_events import (
    check_collisions,
    check_ejections,
    evaluate_events,
)


def test_collision_detection_multi_particle():
    """Testa se detecta colisão de partículas próximas ao raio das estrelas."""
    q_s = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    R_stars = np.array([0.1, 0.1, 0.1])

    q_p = np.array([[0.05, 0.0, 0.0], [5.0, 5.0, 0.0], [10.0, 0.09, 0.0]])

    collided, reasons = check_collisions(q_p, q_s, R_stars)

    assert collided[0] == True
    assert reasons[0] == "Colisão Estrela 0"

    assert collided[1] == False
    assert reasons[1] is None

    assert collided[2] == True
    assert reasons[2] == "Colisão Estrela 1"


def test_ejection_detection():
    """Testa se detecta partículas que ultrapassam R_corte com energia não-negativa."""
    m_s = np.array([1.0, 1.0, 1.0])
    q_s = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    v_s = np.zeros_like(q_s)

    q_p = np.array([[150.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    v_p = np.array([[2.0, 0.0, 0.0], [0.1, 0.0, 0.0]])

    ejected, reasons = check_ejections(q_p, v_p, q_s, v_s, m_s, R_corte=100.0)

    assert ejected[0] == True
    assert reasons[0] == "Ejeção"

    assert ejected[1] == False
    assert reasons[1] is None


def test_evaluate_events_combined():
    """Testa a avaliação integrada de eventos (colisão + ejeção simultâneos)."""
    m_s = np.array([1.0, 1.0, 1.0])
    q_s = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [-5.0, 0.0, 0.0]])
    R_stars = np.array([0.2, 0.2, 0.2])
    v_s = np.zeros_like(q_s)

    q_p = np.array(
        [[0.1, 0.0, 0.0], [200.0, 0.0, 0.0], [2.0, 2.0, 0.0]]
    )
    v_p = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.1, 0.1, 0.0]])

    has_event, event_types = evaluate_events(
        q_p, v_p, q_s, v_s, m_s, R_stars, R_corte=100.0
    )

    assert np.array_equal(has_event, [True, True, False])
    assert event_types[0] == "Colisão Estrela 0"
    assert event_types[1] == "Ejeção"
    assert event_types[2] is None