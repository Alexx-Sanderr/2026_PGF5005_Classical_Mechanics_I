from pathlib import Path
import sys
import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from trissolaris.config_loader import load_config


def test_load_config_explicit_mode(tmp_path):
    """Testa o carregamento de configurações em modo explícito com planetas nomeados."""
    yaml_content = """
    simulation:
      dt: 0.001
      T_max: 10.0
      eps: 0.0001
      R_corte: 100.0
    stars:
      masses: [1.0, 1.0, 1.0]
      radii: [0.01, 0.01, 0.01]
      positions: [[1,0,0], [-0.5, 0.86, 0], [-0.5, -0.86, 0]]
      velocities: [[0,0.5,0], [-0.4,-0.2,0], [0.4,-0.2,0]]
    planets:
      mode: "explicit"
      list:
        planeta_1:
          position: [2.0, 0.0, 0.0]
          velocity: [0.0, 0.8, 0.0]
        planeta_2:
          position: [3.0, 0.0, 0.0]
          velocity: [0.0, 0.6, 0.0]
    """

    config_file = tmp_path / "test_explicit.yaml"
    config_file.write_text(yaml_content, encoding="utf-8")

    cfg = load_config(str(config_file))

    assert cfg["stars"]["masses"].shape == (3,)
    assert cfg["stars"]["positions"].shape == (3, 3)

    parsed_p = cfg["planets_parsed"]
    assert parsed_p["count"] == 2
    assert parsed_p["positions"].shape == (2, 3)
    assert parsed_p["names"] == ["planeta_1", "planeta_2"]

    np.testing.assert_allclose(parsed_p["positions"][0], [2.0, 0.0, 0.0])


def test_load_config_sampling_grid_mode(tmp_path):
    """Testa o gerador de condições iniciais via amostragem em grade."""
    yaml_content = """
    simulation:
      dt: 0.001
      T_max: 10.0
      eps: 0.0001
      R_corte: 100.0
    stars:
      masses: [1.0, 1.0, 1.0]
      radii: [0.01, 0.01, 0.01]
      positions: [[1,0,0], [-0.5, 0.86, 0], [-0.5, -0.86, 0]]
      velocities: [[0,0.5,0], [-0.4,-0.2,0], [0.4,-0.2,0]]
    planets:
      mode: "sampling"
      sampling:
        type: "grid"
        N_r: 5
        N_v: 4
        r_range: [1.0, 5.0]
        v_range: [0.2, 1.0]
        plane_z: 0.0
        seed: 123
    """
    config_file = tmp_path / "test_sampling.yaml"
    config_file.write_text(yaml_content, encoding="utf-8")

    cfg = load_config(str(config_file))
    parsed_p = cfg["planets_parsed"]

    assert parsed_p["count"] == 20
    assert parsed_p["positions"].shape == (20, 3)
    assert parsed_p["velocities"].shape == (20, 3)


def test_load_config_sampling_uniform_mode(tmp_path):
    """Testa o gerador de condições iniciais via amostragem aleatória uniforme."""
    yaml_content = """
    simulation:
      dt: 0.001
      T_max: 10.0
      eps: 0.0001
      R_corte: 100.0
    stars:
      masses: [1.0, 1.0, 1.0]
      radii: [0.01, 0.01, 0.01]
      positions: [[1,0,0], [-0.5, 0.86, 0], [-0.5, -0.86, 0]]
      velocities: [[0,0.5,0], [-0.4,-0.2,0], [0.4,-0.2,0]]
    planets:
      mode: "sampling"
      sampling:
        type: "random_uniform"
        N_r: 5
        N_v: 4
        r_range: [1.0, 5.0]
        v_range: [0.2, 1.0]
        plane_z: 0.0
        seed: 42
    """
    config_file = tmp_path / "test_uniform.yaml"
    config_file.write_text(yaml_content, encoding="utf-8")

    cfg = load_config(str(config_file))
    parsed_p = cfg["planets_parsed"]

    assert parsed_p["count"] == 20
    assert parsed_p["positions"].shape == (20, 3)
    assert parsed_p["velocities"].shape == (20, 3)

    assert parsed_p["names"][0] == "p_rand_0"

    r_computed = np.linalg.norm(parsed_p["positions"][:, :2], axis=1)
    assert np.all(r_computed >= 1.0)
    assert np.all(r_computed <= 5.0)

    v_computed = np.linalg.norm(parsed_p["velocities"], axis=1)
    assert np.all(v_computed >= 0.2 - 1e-12)
    assert np.all(v_computed <= 1.0 + 1e-12)

    np.testing.assert_allclose(parsed_p["positions"][:, 2], 0.0)


def test_file_not_found():
    """Garante que exceções adequadas são lançadas caso o caminho seja inválido."""
    with pytest.raises(FileNotFoundError):
        load_config("caminho_inexistente/config.yaml")