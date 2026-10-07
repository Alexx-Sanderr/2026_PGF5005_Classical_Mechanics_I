from pathlib import Path
import numpy as np
import yaml


def get_project_root() -> Path:
    """Retorna o caminho absoluto do diretório raiz do projeto."""
    return Path(__file__).resolve().parent.parent


def load_config(relative_filepath: str = "config.yaml") -> dict:
    """Carrega o arquivo YAML e converte as seções em matrizes NumPy normalizadas."""
    path = Path(relative_filepath)
    if not path.is_absolute():
        path = get_project_root() / path

    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo de configuração não encontrado em: {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    stars = cfg["stars"]
    stars["masses"] = np.array(stars["masses"], dtype=np.float64)
    stars["radii"] = np.array(stars["radii"], dtype=np.float64)
    stars["positions"] = np.array(stars["positions"], dtype=np.float64)
    stars["velocities"] = np.array(stars["velocities"], dtype=np.float64)

    planets_cfg = cfg["planets"]
    mode = planets_cfg.get("mode", "explicit")

    if mode == "explicit":
        positions = []
        velocities = []
        names = []

        planet_dict = planets_cfg.get("list", {})
        for name, data in planet_dict.items():
            names.append(name)
            positions.append(data["position"])
            velocities.append(data["velocity"])

        q_p = np.array(positions, dtype=np.float64)
        v_p = np.array(velocities, dtype=np.float64)

    elif mode == "sampling":
        sampling = planets_cfg["sampling"]
        sample_type = sampling.get("type", "grid")
        seed = sampling.get("seed", None)

        if seed is not None:
            np.random.seed(seed)

        r_min, r_max = sampling["r_range"]
        v_min, v_max = sampling["v_range"]
        N_r = sampling.get("N_r", 10)
        N_v = sampling.get("N_v", 10)
        z_val = sampling.get("plane_z", 0.0)

        if sample_type == "grid":
            r_vals = np.linspace(r_min, r_max, N_r)
            v_vals = np.linspace(v_min, v_max, N_v)
            R, V = np.meshgrid(r_vals, v_vals)

            r_flat = R.flatten()
            v_flat = V.flatten()
            K = len(r_flat)

            theta = np.random.uniform(0, 2 * np.pi, K)
            phi = np.random.uniform(0, 2 * np.pi, K)

            q_p = np.column_stack(
                [
                    r_flat * np.cos(theta),
                    r_flat * np.sin(theta),
                    np.full(K, z_val),
                ]
            )
            v_p = np.column_stack(
                [
                    v_flat * np.cos(phi),
                    v_flat * np.sin(phi),
                    np.zeros(K),
                ]
            )
            names = [f"p_grid_{i}" for i in range(K)]

        elif sample_type == "random_uniform":
            K = sampling.get("N", N_r * N_v)
            r_flat = np.random.uniform(r_min, r_max, K)
            v_flat = np.random.uniform(v_min, v_max, K)
            theta = np.random.uniform(0, 2 * np.pi, K)
            phi = np.random.uniform(0, 2 * np.pi, K)

            q_p = np.column_stack(
                [
                    r_flat * np.cos(theta),
                    r_flat * np.sin(theta),
                    np.full(K, z_val),
                ]
            )
            v_p = np.column_stack(
                [
                    v_flat * np.cos(phi),
                    v_flat * np.sin(phi),
                    np.zeros(K),
                ]
            )
            names = [f"p_rand_{i}" for i in range(K)]

        else:
            raise ValueError(
                f"Tipo de amostragem '{sample_type}' desconhecido."
            )

    else:
        raise ValueError(
            f"Modo de planeta '{mode}' inválido. Use 'explicit' ou 'sampling'"
        )

    cfg["planets_parsed"] = {
        "positions": q_p,
        "velocities": v_p,
        "names": names,
        "count": len(q_p),
    }

    return cfg