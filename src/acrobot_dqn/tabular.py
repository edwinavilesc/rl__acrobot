"""Q-learning tabular sobre Acrobot-v1, para saber si hacia falta la red.

El agente del entregable usa una red porque la funcion Q depende de productos entre
angulos y velocidades. Eso es un argumento; este modulo lo convierte en una medicion.

La observacion que entrega Gymnasium tiene seis componentes, pero el estado fisico
real son cuatro: dos angulos y dos velocidades. Los cosenos y senos son una
reparametrizacion que le conviene a la red y no a una tabla, asi que aqui se
recuperan los angulos con `atan2` y se discretizan las cuatro variables. Con eso la
tabla tiene `bins_angulo^2 * bins_velocidad^2 * 3` entradas.

El presupuesto es el mismo que el del DQN -1.200 episodios, gamma = 0,99, el mismo
decaimiento de epsilon- y la evaluacion usa las mismas semillas disjuntas, de modo
que la comparacion sea entre metodos y no entre presupuestos.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np

# Limites de las velocidades angulares que declara el observation_space.
OMEGA1_MAX = 4 * np.pi
OMEGA2_MAX = 9 * np.pi


@dataclass
class ConfigTabular:
    """Hiperparametros del agente tabular."""

    env_id: str = "Acrobot-v1"
    episodes: int = 1200
    gamma: float = 0.99
    alpha: float = 0.1          # el valor elegido por el barrido de §10.4
    bins_angulo: int = 12
    bins_velocidad: int = 10
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 60_000
    seed: int = 42
    eval_episodes: int = 100


class Discretizador:
    """Convierte la observacion de seis componentes en un indice de la tabla."""

    def __init__(self, cfg: ConfigTabular):
        self.na = cfg.bins_angulo
        self.nv = cfg.bins_velocidad
        # bordes internos: np.digitize devuelve 0..n-1 con n-1 bordes
        self.bordes_angulo = np.linspace(-np.pi, np.pi, self.na + 1)[1:-1]
        self.bordes_omega1 = np.linspace(-OMEGA1_MAX, OMEGA1_MAX, self.nv + 1)[1:-1]
        self.bordes_omega2 = np.linspace(-OMEGA2_MAX, OMEGA2_MAX, self.nv + 1)[1:-1]

    @property
    def n_estados(self) -> int:
        return self.na * self.na * self.nv * self.nv

    def __call__(self, obs) -> int:
        cos1, sen1, cos2, sen2, omega1, omega2 = obs
        t1 = np.arctan2(sen1, cos1)
        t2 = np.arctan2(sen2, cos2)
        i = int(np.digitize(t1, self.bordes_angulo))
        j = int(np.digitize(t2, self.bordes_angulo))
        k = int(np.digitize(omega1, self.bordes_omega1))
        m = int(np.digitize(omega2, self.bordes_omega2))
        return ((i * self.na + j) * self.nv + k) * self.nv + m


def epsilon_por_paso(paso: int, cfg: ConfigTabular) -> float:
    """Mismo decaimiento lineal por paso de entorno que usa el DQN."""
    frac = min(1.0, paso / cfg.epsilon_decay_steps)
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)


def entrenar(cfg: ConfigTabular, verbose: bool = True):
    """Entrena Q-learning tabular y devuelve la tabla y la historia."""
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    env = gym.make(cfg.env_id)
    env.action_space.seed(cfg.seed)

    disc = Discretizador(cfg)
    tabla_q = np.zeros((disc.n_estados, 3), dtype=np.float32)
    h = {"returns": [], "lengths": [], "terminated": [], "truncated": [], "epsilons": []}
    paso = 0
    inicio = time.time()

    for ep in range(cfg.episodes):
        obs, _ = env.reset(seed=cfg.seed + ep)
        s = disc(obs)
        total, n = 0.0, 0
        while True:
            eps = epsilon_por_paso(paso, cfg)
            a = random.randrange(3) if random.random() < eps else int(tabla_q[s].argmax())
            obs, r, terminado, truncado, _ = env.step(a)
            s2 = disc(obs)
            objetivo = r + cfg.gamma * float(tabla_q[s2].max()) * (1.0 - float(terminado))
            tabla_q[s, a] += cfg.alpha * (objetivo - tabla_q[s, a])
            s = s2
            total += r
            n += 1
            paso += 1
            if terminado or truncado:
                break
        h["returns"].append(total)
        h["lengths"].append(n)
        h["terminated"].append(bool(terminado))
        h["truncated"].append(bool(truncado))
        h["epsilons"].append(eps)
        if verbose and (ep + 1) % 200 == 0:
            print(f"ep={ep+1:4d} mean100={np.mean(h['returns'][-100:]):8.2f} eps={eps:.3f}",
                  flush=True)

    h["seconds"] = time.time() - inicio
    h["config"] = asdict(cfg)
    # celdas cuyo valor se actualizo alguna vez. No es exactamente "visitadas": un
    # estado visto solo como s2 del ultimo paso de un episodio nunca se actualiza.
    h["estados_visitados"] = int((tabla_q != 0).any(axis=1).sum())
    h["estados_totales"] = disc.n_estados
    env.close()
    return tabla_q, h


def evaluar(tabla_q: np.ndarray, cfg: ConfigTabular, n: int | None = None,
            seed_start: int = 10_000) -> dict:
    """Evalua la politica greedy con las mismas semillas que usa el DQN."""
    n = n or cfg.eval_episodes
    disc = Discretizador(cfg)
    env = gym.make(cfg.env_id)
    retornos, largos, terminados = [], [], []
    for semilla in range(seed_start, seed_start + n):
        obs, _ = env.reset(seed=semilla)
        total, pasos = 0.0, 0
        while True:
            obs, r, terminado, truncado, _ = env.step(int(tabla_q[disc(obs)].argmax()))
            total += r
            pasos += 1
            if terminado or truncado:
                retornos.append(total)
                largos.append(pasos)
                terminados.append(bool(terminado))
                break
    env.close()
    return {
        "n": len(retornos),
        "mean_return": float(np.mean(retornos)),
        "median_return": float(np.median(retornos)),
        "sd_return": float(np.std(retornos, ddof=1)),
        "mean_steps": float(np.mean(largos)),
        "terminal_rate": float(np.mean(terminados)),
        "returns": retornos,
    }


def guardar(tabla_q: np.ndarray, h: dict, ev: dict, carpeta: Path, etiqueta: str) -> None:
    """Escribe la tabla, la historia y la evaluacion de una corrida."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    np.save(carpeta / f"tabla_{etiqueta}.npy", tabla_q)
    (carpeta / f"tabular_{etiqueta}.json").write_text(
        json.dumps(h, indent=2), encoding="utf-8")
    (carpeta / f"evaluacion_{etiqueta}.json").write_text(
        json.dumps(ev, indent=2), encoding="utf-8")
