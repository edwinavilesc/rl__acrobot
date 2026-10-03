"""Ablacion: que ocurre si la truncacion se trata como un final real.

Este modulo repite el entrenamiento de `dqn_acrobot_v3.train_dqn` cambiando una
sola cosa: lo que se guarda en el buffer como senal de terminacion.

    correcto:  buf.push(obs, a, r, nxt, terminated)
    ablacion:  buf.push(obs, a, r, nxt, terminated or truncated)

Todo lo demas -arquitectura, hiperparametros, semilla, orden de las llamadas al
generador aleatorio- es identico, de modo que la diferencia entre las dos curvas
solo puede venir de ese cambio.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim

from .dqn_acrobot_v3 import (
    DEVICE,
    Config,
    QNetwork,
    ReplayBuffer,
    choose_action,
    epsilon_by_step,
    optimize_step,
    seed_everything,
)


def train_dqn_sin_distinguir_truncacion(cfg: Config, artifact_dir: Path = Path("artifacts")):
    """Entrena tratando `truncated` como si fuera un estado terminal real."""
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed)
    env = gym.make(cfg.env_id)
    env.action_space.seed(cfg.seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = env.action_space.n

    q = QNetwork(obs_dim, n_actions, cfg.hidden).to(DEVICE)
    tgt = QNetwork(obs_dim, n_actions, cfg.hidden).to(DEVICE)
    tgt.load_state_dict(q.state_dict())
    tgt.eval()
    opt = optim.Adam(q.parameters(), lr=cfg.lr)
    buf = ReplayBuffer(cfg.buffer_size)

    h = {"returns": [], "lengths": [], "losses": [], "grad_norms": [],
         "epsilons": [], "terminated": [], "truncated": []}
    step = 0
    start = time.time()

    for ep in range(cfg.episodes):
        obs, _ = env.reset(seed=cfg.seed + ep)
        total = 0.0
        nstep = 0
        while True:
            eps = epsilon_by_step(step, cfg)
            a = choose_action(obs, q, n_actions, eps)
            nxt, r, terminated, truncated, _ = env.step(a)

            buf.push(obs, a, r, nxt, terminated or truncated)   # ← la unica diferencia

            out = optimize_step(q, tgt, opt, buf, cfg)
            if out is not None:
                loss, gn = out
                h["losses"].append(loss)
                h["grad_norms"].append(gn)

            obs = nxt
            total += r
            nstep += 1
            step += 1
            if step % cfg.target_update_steps == 0:
                tgt.load_state_dict(q.state_dict())
            if terminated or truncated:
                break

        h["returns"].append(total)
        h["lengths"].append(nstep)
        h["epsilons"].append(eps)
        h["terminated"].append(bool(terminated))
        h["truncated"].append(bool(truncated))
        if (ep + 1) % 100 == 0:
            print(f"ep={ep+1:4d} mean100={np.mean(h['returns'][-100:]):8.2f} eps={eps:.3f}",
                  flush=True)
    h["seconds"] = time.time() - start
    h["config"] = asdict(cfg)
    env.close()
    torch.save(q.state_dict(), artifact_dir / f"ablacion_truncacion_{cfg.episodes}ep_state_dict.pt")
    (artifact_dir / f"ablacion_truncacion_{cfg.episodes}ep.json").write_text(
        json.dumps(h, indent=2), encoding="utf-8")
    return q, h
