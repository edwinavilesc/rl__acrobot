"""Double DQN: la misma red y el mismo bucle, con un objetivo de Bellman distinto.

En DQN el objetivo usa `max Q_target(s')`: la red objetivo elige la mejor accion y
ademas estima cuanto vale. Cuando las estimaciones tienen ruido, tomar el maximo
devuelve de forma sistematica un valor demasiado alto, porque el maximo tiende a
caer sobre el error positivo mas grande. Es el sesgo de sobreestimacion que
describen van Hasselt, Guez y Silver (2016).

Double DQN separa las dos funciones. La red en linea elige la accion; la red
objetivo la evalua:

    DQN         y = r + gamma * max_a  Q_target(s', a)
    Double DQN  y = r + gamma * Q_target(s', argmax_a Q_online(s', a))

Todo lo demas -arquitectura, hiperparametros, buffer, exploracion, orden de las
llamadas al generador aleatorio- es identico al de `dqn_acrobot_v3`, de modo que
la diferencia entre las dos corridas solo puede venir del objetivo.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .dqn_acrobot_v3 import (
    DEVICE,
    Config,
    QNetwork,
    ReplayBuffer,
    choose_action,
    epsilon_by_step,
    seed_everything,
)


def optimize_step_double(q_net, target_net, optimizer, buffer, cfg):
    """Un paso de descenso con el objetivo de Double DQN.

    Identico a `dqn_acrobot_v3.optimize_step` salvo en el bloque `no_grad`, donde
    la accion del siguiente estado la elige la red en linea y la evalua la red
    objetivo.
    """
    if len(buffer) < max(cfg.batch_size, cfg.learning_starts):
        return None

    s, a, r, s2, terminated = buffer.sample(cfg.batch_size)
    s = torch.as_tensor(s, dtype=torch.float32, device=DEVICE)
    a = torch.as_tensor(a, dtype=torch.long, device=DEVICE).unsqueeze(1)
    r = torch.as_tensor(r, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    s2 = torch.as_tensor(s2, dtype=torch.float32, device=DEVICE)
    terminated = torch.as_tensor(terminated, dtype=torch.float32, device=DEVICE).unsqueeze(1)

    current_q = q_net(s).gather(1, a)
    with torch.no_grad():
        mejor_accion = q_net(s2).argmax(dim=1, keepdim=True)        # la elige la red en linea
        next_q = target_net(s2).gather(1, mejor_accion)             # la evalua la red objetivo
        target_q = r + cfg.gamma * next_q * (1 - terminated)

    loss = nn.functional.smooth_l1_loss(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    grad_norm = float(nn.utils.clip_grad_norm_(q_net.parameters(), cfg.max_grad_norm))
    optimizer.step()
    return float(loss.item()), grad_norm


def train_double_dqn(cfg: Config, artifact_dir: Path = Path("artifacts")):
    """Entrena con el objetivo de Double DQN y devuelve la red y la historia."""
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
            buf.push(obs, a, r, nxt, terminated)

            out = optimize_step_double(q, tgt, opt, buf, cfg)
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

    torch.save(q.state_dict(), artifact_dir / f"double_dqn_{cfg.episodes}ep_state_dict.pt")
    (artifact_dir / f"double_dqn_{cfg.episodes}ep.json").write_text(
        json.dumps(h, indent=2), encoding="utf-8")
    return q, h
