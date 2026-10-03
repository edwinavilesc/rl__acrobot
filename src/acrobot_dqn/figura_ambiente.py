"""Dibuja como se ve Acrobot-v1 y como el agente entrenado lo resuelve.

Renderiza un episodio de evaluacion con la politica congelada, toma seis
fotogramas y los acompana con la altura del extremo libre paso a paso. Las dos
mitades cuentan lo mismo: el brazo no sube de un tiron, sube acumulando energia
en oscilaciones cada vez mas amplias.
"""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend fijado arriba)

from .dqn_acrobot_v3 import DEVICE, QNetwork  # noqa: E402

SUP, TINTA, TINTA2 = "#fcfcfb", "#0b0b0b", "#52514e"
AZUL, GRIS = "#2a78d6", "#b9b7b0"
ALTURA_META = 1.0
TORQUE = {0: "−1", 1: "0", 2: "+1"}


def _altura(obs) -> float:
    """Altura del extremo libre: −cos θ₁ − cos(θ₁ + θ₂), con la observacion del entorno."""
    cos1, sen1, cos2, sen2 = obs[0], obs[1], obs[2], obs[3]
    return float(-cos1 - (cos1 * cos2 - sen1 * sen2))


def recorrer_episodio(red: QNetwork, semilla: int = 10_000):
    """Corre un episodio greedy y devuelve fotogramas, alturas y acciones."""
    env = gym.make("Acrobot-v1", render_mode="rgb_array")
    obs, _ = env.reset(seed=semilla)
    fotogramas = [np.array(env.render())]
    alturas = [_altura(obs)]
    acciones = []
    while True:
        with torch.no_grad():
            accion = int(red(torch.as_tensor(obs, dtype=torch.float32,
                                             device=DEVICE).unsqueeze(0)).argmax())
        obs, _, terminado, truncado, _ = env.step(accion)
        fotogramas.append(np.array(env.render()))
        alturas.append(_altura(obs))
        acciones.append(accion)
        if terminado or truncado:
            break
    env.close()
    return fotogramas, np.array(alturas), np.array(acciones)


def _recortar(fotogramas, margen: int = 10):
    """Recorta todos con el mismo encuadre cuadrado, centrado en el brazo."""
    pila = np.stack(fotogramas)
    # el brazo es lo que se mueve; la linea de meta esta fija y cruza todo el ancho
    movimiento = (pila.max(axis=0) != pila.min(axis=0)).any(axis=2)
    filas = np.where(movimiento.any(axis=1))[0]
    cols = np.where(movimiento.any(axis=0))[0]
    alto, ancho = pila.shape[1], pila.shape[2]
    lado = min(max(filas.max() - filas.min(), cols.max() - cols.min()) + 2 * margen,
               alto, ancho)
    cf = (filas.min() + filas.max()) // 2
    cc = (cols.min() + cols.max()) // 2
    f0 = int(np.clip(cf - lado // 2, 0, alto - lado))
    c0 = int(np.clip(cc - lado // 2, 0, ancho - lado))
    return [f[f0:f0 + lado, c0:c0 + lado] for f in fotogramas]


def _pasos_a_mostrar(alturas: np.ndarray, n: int = 6) -> list[int]:
    """Elige el arranque, el final y los picos del balanceo mejor repartidos."""
    picos = [i for i in range(1, len(alturas) - 1)
             if (alturas[i] - alturas[i - 1]) * (alturas[i + 1] - alturas[i]) < 0]
    intermedios = [p for p in picos if alturas[p] > alturas[p - 1]]   # solo los maximos
    if len(intermedios) > n - 2:
        idx = np.linspace(0, len(intermedios) - 1, n - 2).round().astype(int)
        intermedios = [intermedios[i] for i in idx]
    return [0, *intermedios, len(alturas) - 1]


def dibujar(checkpoint="resultados/dqn_acrobot_1200ep_state_dict.pt",
            salida="resultados/ambiente_acrobot.png", semilla: int = 10_000) -> Path:
    """Escribe la figura del entorno y devuelve su ruta."""
    red = QNetwork(6, 3).to(DEVICE)
    red.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    red.eval()

    fotogramas, alturas, acciones = recorrer_episodio(red, semilla)
    fotogramas = _recortar(fotogramas)
    pasos = _pasos_a_mostrar(alturas)

    fig = plt.figure(figsize=(12.6, 6.4), dpi=170)
    fig.patch.set_facecolor(SUP)
    rejilla = fig.add_gridspec(2, len(pasos), height_ratios=[2.6, 1],
                               hspace=0.12, wspace=0.05,
                               left=0.05, right=0.985, top=0.855, bottom=0.10)

    for k, paso in enumerate(pasos):
        ax = fig.add_subplot(rejilla[0, k])
        ax.imshow(fotogramas[paso])
        ax.set_xticks([])
        ax.set_yticks([])
        for lado in ax.spines.values():
            lado.set_color(GRIS)
            lado.set_linewidth(0.8)
        torque = TORQUE[int(acciones[paso])] if paso < len(acciones) else "—"
        ax.set_title(f"paso {paso}", color=TINTA, fontsize=10.5, fontweight="600", pad=6)
        ax.set_xlabel(f"altura {alturas[paso]:+.2f}".replace("-", "\u2212") + f"  ·  torque {torque}",
                      color=TINTA2, fontsize=9, labelpad=6)

    eje = fig.add_subplot(rejilla[1, :])
    eje.set_facecolor(SUP)
    eje.plot(alturas, color=AZUL, lw=2.0, zorder=3)
    eje.axhline(ALTURA_META, color=TINTA2, lw=1.1, ls=(0, (5, 4)), alpha=0.8, zorder=2)
    eje.scatter(pasos, alturas[pasos], s=42, color=AZUL, zorder=4,
                edgecolor=SUP, linewidth=1.4)
    eje.text(1, ALTURA_META + 0.12, "altura objetivo", color=TINTA2, fontsize=9, va="bottom")
    eje.set_xlim(0, len(alturas) - 1)
    eje.set_ylim(-2.35, 1.75)
    eje.set_xlabel("paso del episodio", color=TINTA2, fontsize=9.5)
    eje.set_ylabel("altura del extremo", color=TINTA2, fontsize=9.5)
    eje.grid(axis="y", color=GRIS, lw=0.6, alpha=0.45)
    eje.set_axisbelow(True)
    for lado in ("top", "right"):
        eje.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        eje.spines[lado].set_color(GRIS)
        eje.spines[lado].set_linewidth(0.8)
    eje.tick_params(colors=TINTA2, labelsize=9, length=3)

    fig.suptitle("El brazo no sube de un tirón: acumula energía en oscilaciones cada vez más amplias",
                 color=TINTA, fontsize=13, fontweight="600", x=0.045, ha="left", y=0.965)
    fig.savefig(salida, facecolor=SUP)
    plt.close(fig)
    return Path(salida)
