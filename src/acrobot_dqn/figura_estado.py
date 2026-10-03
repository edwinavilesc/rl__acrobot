"""Dibuja el estado y las acciones de Acrobot-v1 sobre el propio mecanismo.

Lo primero que hay que entender en un problema de aprendizaje por refuerzo es qué
significa cada variable del estado, porque el estado es la entrada de la red y tiene
que ser numeros. Esta figura pone las seis componentes de la observacion y las tres
acciones sobre el dibujo del brazo, con los valores reales de un instante concreto de
la corrida.

La geometria es la misma que usa Gymnasium para renderizar: el primer eslabon sale del
anclaje con angulo theta1 medido desde la vertical hacia abajo, y el segundo continua
con theta1 + theta2. Es decir que theta2 NO se mide desde la vertical sino desde la
prolongacion del primer eslabon: es el angulo del codo, y asi se dibuja su arco.
"""
from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend fijado arriba)
from matplotlib.patches import Arc, FancyArrowPatch  # noqa: E402

from .dqn_acrobot_v3 import DEVICE, QNetwork  # noqa: E402

SUP, TINTA, TINTA2 = "#fcfcfb", "#0b0b0b", "#52514e"
AZUL, NARANJA, GRIS = "#2a78d6", "#eb6834", "#b9b7b0"
ALTURA_META = 1.0
ACCIONES = [("0", "−1", "torque horario"), ("1", "0", "sin torque"), ("2", "+1", "torque antihorario")]


def _posiciones(theta1: float, theta2: float):
    """Coordenadas del codo y del extremo libre, con el anclaje en el origen."""
    codo = (np.sin(theta1), -np.cos(theta1))
    extremo = (codo[0] + np.sin(theta1 + theta2), codo[1] - np.cos(theta1 + theta2))
    return codo, extremo


def _instante(checkpoint: Path, semilla: int, paso: int):
    """Devuelve la observacion del agente entrenado en el paso pedido."""
    red = QNetwork(6, 3).to(DEVICE)
    red.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    red.eval()
    env = gym.make("Acrobot-v1")
    obs, _ = env.reset(seed=semilla)
    for _ in range(paso):
        with torch.no_grad():
            accion = int(red(torch.as_tensor(obs, dtype=torch.float32,
                                             device=DEVICE).unsqueeze(0)).argmax())
        obs, _, terminado, truncado, _ = env.step(accion)
        if terminado or truncado:
            break
    env.close()
    return obs


def _panel_mecanismo(ax, obs):
    """El brazo con theta1, theta2 y las tres acciones señaladas sobre el motor."""
    cos1, sen1, cos2, sen2, omega1, omega2 = (float(x) for x in obs)
    theta1 = np.arctan2(sen1, cos1)
    theta2 = np.arctan2(sen2, cos2)
    codo, extremo = _posiciones(theta1, theta2)

    # direcciones unitarias de cada eslabon y sus perpendiculares, para colocar texto
    d1 = np.array([np.sin(theta1), -np.cos(theta1)])
    d2 = np.array([np.sin(theta1 + theta2), -np.cos(theta1 + theta2)])
    n1 = np.array([-d1[1], d1[0]])

    ax.axhline(ALTURA_META, color=TINTA2, lw=1.2, ls=(0, (5, 4)), alpha=0.85, zorder=2)
    ax.text(2.15, ALTURA_META + 0.08, "altura objetivo", color=TINTA2, fontsize=9.5,
            ha="right", va="bottom")

    # referencias desde las que se mide cada angulo: theta1 desde la vertical del
    # anclaje, theta2 desde la PROLONGACION del primer eslabon, no desde la vertical
    ax.plot([0, 0], [0, -1.15], color=GRIS, lw=1.0, ls=(0, (2, 3)), zorder=2)
    prolongacion = np.array(codo) + 1.0 * d1
    ax.plot([codo[0], prolongacion[0]], [codo[1], prolongacion[1]], color=GRIS, lw=1.0,
            ls=(0, (2, 3)), zorder=2)

    ax.plot([0, codo[0]], [0, codo[1]], color=AZUL, lw=7, solid_capstyle="round", zorder=4)
    ax.plot([codo[0], extremo[0]], [codo[1], extremo[1]], color=AZUL, lw=7,
            solid_capstyle="round", zorder=4)
    ax.scatter([0], [0], s=95, color=TINTA, zorder=6)
    ax.scatter([codo[0]], [codo[1]], s=170, color=NARANJA, zorder=6)
    ax.scatter([extremo[0]], [extremo[1]], s=95, color=AZUL, zorder=6,
               edgecolor=SUP, linewidth=1.8)

    # arcos de los dos angulos. Cada uno se abre desde su propia referencia: theta1
    # desde la vertical hacia abajo (-90 grados en coordenadas de Arc) y theta2 desde
    # la prolongacion del primer eslabon, que ya esta girada theta1.
    for centro, base, ang, radio, etiqueta in (
            (np.zeros(2), -90.0, theta1, 0.46, r"$\theta_1$"),
            (np.array(codo), -90.0 + np.degrees(theta1), theta2, 0.38, r"$\theta_2$")):
        grados = np.degrees(ang)
        # Arc dibuja en sentido antihorario: para un angulo negativo hay que invertir
        # los extremos, o traza el arco largo en lugar del corto
        desde, hasta = (base, base + grados) if grados >= 0 else (base + grados, base)
        ax.add_patch(Arc(tuple(centro), 2 * radio, 2 * radio, angle=0,
                         theta1=desde, theta2=hasta, color=TINTA2, lw=1.5, zorder=5))
        medio = np.radians(base + grados / 2)
        pos = centro + (radio + 0.22) * np.array([np.cos(medio), np.sin(medio)])
        ax.text(pos[0], pos[1], etiqueta, color=TINTA, fontsize=14, ha="center", va="center")

    # etiquetas de las piezas, apartadas del brazo
    ax.annotate("anclaje fijo", xy=(0, 0), xytext=(0.42, 0.52), color=TINTA2, fontsize=9.5,
                arrowprops=dict(arrowstyle="-", color=GRIS, lw=1.0))
    # la etiqueta del motor sale perpendicular al primer eslabon, hacia el lado libre
    lado = n1 if np.dot(n1, np.array(extremo) - np.array(codo)) < 0 else -n1
    pos_codo = np.array(codo) + 1.30 * lado
    # la etiqueta se queda dentro del panel aunque el brazo caiga cerca del borde
    pos_codo[0] = float(np.clip(pos_codo[0], -1.35, 1.35))
    ax.annotate("articulación con motor\nes la única que actúa", xy=codo,
                xytext=(pos_codo[0], pos_codo[1]), color=NARANJA, fontsize=9.5,
                fontweight="600", ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=NARANJA, lw=1.0, alpha=0.55))
    # el extremo libre se etiqueta de lado, para no chocar con el pie de la figura
    n2 = np.array([-d2[1], d2[0]])
    pos_ext = np.array(extremo) + 0.10 * d2 + 0.52 * (n2 if n2[0] < 0 else -n2)
    ax.text(pos_ext[0], pos_ext[1], "extremo libre", color=AZUL, fontsize=9.5,
            fontweight="600", ha="center", va="center")

    # las dos acciones con torque: un par de flechas curvas en el lado libre del motor.
    # Van ahi y no en los huecos entre eslabones porque esos ya los ocupan los arcos
    # de theta1 y theta2.
    centro_acc = np.array(codo) + 0.52 * lado
    for signo in (+1, -1):
        ax.add_patch(FancyArrowPatch(
            tuple(centro_acc + np.array([-signo * 0.24, -0.09])),
            tuple(centro_acc + np.array([-signo * 0.07, 0.21])),
            connectionstyle=f"arc3,rad={0.55 * signo}", arrowstyle="-|>",
            mutation_scale=12, color=NARANJA, lw=1.7, zorder=5))

    ax.text(0.5, 0.115,
            "las líneas punteadas son las referencias de cada ángulo:\n"
            "$\\theta_1$ desde la vertical, $\\theta_2$ desde el propio primer eslabón",
            transform=ax.transAxes, color=TINTA2, fontsize=9, ha="center")
    ax.text(0.5, 0.045, "acciones:  −1  ·  0  ·  +1  (torque sobre esa articulación)",
            transform=ax.transAxes, color=NARANJA, fontsize=10.5, fontweight="600",
            ha="center")
    ax.set_xlim(-2.05, 2.05)
    ax.set_ylim(-2.55, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("El mecanismo y lo que el agente decide", color=TINTA, fontsize=11.5,
                 fontweight="600", pad=10, loc="left")
    return theta1, theta2, omega1, omega2


def _panel_observacion(ax, obs, theta1, theta2):
    """La observacion real, componente por componente."""
    cos1, sen1, cos2, sen2, omega1, omega2 = (float(x) for x in obs)
    filas = [
        ("0", r"$\cos\theta_1$", cos1, "[−1, 1]", "orientación del primer eslabón"),
        ("1", r"$\sin\theta_1$", sen1, "[−1, 1]", "el mismo ángulo, sin discontinuidad"),
        ("2", r"$\cos\theta_2$", cos2, "[−1, 1]", "orientación del segundo respecto al primero"),
        ("3", r"$\sin\theta_2$", sen2, "[−1, 1]", ""),
        ("4", r"$\omega_1$", omega1, "±4π", "velocidad angular del primer eslabón"),
        ("5", r"$\omega_2$", omega2, "±9π", "velocidad angular del segundo"),
    ]
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.5, len(filas) + 1.9)
    ax.axis("off")
    ax.text(0.0, len(filas) + 1.25, "índice", color=TINTA2, fontsize=9, fontweight="600")
    ax.text(0.10, len(filas) + 1.25, "componente", color=TINTA2, fontsize=9, fontweight="600")
    ax.text(0.34, len(filas) + 1.25, "valor aquí", color=TINTA2, fontsize=9, fontweight="600",
            ha="right")
    ax.text(0.40, len(filas) + 1.25, "rango", color=TINTA2, fontsize=9, fontweight="600")
    ax.text(0.58, len(filas) + 1.25, "qué significa", color=TINTA2, fontsize=9, fontweight="600")
    ax.plot([0, 1], [len(filas) + 1.05, len(filas) + 1.05], color=GRIS, lw=0.9)

    for k, (idx, nombre, valor, rango, significado) in enumerate(filas):
        y = len(filas) - k
        ax.text(0.0, y, idx, color=TINTA2, fontsize=9.5)
        ax.text(0.10, y, nombre, color=TINTA, fontsize=12)
        ax.text(0.34, y, f"{valor:+.3f}".replace("-", "−"), color=AZUL, fontsize=10.5,
                fontweight="600", ha="right", family="monospace")
        ax.text(0.40, y, rango, color=TINTA2, fontsize=9)
        ax.text(0.58, y, significado, color=TINTA2, fontsize=9)

    ax.plot([0, 1], [0.45, 0.45], color=GRIS, lw=0.9)
    ax.text(0.0, -0.55,
            "Los ángulos llegan como seno y coseno, no como θ: así 0 y 2π son el mismo punto\n"
            "y la red no tiene que aprender esa discontinuidad. Las seis son continuas, que es\n"
            "lo que obliga a una red o a discretizar (§10.4).",
            color=TINTA2, fontsize=9, va="bottom")
    ax.set_title("La observación, en el instante dibujado", color=TINTA, fontsize=11.5,
                 fontweight="600", pad=10, loc="left")


def dibujar(checkpoint="resultados/dqn_acrobot_1200ep_state_dict.pt",
            salida="resultados/estado_y_acciones.png",
            semilla: int = 10_000, paso: int = 29) -> Path:
    """Escribe la figura del estado y las acciones, y devuelve su ruta."""
    obs = _instante(Path(checkpoint), semilla, paso)

    fig, (izq, der) = plt.subplots(1, 2, figsize=(13.0, 6.0), dpi=175,
                                   gridspec_kw={"width_ratios": [1, 1.35]})
    fig.patch.set_facecolor(SUP)
    for ax in (izq, der):
        ax.set_facecolor(SUP)

    theta1, theta2, _, _ = _panel_mecanismo(izq, obs)
    _panel_observacion(der, obs, theta1, theta2)

    fig.suptitle("Qué observa el agente y qué puede hacer",
                 color=TINTA, fontsize=13, fontweight="600", x=0.012, ha="left", y=0.965)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.85, bottom=0.05, wspace=0.02)
    fig.savefig(salida, facecolor=SUP)
    plt.close(fig)

    # deja tambien los numeros como artefacto, para que el README no los invente
    datos = {"semilla": semilla, "paso": paso,
             "observacion": [round(float(x), 4) for x in obs],
             "theta1_rad": round(float(theta1), 4), "theta2_rad": round(float(theta2), 4)}
    Path(salida).with_suffix(".json").write_text(
        json.dumps(datos, indent=2), encoding="utf-8")
    return Path(salida)
