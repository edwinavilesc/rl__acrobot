"""Mide el sesgo de los valores Q: lo que la red predice frente a lo que el agente obtiene.

Hay dos preguntas distintas y este modulo las separa, porque mezclarlas fue el error
que nos costo mas caro en §10.3.

**Sesgo total.** Para cada estado que el agente visita, `max_a Q(s,a)` frente al retorno
descontado que de verdad consigue desde ese estado. Es la magnitud que interesa al
lector, pero es una suma de efectos: el optimismo del maximo, el retraso de la red
objetivo, el error de aproximacion y la convergencia incompleta. Medirla en un solo
estado -el inicial- da una foto parcial, asi que aqui se estratifica por cuantos pasos
le faltan al agente para llegar a la meta.

Lo que este modulo **no** mide es el sesgo de seleccion aislado -la diferencia entre
`max_a Q_target(s,a)` y `Q_target(s, argmax_a Q_online(s,a))` sobre el mismo lote-, que es
la unica cantidad que identifica sin mezcla el optimismo descrito por van Hasselt, Guez y
Silver (2016). Requiere la red objetivo en el momento de la actualizacion y solo guardamos
la red en linea al final del entrenamiento. Queda declarado en §10.3 como la medicion
pendiente.
"""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from .dqn_acrobot_v3 import DEVICE, QNetwork

GAMMA = 0.99
SEMILLAS_EVAL = range(10_000, 10_100)
# Tramos de "pasos que le faltan al agente para alcanzar la meta". El ultimo queda
# abierto porque los estados muy lejanos son pocos y conviene agruparlos.
TRAMOS = [(0, 20), (20, 50), (50, 80), (80, 90), (90, 100), (100, 10**9)]


def cargar(ruta: Path) -> QNetwork:
    """Carga una red entrenada desde un state_dict y la deja en modo evaluacion."""
    red = QNetwork(6, 3).to(DEVICE)
    red.load_state_dict(torch.load(ruta, map_location=DEVICE))
    red.eval()
    return red


def recorrer(red: QNetwork, episodios: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recorre la evaluacion greedy y devuelve, por estado visitado:

    pasos_restantes, valor predicho `max_a Q(s,a)` y retorno descontado real.
    """
    env = gym.make("Acrobot-v1")
    restantes, predicho, real = [], [], []
    for k, semilla in enumerate(SEMILLAS_EVAL):
        if k >= episodios:
            break
        obs, _ = env.reset(seed=semilla)
        valores, recompensas = [], []
        while True:
            with torch.no_grad():
                q = red(torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0))
            valores.append(float(q.max()))
            obs, r, terminado, truncado, _ = env.step(int(q.argmax()))
            recompensas.append(r)
            if terminado or truncado:
                break

        # retorno descontado hacia adelante desde cada estado de la trayectoria
        acumulado, retornos = 0.0, []
        for r in reversed(recompensas):
            acumulado = r + GAMMA * acumulado
            retornos.append(acumulado)
        retornos.reverse()

        n = len(recompensas)
        restantes.extend(range(n, 0, -1))
        predicho.extend(valores)
        real.extend(retornos)
    env.close()
    return np.array(restantes), np.array(predicho), np.array(real)


def sesgo_total(red: QNetwork, episodios: int = 100) -> dict:
    """Sesgo global, en el estado inicial y estratificado por distancia a la meta."""
    restantes, predicho, real = recorrer(red, episodios)
    sesgo = predicho - real
    # dentro de cada trayectoria `restantes` decrece de n a 1, asi que el primer estado de
    # cada episodio es el unico en que el valor sube respecto al anterior
    es_inicial = np.r_[True, restantes[1:] > restantes[:-1]]

    por_tramo = {}
    for lo, hi in TRAMOS:
        m = (restantes >= lo) & (restantes < hi)
        if m.any():
            etiqueta = f"{lo}-{hi - 1}" if hi < 10**9 else f"{lo}+"
            # el maximo retorno descontado posible con `lo` pasos por delante: sirve para
            # saber si el signo del sesgo estaba disponible o forzado por la aritmetica
            cota = -(1 - GAMMA ** (lo - 1)) / (1 - GAMMA) if lo > 0 else 0.0
            por_tramo[etiqueta] = {
                "n": int(m.sum()),
                "valor_predicho": round(float(predicho[m].mean()), 2),
                "retorno_real": round(float(real[m].mean()), 2),
                "sesgo_medio": round(float(sesgo[m].mean()), 2),
                "cota_superior_del_retorno_real": round(float(cota), 2),
            }

    return {
        "estados_visitados": int(len(sesgo)),
        "sesgo_global": round(float(sesgo.mean()), 2),
        "sesgo_en_s0": round(float(sesgo[es_inicial].mean()), 2),
        "v_predicho_en_s0": round(float(predicho[es_inicial].mean()), 2),
        "retorno_real_desde_s0": round(float(real[es_inicial].mean()), 2),
        "por_pasos_restantes": por_tramo,
    }
