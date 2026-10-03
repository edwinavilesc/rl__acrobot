"""
Ver el agente DQN entrenado jugando Acrobot-v1 en tiempo real.

Uso:
    python ver_en_vivo.py              # 3 episodios con el agente entrenado
    python ver_en_vivo.py --episodios 5
    python ver_en_vivo.py --semilla 10001
"""
import argparse
import time
from pathlib import Path

import gymnasium as gym
import torch

from acrobot_dqn.dqn_acrobot_v3 import DEVICE, QNetwork

PESOS = Path("resultados/dqn_acrobot_1200ep_state_dict.pt")


def ver(episodios: int = 3, semilla_base: int = 10_000, pausa: float = 0.02):
    # Cargar red entrenada
    red = QNetwork(6, 3, 128).to(DEVICE)
    red.load_state_dict(torch.load(PESOS, map_location=DEVICE, weights_only=True))
    red.eval()

    print(f"Dispositivo: {DEVICE}  |  Pesos: {PESOS.name}")
    print(f"Corriendo {episodios} episodio(s) — cierra la ventana para salir\n")

    env = gym.make("Acrobot-v1", render_mode="human")

    for ep in range(episodios):
        obs, _ = env.reset(seed=semilla_base + ep)
        retorno, pasos = 0.0, 0

        while True:
            env.render()
            with torch.no_grad():
                accion = int(
                    red(torch.as_tensor(obs, dtype=torch.float32,
                                        device=DEVICE).unsqueeze(0)).argmax()
                )
            obs, reward, terminated, truncated, _ = env.step(accion)
            retorno += reward
            pasos += 1
            time.sleep(pausa)           # ajusta la velocidad (0 = máxima velocidad)

            if terminated or truncated:
                estado = "META ✓" if terminated else "truncado"
                print(f"  Episodio {ep + 1:2d}  |  retorno {retorno:6.0f}  "
                      f"|  pasos {pasos:3d}  |  {estado}")
                time.sleep(0.5)         # pausa breve entre episodios
                break

    env.close()
    print("\nListo.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodios", type=int, default=3)
    p.add_argument("--semilla",   type=int, default=10_000)
    p.add_argument("--pausa",     type=float, default=0.02,
                   help="segundos entre frames (0 = sin pausa)")
    args = p.parse_args()
    ver(args.episodios, args.semilla, args.pausa)
