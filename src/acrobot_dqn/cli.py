"""Interfaz de línea de comandos del entregable — `acrobot`.

Contenido:
  - cmd_*()   : una función por subcomando (version, inspect, train, evaluate, plot,
                results, verify, rangos, ablacion, minilotes, double, sesgo, ambiente,
                tabular, estado)
  - _parser() : el parser de argparse que conecta comandos con sus opciones
  - main()    : punto de entrada

Aquí solo hay lectura de argumentos y formato de salida. El comportamiento vive en
`acrobot_dqn.dqn_acrobot_v3`, que no se modifica.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import __version__
from .dqn_acrobot_v3 import DEVICE, Config, evaluate, train_dqn

RAIZ = Path(__file__).resolve().parents[2]
RESULTADOS = RAIZ / "resultados"
UMBRAL_OFICIAL = -100.0
ALEATORIO_MEDIDO = -500.0

# Las seis componentes de la observacion de Acrobot-v1, en el orden en que el
# entorno las entrega, con el rango que declara su observation_space.
OBS_NOMBRES = ["cos theta1", "sen theta1", "cos theta2", "sen theta2", "omega1", "omega2"]
OBS_DECLARADO = [(-1, 1), (-1, 1), (-1, 1), (-1, 1),
                 (-4 * np.pi, 4 * np.pi), (-9 * np.pi, 9 * np.pi)]


# ────────────────────────────────────────────────────────────────── subcomandos

def cmd_version(_args: argparse.Namespace) -> None:
    """Versiones del entorno de ejecución."""
    import platform

    import gymnasium as gym
    import torch
    print(f"acrobot-dqn   {__version__}")
    print(f"Python        {platform.python_version()}")
    print(f"PyTorch       {torch.__version__}")
    print(f"Gymnasium     {gym.__version__}")
    print(f"NumPy         {np.__version__}")
    print(f"dispositivo   {DEVICE}")
    if str(DEVICE) == "mps":
        print("\n  Aviso: en mps los resultados no son idénticos bit a bit a los de cpu,")
        print("  porque seed_everything no siembra el generador de mps.")


def cmd_inspect(_args: argparse.Namespace) -> None:
    """Audita Acrobot-v1: espacios, dtype, recompensa y terminación."""
    import gymnasium as gym
    env = gym.make("Acrobot-v1")
    print("Acrobot-v1")
    print(f"  observation_space   {env.observation_space}")
    print(f"    shape             {env.observation_space.shape}   dtype {env.observation_space.dtype}")
    print(f"  action_space        {env.action_space}")
    print(f"    ¿es Discrete?     {isinstance(env.action_space, gym.spaces.Discrete)}  (DQN lo exige)")
    print(f"  max_episode_steps   {env.spec.max_episode_steps}")
    print(f"  reward_threshold    {env.spec.reward_threshold}   ← criterio oficial de resolución")

    n = 30
    recompensas, terminados, truncados, largos, retornos = set(), 0, 0, [], []
    for ep in range(n):
        env.reset(seed=1000 + ep)
        env.action_space.seed(1000 + ep)   # sin esto el muestreo no es reproducible
        pasos, retorno = 0, 0.0
        while True:
            _, r, te, tr, _ = env.step(env.action_space.sample())
            recompensas.add(float(r))
            retorno += float(r)
            pasos += 1
            if te or tr:
                terminados += te
                truncados += tr
                largos.append(pasos)
                retornos.append(retorno)
                break
    env.close()
    print(f"\n  {n} episodios con política aleatoria:")
    print(f"    recompensas observadas  {sorted(recompensas)}")
    print(f"    alcanzan la meta        {terminados}/{n}")
    print(f"    truncan a los 500 pasos {truncados}/{n}")
    print(f"    retorno medio           {np.mean(retornos):.2f}   ← suma de recompensas, no −pasos")
    print(f"    pasos medios            {np.mean(largos):.2f}")
    print("\n  La recompensa es -1 por paso salvo en el que alcanza la meta, que da 0")
    print("  (gymnasium: reward = -1.0 if not terminated else 0.0). La politica aleatoria")
    print("  casi nunca llega a ese paso, de modo que su retorno se acerca al suelo de -500.")


def cmd_train(args: argparse.Namespace) -> None:
    """Entrena el agente. Por defecto en modo smoke, para no gastar una corrida."""
    episodios = 10 if args.mode == "smoke" else args.episodes
    cfg = replace(Config(), episodes=episodios, seed=args.seed)
    salida = Path(args.output)
    salida.mkdir(parents=True, exist_ok=True)
    print(f"modo {args.mode} · {episodios} episodios · semilla {args.seed} · {DEVICE}")
    print(f"artefactos en {salida}\n")
    q, h = train_dqn(cfg, artifact_dir=salida)
    r = np.asarray(h["returns"], float)
    print(f"\nterminado en {h['seconds']:.1f} s")
    print(f"  media últimos 100      {r[-min(100, len(r)):].mean():.2f}")
    print(f"  mejor episodio         {r.max():.0f} (episodio {int(np.argmax(r)) + 1})")
    print(f"  alcanzan la meta       {100 * np.mean(h['terminated']):.1f} %")
    if args.mode == "smoke":
        print("\n  El smoke test verifica el pipeline, no el desempeño.")
        print("  Con 10 episodios ε sigue en ~0,92: el agente es casi aleatorio y da ≈ -500.")
        print("  Para la corrida real:  acrobot train --mode full")


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evalúa una política greedy con semillas disjuntas de las de entrenamiento."""
    import torch

    from .dqn_acrobot_v3 import QNetwork
    ruta = Path(args.checkpoint)
    if not ruta.exists():
        sys.exit(f"no encuentro el checkpoint: {ruta}")
    cfg = replace(Config(), eval_episodes=args.episodes)
    q = QNetwork(6, 3, cfg.hidden).to(DEVICE)
    q.load_state_dict(torch.load(ruta, map_location=DEVICE))
    q.eval()
    print(f"evaluando {ruta.name} · {args.episodes} episodios · semillas {args.seed_start}–"
          f"{args.seed_start + args.episodes - 1}\n")
    m = evaluate(q, cfg, n=args.episodes, seed_start=args.seed_start)
    ev = np.asarray(m["returns"], float)
    ic = 1.96 * ev.std(ddof=1) / np.sqrt(len(ev))
    print(f"  retorno medio           {m['mean_return']:.2f}")
    print(f"  IC 95 %                 [{ev.mean() - ic:.2f}, {ev.mean() + ic:.2f}]")
    print(f"  mediana / SD            {m['median_return']:.1f} / {m['sd_return']:.2f}")
    print(f"  pasos medios            {m['mean_steps']:.2f}")
    print(f"  terminación verdadera   {int(m['terminal_rate'] * len(ev))}/{len(ev)}")
    print(f"  mejores que {UMBRAL_OFICIAL:.0f}         {int((ev > UMBRAL_OFICIAL).sum())}/{len(ev)}")
    print(f"\n  referencias: umbral oficial {UMBRAL_OFICIAL:.0f} · aleatorio {ALEATORIO_MEDIDO:.0f}")
    if args.save:
        Path(args.save).write_text(json.dumps(m, indent=2))
        print(f"  guardado en {args.save}")


def cmd_plot(args: argparse.Namespace) -> None:
    """Regenera la curva de aprendizaje desde la historia guardada."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hist = Path(args.history)
    if not hist.exists():
        sys.exit(f"no encuentro la historia: {hist}")
    r = np.asarray(json.loads(hist.read_text())["returns"], float)
    w = min(args.window, len(r))
    ma = np.convolve(r, np.ones(w) / w, "valid")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(r, alpha=.35, label="retorno/episodio")
    ax.plot(np.arange(w - 1, len(r)), ma, label=f"media móvil {w}")
    ax.axhline(UMBRAL_OFICIAL, ls="--", lw=1, color="tab:red", label="umbral oficial (-100)")
    ax.axhline(ALEATORIO_MEDIDO, ls=":", lw=1, color="gray", label="política aleatoria (-500)")
    ax.set_xlabel("Episodio")
    ax.set_ylabel("Retorno")
    ax.set_title(f"DQN · Acrobot-v1 · {len(r)} episodios")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output, dpi=180)
    print(f"curva escrita en {args.output}")


def cmd_results(_args: argparse.Namespace) -> None:
    """Imprime los resultados verificados que respaldan el README."""
    f = RESULTADOS / "semillas" / "RESUMEN_5_SEMILLAS.json"
    if not f.exists():
        sys.exit(f"no encuentro {f}")
    d = json.loads(f.read_text())
    print("Evaluación greedy · 100 episodios por semilla · semillas 10000-10099\n")
    print(f"  {'semilla':>8} {'retorno':>9} {'SD':>7} {'>-100':>8} {'terminación':>12}")
    for s in d["semillas"]:
        print(f"  {s['seed']:>8} {s['eval_medio']:>9.2f} {s['eval_sd']:>7.2f} "
              f"{s['mejores_que_-100']:>6}/100 {int(s['terminacion'] * 100):>9}/100")
    a = d["agregado"]
    print(f"\n  media de las medias       {a['media_de_medias']:.2f}")
    print(f"  SD entre semillas         {a['sd_entre_semillas']:.2f}")
    print(f"  SD dentro de cada semilla {a['sd_dentro_promedio']:.2f}")
    print(f"  terminaciones             {a['terminaciones']}/500")
    print(f"  mejores que -100          {a['mejores_que_-100']}/500")
    print(f"\n  La variabilidad entre semillas es "
          f"{a['sd_dentro_promedio'] / a['sd_entre_semillas']:.1f} veces menor que la de dentro:")
    print("  el resultado no depende de la inicialización.")


def cmd_verify(_args: argparse.Namespace) -> None:
    """Comprueba que cada cifra del README coincide con los artefactos."""
    script = RAIZ / "verificar_resultados.py"
    if not script.exists():
        sys.exit(f"no encuentro {script}")
    import subprocess
    sys.exit(subprocess.call([sys.executable, str(script), str(RAIZ)]))


def cmd_rangos(args: argparse.Namespace) -> None:
    """Mide el rango de cada componente de la observación con dos políticas."""
    import gymnasium as gym
    import torch

    from .dqn_acrobot_v3 import QNetwork

    red = None
    if not args.solo_aleatoria:
        red = QNetwork(6, 3)
        red.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        red.eval()

    def recorrer(politica, semilla_base, episodios):
        env = gym.make("Acrobot-v1")
        obs = []
        for ep in range(episodios):
            o, _ = env.reset(seed=semilla_base + ep)
            env.action_space.seed(semilla_base + ep)
            while True:
                obs.append(o)
                o, _, te, tr, _ = env.step(politica(o, env))
                if te or tr:
                    obs.append(o)      # el estado final tambien cuenta para el rango
                    break
        env.close()
        return np.array(obs)

    def greedy(o, _env):
        with torch.no_grad():
            return int(red(torch.tensor(o).unsqueeze(0)).argmax())

    medidas = {"aleatoria": recorrer(lambda o, e: e.action_space.sample(), 1000, 30)}
    if red is not None:
        medidas["entrenada"] = recorrer(greedy, 10_000, 100)

    salida = {"declarado": {n: list(d) for n, d in zip(OBS_NOMBRES, OBS_DECLARADO, strict=True)}}
    print(f"{'componente':12} {'declarado':>22} {'aleatoria':>18} {'entrenada':>18}")
    for i, n in enumerate(OBS_NOMBRES):
        fila = {}
        for k, obs in medidas.items():
            fila[k] = [round(float(obs[:, i].min()), 3), round(float(obs[:, i].max()), 3)]
        salida[n] = fila
        dec = f"[{OBS_DECLARADO[i][0]:+.3f}, {OBS_DECLARADO[i][1]:+.3f}]"
        al = f"[{fila['aleatoria'][0]:+.3f}, {fila['aleatoria'][1]:+.3f}]"
        en = f"[{fila['entrenada'][0]:+.3f}, {fila['entrenada'][1]:+.3f}]" if "entrenada" in fila else "-"
        print(f"{n:12} {dec:>22} {al:>18} {en:>18}")

    print("\n  aleatoria: 30 episodios, semillas 1000-1029")
    if red is not None:
        print("  entrenada: 100 episodios, semillas 10000-10099 (las de evaluación)")
    if args.save:
        Path(args.save).write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  guardado en {args.save}")



def cmd_ablacion(args: argparse.Namespace) -> None:
    """Entrena la versión que confunde truncación con terminación y la compara."""
    from dataclasses import replace

    from .ablacion import train_dqn_sin_distinguir_truncacion
    from .dqn_acrobot_v3 import Config, evaluate

    if args.solo_figura:
        from .figura_ablacion import dibujar
        ruta = dibujar(RESULTADOS, RESULTADOS / "ablacion_truncacion.png")
        print(f"figura regenerada desde los artefactos: {ruta}")
        return

    cfg = replace(Config(), episodes=args.episodes, seed=args.seed)
    print(f"Entrenando la ablación: {cfg.episodes} episodios, semilla {cfg.seed}")
    print("Única diferencia: buf.push(..., terminated or truncated)\n")
    q, h = train_dqn_sin_distinguir_truncacion(cfg, Path(args.output))
    ev = evaluate(q, cfg, n=100)
    print(f"\n  ablación  retorno medio {ev['mean_return']:.2f} · "
          f"terminaciones {int(ev['terminal_rate']*100)}/100")
    ref = RESULTADOS / "evaluation_metrics_100ep.json"
    if ref.exists():
        r = json.loads(ref.read_text(encoding="utf-8"))
        print(f"  correcto  retorno medio {r['mean_return']:.2f} · "
              f"terminaciones {int(r['terminal_rate']*100)}/100")
        print(f"\n  diferencia {abs(ev['mean_return'] - r['mean_return']):.2f} · "
              f"desviación entre semillas de la versión correcta: 1,42")



def cmd_minilotes(args: argparse.Namespace) -> None:
    """Cuenta en cuantos minilotes entra el error de la ablacion (§10.2)."""
    from .minilotes import informe

    ruta = Path(args.historia)
    if not ruta.exists():
        print(f"no encuentro {ruta}")
        return
    print("El error de §10.2 mal etiqueta la ultima transicion de cada episodio truncado.")
    print("Lo que importa no es que fraccion del buffer ocupan, sino en cuantos lotes entran.\n")
    informe(ruta)


def cmd_double(args: argparse.Namespace) -> None:
    """Entrena Double DQN y lo compara con el DQN del entregable."""
    from dataclasses import replace

    from .double_dqn import train_double_dqn
    from .dqn_acrobot_v3 import Config, evaluate

    if args.solo_figura:
        from .figura_double import dibujar
        print(f"figura regenerada desde los artefactos: {dibujar(RESULTADOS, RESULTADOS / 'double_dqn.png')}")
        return

    cfg = replace(Config(), episodes=args.episodes, seed=args.seed)
    print(f"Entrenando Double DQN: {cfg.episodes} episodios, semilla {cfg.seed}")
    print("Diferencia con DQN: la accion del objetivo la elige la red en linea\n")
    q, _ = train_double_dqn(cfg, Path(args.output))
    ev = evaluate(q, cfg, n=100)
    print(f"\n  Double DQN  retorno medio {ev['mean_return']:.2f} · "
          f"terminaciones {int(ev['terminal_rate'] * 100)}/100")

    resumen = RESULTADOS / "double_dqn/RESUMEN_DOUBLE_DQN.json"
    if resumen.exists():
        r = json.loads(resumen.read_text(encoding="utf-8"))
        dqn = r["dqn"]["retorno_por_semilla"].get(str(args.seed))
        if dqn is not None:
            print(f"  DQN         retorno medio {dqn:.2f}   (misma semilla, misma configuracion)")



def cmd_sesgo(args: argparse.Namespace) -> None:
    """Mide el sesgo de los valores Q, estratificado por distancia a la meta."""
    from .sesgo import cargar, sesgo_total

    pesos = {"dqn": {42: RESULTADOS / "dqn_acrobot_1200ep_state_dict.pt"}, "double_dqn": {}}
    for s in (43, 44):
        pesos["dqn"][s] = RESULTADOS / f"pesos/dqn_seed_{s}.pt"
    for s in (42, 43, 44):
        pesos["double_dqn"][s] = RESULTADOS / f"pesos/double_dqn_seed_{s}.pt"

    tramos = ("0-19", "20-49", "50-79", "80-89", "90-99", "100+")
    print(f"{'metodo':12} {'sem':>4} {'en s0':>8} {'global':>8}  " +
          "  ".join(f"{t:>7}" for t in tramos))
    for metodo, rutas in pesos.items():
        for semilla, ruta in rutas.items():
            if not ruta.exists():
                print(f"{metodo:12} {semilla:>4}   falta {ruta.name}")
                continue
            r = sesgo_total(cargar(ruta), episodios=args.episodes)
            fila = "  ".join(
                (f"{r['por_pasos_restantes'][t]['sesgo_medio']:>+7.1f}"
                 if t in r["por_pasos_restantes"] else f"{'—':>7}")
                for t in tramos)
            print(f"{metodo:12} {semilla:>4} {r['sesgo_en_s0']:>8.2f} {r['sesgo_global']:>8.2f}  {fila}")
    print("\n  Columnas: pasos que le faltan al agente para alcanzar la meta.")
    print("  Sesgo = valor que la red predice menos el retorno descontado real desde ese estado.")



def cmd_ambiente(args: argparse.Namespace) -> None:
    """Renderiza un episodio del agente entrenado y dibuja la figura de §1."""
    from .figura_ambiente import dibujar
    ruta = dibujar(args.checkpoint, RESULTADOS / "ambiente_acrobot.png", args.seed)
    print(f"figura del entorno escrita en {ruta}")



def cmd_tabular(args: argparse.Namespace) -> None:
    """Entrena Q-learning tabular y lo compara con el DQN del entregable."""
    from dataclasses import replace

    from .tabular import ConfigTabular, entrenar, evaluar

    if args.solo_figura:
        from .figura_tabular import dibujar
        print(f"figura regenerada desde los artefactos: "
              f"{dibujar(RESULTADOS, RESULTADOS / 'tabular.png')}")
        return

    cfg = replace(ConfigTabular(), episodes=args.episodes, seed=args.seed,
                  bins_angulo=args.bins_angulo, bins_velocidad=args.bins_velocidad,
                  alpha=args.alpha)
    print(f"Q-learning tabular: {cfg.episodes} episodios, semilla {cfg.seed}")
    print(f"Rejilla {cfg.bins_angulo}x{cfg.bins_angulo}x{cfg.bins_velocidad}x{cfg.bins_velocidad}, "
          f"alpha {cfg.alpha}\n")
    tabla_q, h = entrenar(cfg)
    ev = evaluar(tabla_q, cfg)
    print(f"\n  tabular  retorno medio {ev['mean_return']:.2f} · "
          f"terminaciones {int(ev['terminal_rate'] * 100)}/100")
    print(f"           estados visitados {h['estados_visitados']:,} de {h['estados_totales']:,} "
          f"({h['estados_visitados'] / h['estados_totales']:.1%})")

    resumen = RESULTADOS / "tabular/RESUMEN_TABULAR.json"
    if resumen.exists():
        r = json.loads(resumen.read_text(encoding="utf-8"))
        print(f"  DQN      retorno medio {r['dqn_mismas_semillas']['retorno_medio']:.2f}   "
              f"(mismas semillas, mismo presupuesto)")



def cmd_estado(args: argparse.Namespace) -> None:
    """Dibuja el estado y las acciones sobre el mecanismo (figura de §2)."""
    from .figura_estado import dibujar
    ruta = dibujar(args.checkpoint, RESULTADOS / "estado_y_acciones.png", args.seed, args.paso)
    print(f"figura del estado escrita en {ruta}")


def cmd_demo(args: argparse.Namespace) -> None:
    """Corre el agente entrenado en vivo con ventana pygame (render_mode=human)."""
    import time
    import torch
    from .dqn_acrobot_v3 import QNetwork
    import gymnasium as gym

    ruta = Path(args.checkpoint)
    if not ruta.exists():
        sys.exit(f"no encuentro el checkpoint: {ruta}\n"
                 "  Pista: corre primero  acrobot train --mode full  para generarlo.")

    red = QNetwork(6, 3, 128).to(DEVICE)
    red.load_state_dict(torch.load(ruta, map_location=DEVICE, weights_only=True))
    red.eval()

    print(f"dispositivo : {DEVICE}")
    print(f"pesos       : {ruta.name}")
    print(f"episodios   : {args.episodios}  |  semilla base : {args.seed}")
    print(f"pausa       : {args.pausa} s/frame")
    print("─" * 48)

    env = gym.make("Acrobot-v1", render_mode="human")
    for ep in range(args.episodios):
        obs, _ = env.reset(seed=args.seed + ep)
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
            if args.pausa > 0:
                time.sleep(args.pausa)
            if terminated or truncated:
                estado = "META ✓" if terminated else "truncado"
                print(f"  ep {ep + 1:2d}  |  retorno {retorno:6.0f}  "
                      f"|  pasos {pasos:3d}  |  {estado}")
                time.sleep(0.4)
                break
    env.close()
    print("─" * 48)


# ────────────────────────────────────────────────────────────────────── parser

def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="acrobot",
        description="DQN sobre Acrobot-v1 — U3 Taller 2, Simulación y Aprendizaje por Refuerzo.")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("version", help="versiones del entorno y dispositivo").set_defaults(func=cmd_version)
    sub.add_parser("inspect", help="audita Acrobot-v1: espacios, recompensa y terminación") \
       .set_defaults(func=cmd_inspect)
    sub.add_parser("results", help="resultados verificados de las cinco semillas").set_defaults(func=cmd_results)
    sub.add_parser("verify", help="comprueba el README contra los artefactos").set_defaults(func=cmd_verify)

    es = sub.add_parser("estado", help="dibuja el estado y las acciones sobre el mecanismo")
    es.add_argument("--checkpoint", default=str(RESULTADOS / "dqn_acrobot_1200ep_state_dict.pt"))
    es.add_argument("--seed", type=int, default=10_000)
    es.add_argument("--paso", type=int, default=29)
    es.set_defaults(func=cmd_estado)

    dm = sub.add_parser("demo", help="corre el agente en vivo con ventana pygame")
    dm.add_argument("--checkpoint", default=str(RESULTADOS / "dqn_acrobot_1200ep_state_dict.pt"))
    dm.add_argument("--episodios", type=int, default=3)
    dm.add_argument("--seed", type=int, default=10_000)
    dm.add_argument("--pausa", type=float, default=0.02,
                    help="segundos entre frames (0 = máxima velocidad)")
    dm.set_defaults(func=cmd_demo)

    tb = sub.add_parser("tabular", help="entrena Q-learning tabular y lo compara con el DQN")
    tb.add_argument("--episodes", type=int, default=1200)
    tb.add_argument("--seed", type=int, default=42)
    tb.add_argument("--bins-angulo", type=int, default=12)
    tb.add_argument("--bins-velocidad", type=int, default=10)
    tb.add_argument("--alpha", type=float, default=0.1)
    tb.add_argument("--solo-figura", action="store_true",
                    help="regenera la figura de §10.4 sin reentrenar")
    tb.set_defaults(func=cmd_tabular)

    am = sub.add_parser("ambiente", help="renderiza un episodio y dibuja la figura del entorno")
    am.add_argument("--checkpoint", default=str(RESULTADOS / "dqn_acrobot_1200ep_state_dict.pt"))
    am.add_argument("--seed", type=int, default=10_000)
    am.set_defaults(func=cmd_ambiente)

    sg = sub.add_parser("sesgo", help="mide el sesgo de los valores Q por distancia a la meta")
    sg.add_argument("--episodes", type=int, default=100)
    sg.set_defaults(func=cmd_sesgo)

    dd = sub.add_parser("double", help="entrena Double DQN y lo compara con el DQN del entregable")
    dd.add_argument("--episodes", type=int, default=1200)
    dd.add_argument("--seed", type=int, default=42)
    dd.add_argument("--output", default="artifacts_double")
    dd.add_argument("--solo-figura", action="store_true",
                    help="regenera la figura de §10.3 sin reentrenar")
    dd.set_defaults(func=cmd_double)

    ab = sub.add_parser("ablacion", help="entrena la versión que confunde truncación con terminación")
    ab.add_argument("--episodes", type=int, default=1200)
    ab.add_argument("--seed", type=int, default=42)
    ab.add_argument("--output", default="artifacts_ablacion")
    ab.add_argument("--solo-figura", action="store_true",
                    help="regenera la figura de §10.2 sin reentrenar")
    ab.set_defaults(func=cmd_ablacion)

    ml = sub.add_parser("minilotes",
                        help="cuenta en cuántos minilotes entra el error de §10.2")
    ml.add_argument("--historia", default=str(RESULTADOS / "ablacion/ablacion_seed_42.json"))
    ml.set_defaults(func=cmd_minilotes)

    ra = sub.add_parser("rangos", help="mide los rangos de observación con las dos políticas")
    ra.add_argument("--checkpoint", default=str(RESULTADOS / "dqn_acrobot_1200ep_state_dict.pt"))
    ra.add_argument("--solo-aleatoria", action="store_true", help="omite la política entrenada")
    ra.add_argument("--save", help="guardar las mediciones en un JSON")
    ra.set_defaults(func=cmd_rangos)

    t = sub.add_parser("train", help="entrena el agente")
    t.add_argument("--mode", choices=["smoke", "full"], default="smoke",
                   help="smoke = 10 episodios para probar el pipeline (por defecto)")
    t.add_argument("--episodes", type=int, default=1200, help="episodios en modo full")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--output", default="artifacts", help="carpeta para los artefactos")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("evaluate", help="evalúa una política greedy")
    e.add_argument("--checkpoint", default=str(RESULTADOS / "dqn_acrobot_1200ep_state_dict.pt"))
    e.add_argument("--episodes", type=int, default=100)
    e.add_argument("--seed-start", type=int, default=10_000,
                   help="disjuntas de las de entrenamiento (42-1241)")
    e.add_argument("--save", help="guardar las métricas en un JSON")
    e.set_defaults(func=cmd_evaluate)

    g = sub.add_parser("plot", help="regenera la curva de aprendizaje")
    g.add_argument("--history", default=str(RESULTADOS / "training_history_resumen.json"))
    g.add_argument("--output", default="curva.png")
    g.add_argument("--window", type=int, default=50)
    g.set_defaults(func=cmd_plot)
    return p


def main() -> None:
    args = _parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
