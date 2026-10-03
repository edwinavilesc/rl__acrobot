"""Regenera la figura de §10.4 a partir de los artefactos de `resultados/`.

Dos paneles:

  izquierda  el retorno de evaluacion del DQN y del tabular, tres semillas cada uno
  derecha    las doce configuraciones del barrido, para que se vea que el tabular
             tuvo una oportunidad honesta y ninguna llego al umbral
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend fijado arriba)

SUP, TINTA, TINTA2 = "#fcfcfb", "#0b0b0b", "#52514e"
AZUL, VERDE, GRIS = "#2a78d6", "#008300", "#b9b7b0"
SEMILLAS = (42, 43, 44)
UMBRAL = -100.0


def _cargar(res: Path):
    """Retornos de evaluacion por metodo y semilla, mas el barrido y el resumen."""
    dqn = {42: json.loads((res / "evaluation_metrics_100ep.json").read_text(encoding="utf-8"))["returns"]}
    for s in (43, 44):
        d = json.loads((res / f"semillas/resultado_seed_{s}.json").read_text(encoding="utf-8"))
        dqn[s] = d["retornos_evaluacion"]
    tab = {s: json.loads((res / f"tabular/evaluacion_seed_{s}.json").read_text(encoding="utf-8"))["returns"]
           for s in SEMILLAS}
    barrido = json.loads((res / "tabular/barrido_hiperparametros.json").read_text(encoding="utf-8"))
    resumen = json.loads((res / "tabular/RESUMEN_TABULAR.json").read_text(encoding="utf-8"))
    return dqn, tab, barrido, resumen


def _panel_retornos(ax, dqn, tab):
    """Caja y puntos del retorno de evaluacion, un par por semilla."""
    rng = np.random.default_rng(0)
    for i, s in enumerate(SEMILLAS):
        for j, (datos, col) in enumerate(((dqn[s], AZUL), (tab[s], VERDE))):
            x = i + (j - 0.5) * 0.42
            y = np.asarray(datos, float)
            ax.scatter(x + rng.normal(0, 0.045, len(y)), y, s=11, color=col, alpha=0.28,
                       linewidths=0, zorder=3)
            ax.boxplot([y], positions=[x], widths=0.28, patch_artist=True, showfliers=False,
                       medianprops=dict(color=col, lw=2.0),
                       boxprops=dict(facecolor=SUP, edgecolor=col, lw=1.6),
                       whiskerprops=dict(color=col, lw=1.2),
                       capprops=dict(color=col, lw=1.2), zorder=4)
            ax.text(x, -22, f"{np.mean(y):.0f}".replace("-", "−"), ha="center", va="top",
                    color=col, fontsize=9.5, fontweight="600")

    ax.axhline(UMBRAL, color=TINTA2, lw=1.1, ls=(0, (5, 4)), alpha=0.8, zorder=2)
    ax.set_xticks(range(len(SEMILLAS)))
    ax.set_xticklabels([f"semilla {s}" for s in SEMILLAS], color=TINTA2, fontsize=9.5)
    ax.set_xlim(-0.75, len(SEMILLAS) - 0.35)
    ax.set_ylim(-510, -10)
    ax.set_ylabel("retorno de los 100 episodios", color=TINTA2, fontsize=9.5)
    ax.text(-0.72, UMBRAL + 8, " umbral −100", color=TINTA2, fontsize=8.5, va="bottom")
    ax.set_title("Desempeño con el mismo número de episodios", color=TINTA, fontsize=11,
                 fontweight="600", pad=24, loc="left")


def _panel_barrido(ax, barrido, resumen):
    """Las doce configuraciones probadas, ordenadas por resultado."""
    orden = sorted(barrido, key=lambda r: r["eval_retorno"])
    y = np.arange(len(orden))
    mejor = max(barrido, key=lambda r: r["eval_retorno"])

    ax.axvline(UMBRAL, color=TINTA2, lw=1.1, ls=(0, (5, 4)), alpha=0.8, zorder=2)
    dqn = resumen["dqn_mismas_semillas"]["retorno_medio"]
    ax.axvline(dqn, color=AZUL, lw=1.6, zorder=2)
    ax.text(dqn - 6, 0.2, "DQN", color=AZUL, fontsize=9.5, fontweight="600",
            ha="right", va="bottom")
    ax.text(UMBRAL - 6, 1.4, "umbral", color=TINTA2, fontsize=8.5, ha="right", va="bottom")

    for k, fila in enumerate(orden):
        es_mejor = fila is mejor or (fila["bins_angulo"], fila["bins_velocidad"], fila["alpha"]) == \
            (mejor["bins_angulo"], mejor["bins_velocidad"], mejor["alpha"])
        ax.plot([-500, fila["eval_retorno"]], [k, k], color=VERDE, lw=1.2,
                alpha=0.5 if not es_mejor else 0.9, zorder=3)
        ax.scatter([fila["eval_retorno"]], [k], s=60 if es_mejor else 34, color=VERDE,
                   zorder=4, edgecolor=SUP if es_mejor else "none", linewidth=1.4)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{f['bins_angulo']}×{f['bins_velocidad']} · α={f['alpha']}"
                        for f in orden], color=TINTA2, fontsize=8.5)
    ax.set_ylim(-0.7, len(orden) - 0.3)
    ax.set_xlim(-500, -30)
    ax.set_xlabel("retorno de evaluación", color=TINTA2, fontsize=9.5)
    ax.text(-495, len(orden) - 0.4, "rejilla · tasa de aprendizaje", color=TINTA2,
            fontsize=8.5, va="top", style="italic")
    ax.set_title("Las doce configuraciones que probamos", color=TINTA, fontsize=11,
                 fontweight="600", pad=24, loc="left")


def dibujar(resultados="resultados", salida="resultados/tabular.png") -> Path:
    """Escribe la figura de §10.4 y devuelve su ruta."""
    res = Path(resultados)
    dqn, tab, barrido, resumen = _cargar(res)

    fig, (izq, der) = plt.subplots(1, 2, figsize=(12.4, 5.2), dpi=180,
                                   gridspec_kw={"width_ratios": [1, 1.12]})
    fig.patch.set_facecolor(SUP)
    for ax in (izq, der):
        ax.set_facecolor(SUP)
        ax.grid(axis="y" if ax is izq else "x", color=GRIS, lw=0.6, alpha=0.45)
        ax.set_axisbelow(True)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
        for lado in ("left", "bottom"):
            ax.spines[lado].set_color(GRIS)
            ax.spines[lado].set_linewidth(0.8)
        ax.tick_params(colors=TINTA2, labelsize=9, length=3)

    _panel_retornos(izq, dqn, tab)
    _panel_barrido(der, barrido, resumen)
    izq.tick_params(axis="x", length=0, pad=7)
    der.tick_params(axis="y", length=0, pad=4)

    manijas = [plt.Line2D([], [], color=AZUL, lw=2.6, label="DQN, el agente del entregable"),
               plt.Line2D([], [], color=VERDE, lw=2.6, label="Q-learning tabular")]
    leg = fig.legend(handles=manijas, loc="lower center", ncol=2, frameon=False, fontsize=9.5,
                     bbox_to_anchor=(0.5, 0.005))
    for t in leg.get_texts():
        t.set_color(TINTA2)

    fig.suptitle("El tabular alcanza la meta siempre, y aun así tarda dos veces y media más",
                 color=TINTA, fontsize=12.5, fontweight="600", x=0.011, ha="left", y=0.975)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.80, bottom=0.155, wspace=0.30)
    fig.savefig(salida, facecolor=SUP)
    plt.close(fig)
    return Path(salida)
