"""Regenera la figura de §10.2 a partir de los artefactos de `resultados/`."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SUP, TINTA, TINTA2 = "#fcfcfb", "#0b0b0b", "#52514e"
AZUL, NARANJA, GRIS = "#2a78d6", "#eb6834", "#b9b7b0"
SEMILLAS = (42, 43, 44)


def _cargar(res: Path):
    correcto, ablacion, term_c, term_a = {}, {}, {}, {}
    d = json.loads((res / "evaluation_metrics_100ep.json").read_text(encoding="utf-8"))
    correcto[42], term_c[42] = d["returns"], d["terminal_rate"]
    for s in (43, 44):
        d = json.loads((res / f"semillas/resultado_seed_{s}.json").read_text(encoding="utf-8"))
        correcto[s] = d["retornos_evaluacion"]
        term_c[s] = d["evaluacion_greedy"]["tasa_terminacion"]
    for s in SEMILLAS:
        d = json.loads((res / f"ablacion/evaluacion_seed_{s}.json").read_text(encoding="utf-8"))
        ablacion[s], term_a[s] = d["returns"], d["terminal_rate"]
    return correcto, ablacion, term_c, term_a


def dibujar(resultados="resultados", salida="resultados/ablacion_truncacion.png") -> Path:
    res = Path(resultados)
    correcto, ablacion, term_c, term_a = _cargar(res)

    fig, ax = plt.subplots(figsize=(10.2, 5.4), dpi=180)
    fig.patch.set_facecolor(SUP)
    ax.set_facecolor(SUP)
    ax.axhline(-100, color=TINTA2, lw=1.1, ls=(0, (5, 4)), alpha=0.8, zorder=2)

    rng = np.random.default_rng(0)
    for i, s in enumerate(SEMILLAS):
        for j, (datos, col, term) in enumerate(((correcto[s], AZUL, term_c[s]),
                                                (ablacion[s], NARANJA, term_a[s]))):
            x = i + (j - 0.5) * 0.42
            y = np.asarray(datos, float)
            ax.scatter(x + rng.normal(0, 0.045, len(y)), y, s=11, color=col, alpha=0.30,
                       linewidths=0, zorder=3)
            ax.boxplot([y], positions=[x], widths=0.28, patch_artist=True, showfliers=False,
                       medianprops=dict(color=col, lw=2.0),
                       boxprops=dict(facecolor=SUP, edgecolor=col, lw=1.6),
                       whiskerprops=dict(color=col, lw=1.2),
                       capprops=dict(color=col, lw=1.2), zorder=4)
            ax.text(x, -47, f"{int(term * 100)}/100", ha="center", va="top",
                    color=col, fontsize=9.5, fontweight="600")

    ax.text(-0.58, -47, "alcanzan\nla meta", ha="center", va="top", color=TINTA2, fontsize=8.5)
    ax.set_xticks(range(len(SEMILLAS)))
    ax.set_xticklabels([f"semilla {s}" for s in SEMILLAS], color=TINTA2, fontsize=10)
    ax.set_xlim(-1.15, len(SEMILLAS) - 0.55)
    ax.set_ylim(-545, -28)
    ax.set_yticks([-100, -200, -300, -400, -500])
    ax.set_yticklabels(["−100", "−200", "−300", "−400", "−500"])
    ax.text(-1.12, -100, "umbral −100", color=TINTA2, fontsize=9, va="bottom", ha="left")
    ax.set_ylabel("retorno de los 100 episodios de evaluación", color=TINTA2, fontsize=10)
    ax.set_title("Dos de tres semillas no se distinguen; la tercera no supera el umbral",
                 color=TINTA, fontsize=13, fontweight="600", pad=14, loc="left")
    ax.grid(axis="y", color=GRIS, lw=0.6, alpha=0.45)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(GRIS)
        ax.spines[lado].set_linewidth(0.8)
    ax.tick_params(colors=TINTA2, labelsize=9.5, length=3)
    ax.tick_params(axis="x", length=0, pad=8)

    manijas = [plt.Line2D([], [], color=AZUL, lw=2.4, label="bootstrap solo en terminación real"),
               plt.Line2D([], [], color=NARANJA, lw=2.4, label="truncación tratada como terminación")]
    leg = fig.legend(handles=manijas, loc="lower center", ncol=2, frameon=False, fontsize=9.5,
                     bbox_to_anchor=(0.5, 0.005))
    for t in leg.get_texts():
        t.set_color(TINTA2)
    fig.subplots_adjust(left=0.105, right=0.975, top=0.885, bottom=0.165)
    fig.savefig(salida, facecolor=SUP)
    plt.close(fig)
    return Path(salida)
