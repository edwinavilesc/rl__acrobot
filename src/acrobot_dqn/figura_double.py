"""Regenera la figura de §10.3 a partir de los artefactos de `resultados/`.

Dos paneles sobre el mismo eje horizontal de semillas:

  izquierda  el retorno de los 100 episodios de evaluacion, para los dos metodos
  derecha    el sesgo -valor predicho menos retorno real- por tramo de pasos que
             le faltan al agente para llegar a la meta

El panel derecho se dibuja con barras sobre un eje proporcional y no con una linea,
porque los tramos tienen anchos muy distintos (20, 30, 30, 10, 10 pasos) y numeros de
muestras que van de 9.000 a 206: unirlos con una linea sugiere una tendencia continua
que la medicion no sostiene. Por la misma razon se sombrea el tramo en el que la cota
aritmetica del retorno real fuerza el signo positivo.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend fijado arriba)

SUP, TINTA, TINTA2 = "#fcfcfb", "#0b0b0b", "#52514e"
AZUL, VIOLETA, GRIS = "#2a78d6", "#4a3aa7", "#b9b7b0"
SEMILLAS = (42, 43, 44)


def _cargar(res: Path):
    """Devuelve, por metodo y semilla, los retornos de evaluacion y el sesgo."""
    datos = {"dqn": {}, "double": {}}

    d = json.loads((res / "evaluation_metrics_100ep.json").read_text(encoding="utf-8"))
    datos["dqn"][42] = {"retornos": d["returns"]}
    for s in (43, 44):
        d = json.loads((res / f"semillas/resultado_seed_{s}.json").read_text(encoding="utf-8"))
        datos["dqn"][s] = {"retornos": d["retornos_evaluacion"]}
    for s in SEMILLAS:
        d = json.loads((res / f"double_dqn/evaluacion_seed_{s}.json").read_text(encoding="utf-8"))
        datos["double"][s] = {"retornos": d["retornos"]}

    sesgo = json.loads((res / "double_dqn/medicion_sesgo.json").read_text(encoding="utf-8"))
    return datos, sesgo


def _panel_retornos(ax, datos):
    """Caja y puntos del retorno de evaluacion, un par de cajas por semilla."""
    rng = np.random.default_rng(0)
    for i, s in enumerate(SEMILLAS):
        for j, (clave, col) in enumerate((("dqn", AZUL), ("double", VIOLETA))):
            x = i + (j - 0.5) * 0.42
            y = np.asarray(datos[clave][s]["retornos"], float)
            ax.scatter(x + rng.normal(0, 0.045, len(y)), y, s=11, color=col, alpha=0.30,
                       linewidths=0, zorder=3)
            ax.boxplot([y], positions=[x], widths=0.28, patch_artist=True, showfliers=False,
                       medianprops=dict(color=col, lw=2.0),
                       boxprops=dict(facecolor=SUP, edgecolor=col, lw=1.6),
                       whiskerprops=dict(color=col, lw=1.2),
                       capprops=dict(color=col, lw=1.2), zorder=4)
            ax.text(x, -46, f"{np.mean(y):.2f}".replace("-", "−"), ha="center", va="top",
                    color=col, fontsize=9, fontweight="600")

    ax.axhline(-100, color=TINTA2, lw=1.1, ls=(0, (5, 4)), alpha=0.8, zorder=2)
    ax.set_xticks(range(len(SEMILLAS)))
    ax.set_xticklabels([f"semilla {s}" for s in SEMILLAS], color=TINTA2, fontsize=9.5)
    ax.set_xlim(-0.75, len(SEMILLAS) - 0.35)
    ax.set_ylim(-158, -40)   # abarca el peor episodio real (-150) sin recortar nada
    ax.set_ylabel("retorno de los 100 episodios", color=TINTA2, fontsize=9.5)
    ax.text(-0.72, -100, " umbral −100", color=TINTA2, fontsize=8.5, va="bottom")
    ax.set_title("Desempeño", color=TINTA, fontsize=11, fontweight="600", pad=26, loc="left")


# limites de cada tramo en pasos; el ultimo es abierto y se dibuja con ancho fijo
TRAMOS = (("0-19", 0, 20), ("20-49", 20, 50), ("50-79", 50, 80),
          ("80-89", 80, 90), ("90-99", 90, 100), ("100+", 100, 122))
# desde este tramo la cota aritmetica del retorno real esta al nivel del suelo que
# predice la red, asi que un sesgo positivo no distingue sobreestimacion de aritmetica
DESDE_FORZADO = 80


def _panel_sesgo(ax, resumen_sesgo):
    """Sesgo medio por tramo de pasos restantes, con el tamano de cada tramo a escala."""
    ax.axvspan(DESDE_FORZADO, TRAMOS[-1][2], color=GRIS, alpha=0.22, zorder=1, lw=0)
    ax.axhline(0, color=TINTA2, lw=1.1, ls=(0, (5, 4)), alpha=0.8, zorder=2)

    for j, (clave, col, nombre) in enumerate((("dqn", AZUL, "DQN"),
                                              ("double_dqn", VIOLETA, "Double DQN"))):
        por_tramo = resumen_sesgo["agregado"][clave]["por_tramo"]
        for etq, lo, hi in TRAMOS:
            d = por_tramo.get(etq)
            if d is None:
                continue
            ancho = (hi - lo) * 0.40
            ax.bar(lo + (hi - lo) / 2 + (j - 0.5) * ancho, d["sesgo_medio"], width=ancho,
                   color=col, alpha=0.85, zorder=3,
                   label=nombre if etq == "0-19" else None)

    # el numero de muestras de cada tramo, que es lo que la linea escondia
    for etq, lo, hi in TRAMOS:
        ns = [resumen_sesgo["agregado"][c]["por_tramo"][etq]["n_total"]
              for c in ("dqn", "double_dqn") if etq in resumen_sesgo["agregado"][c]["por_tramo"]]
        ax.text(lo + (hi - lo) / 2, -12.6, "n " + " · ".join(f"{v:,}".replace(",", ".")
                                                            for v in ns),
                ha="left", va="bottom", color=TINTA2, fontsize=7.4, rotation=90)
        ax.axvline(hi, color=GRIS, lw=0.7, alpha=0.6, zorder=1)

    ax.annotate("en la zona sombreada la cota aritmética del retorno real está al\n"
                "nivel del suelo que predice la red, así que el signo positivo\n"
                "no distingue sobreestimación de aritmética",
                xy=(0.015, 0.97), xycoords="axes fraction", ha="left", va="top",
                color=TINTA2, fontsize=8.2, style="italic")
    ax.set_xticks([lo for _, lo, _ in TRAMOS] + [100])
    ax.set_xticklabels([str(lo) for _, lo, _ in TRAMOS] + ["100+"], color=TINTA2, fontsize=9)
    ax.set_xlim(-2, TRAMOS[-1][2])
    ax.set_ylim(-13, 14)
    ax.set_xlabel("pasos que le faltan al agente para llegar a la meta (tramos a escala)",
                  color=TINTA2, fontsize=9.5)
    ax.set_ylabel("sesgo: valor predicho − retorno real", color=TINTA2, fontsize=9.5)
    ax.set_title("El sesgo, tramo por tramo y con su muestra",
                 color=TINTA, fontsize=11, fontweight="600", pad=26, loc="left")


def dibujar(resultados="resultados", salida="resultados/double_dqn.png") -> Path:
    """Escribe la figura de §10.3 y devuelve su ruta."""
    res = Path(resultados)
    datos, sesgo = _cargar(res)

    fig, (izq, der) = plt.subplots(1, 2, figsize=(12.4, 5.0), dpi=180,
                                   gridspec_kw={"width_ratios": [1.05, 1]})
    fig.patch.set_facecolor(SUP)
    for ax in (izq, der):
        ax.set_facecolor(SUP)
        ax.grid(axis="y", color=GRIS, lw=0.6, alpha=0.45)
        ax.set_axisbelow(True)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
        for lado in ("left", "bottom"):
            ax.spines[lado].set_color(GRIS)
            ax.spines[lado].set_linewidth(0.8)
        ax.tick_params(colors=TINTA2, labelsize=9, length=3)

    _panel_retornos(izq, datos)
    _panel_sesgo(der, sesgo)
    izq.tick_params(axis="x", length=0, pad=7)
    der.tick_params(axis="x", length=0, pad=6)

    manijas = [plt.Line2D([], [], color=AZUL, lw=2.6, label="DQN, el agente del entregable"),
               plt.Line2D([], [], color=VIOLETA, lw=2.6, label="Double DQN")]
    leg = fig.legend(handles=manijas, loc="lower center", ncol=2, frameon=False, fontsize=9.5,
                     bbox_to_anchor=(0.5, 0.005))
    for t in leg.get_texts():
        t.set_color(TINTA2)

    fig.suptitle("Double DQN rinde algo mejor; el sesgo de los valores no es un número, tiene estructura",
                 color=TINTA, fontsize=12.5, fontweight="600", x=0.011, ha="left", y=0.975)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.80, bottom=0.165, wspace=0.28)
    fig.savefig(salida, facecolor=SUP)
    plt.close(fig)
    return Path(salida)
