"""Cuenta en cuantas actualizaciones entra el error de la ablacion de §10.2.

La primera cuenta que hicimos para explicar la ablacion fue la fraccion de las
transiciones recogidas que quedan corruptas: 41 de 154.568, un 0,027 %. Esa cuenta es
correcta y mide lo que no importa. (Nota sobre el nombre: no es una fraccion del
buffer, que solo guarda 100.000 transiciones a la vez; es de todo lo recogido.)

Lo que decide el dano de una transicion corrupta es en cuantos minilotes entra, no
que porcion de la experiencia ocupa. Este modulo recorre la corrida paso a paso y, en cada
actualizacion, calcula la probabilidad de que el lote de 64 tomado al azar del buffer
vivo contenga al menos una transicion contaminada:

    P(al menos una) = 1 - (1 - contaminadas_vivas / tamano_del_buffer) ** 64

La suma de esas probabilidades a lo largo de la corrida es el numero esperado de
minilotes afectados. Las transiciones salen del buffer cuando el buffer circular de
100.000 da la vuelta, asi que solo se cuentan las que siguen vivas en cada paso.

Se ejecuta con `acrobot minilotes` y es la fuente del 3,8 % que cita §10.2.
"""
from __future__ import annotations

import json
from pathlib import Path

CAPACIDAD = 100_000
TAMANO_LOTE = 64
INICIO_APRENDIZAJE = 1_000


def contar(historia: Path | str = "resultados/ablacion/ablacion_seed_42.json") -> dict:
    """Devuelve las transiciones contaminadas y los minilotes que llegan a contenerlas."""
    h = json.loads(Path(historia).read_text(encoding="utf-8"))
    paso, lotes, esperanza = 0, 0, 0.0
    contaminadas: list[int] = []
    por_paso: list[float] = []   # para ver si la tasa es pareja o se concentra

    for largo, truncado in zip(h["lengths"], h["truncated"], strict=True):
        for k in range(largo):
            paso += 1
            if paso >= max(TAMANO_LOTE, INICIO_APRENDIZAJE):
                lotes += 1
                vivas = sum(1 for p in contaminadas if paso - p < CAPACIDAD)
                p_lote = 0.0
                if vivas:
                    ocupacion = min(paso, CAPACIDAD)
                    p_lote = 1 - (1 - vivas / ocupacion) ** TAMANO_LOTE
                    esperanza += p_lote
                por_paso.append(p_lote)
            # solo la ultima transicion de un episodio truncado queda mal etiquetada
            if k == largo - 1 and truncado:
                contaminadas.append(paso)

    pasos = sum(h["lengths"])
    return {
        "archivo": str(historia),
        "pasos_de_entorno": pasos,
        "transiciones_contaminadas": len(contaminadas),
        "fraccion_de_lo_recogido": len(contaminadas) / pasos,
        "minilotes": lotes,
        "minilotes_afectados": esperanza,
        "fraccion_de_minilotes": esperanza / lotes,
        "tasa_por_sexto": [sum(por_paso[i * lotes // 6:(i + 1) * lotes // 6])
                           / len(por_paso[i * lotes // 6:(i + 1) * lotes // 6])
                           for i in range(6)],
    }


def informe(historia: Path | str = "resultados/ablacion/ablacion_seed_42.json") -> dict:
    """Imprime las dos cifras enfrentadas, el reparto por sextos, y devuelve el conteo."""
    d = contar(historia)
    print(f"transiciones contaminadas: {d['transiciones_contaminadas']} de "
          f"{d['pasos_de_entorno']:,} recogidas = {d['fraccion_de_lo_recogido']:.3%}")
    print(f"minilotes que llegan a contener alguna: {d['minilotes_afectados']:,.0f} de "
          f"{d['minilotes']:,} = {d['fraccion_de_minilotes']:.1%}")
    factor = d["fraccion_de_minilotes"] / d["fraccion_de_lo_recogido"]
    print(f"la segunda cifra es {factor:.0f} veces la primera\n")
    print("y ese 3,8 % no esta repartido parejo (tasa local por sextos del entrenamiento):")
    for i, tasa in enumerate(d["tasa_por_sexto"], start=1):
        print(f"  sexto {i}: {tasa:.2%}")
    return d
