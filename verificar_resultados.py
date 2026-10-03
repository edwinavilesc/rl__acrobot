#!/usr/bin/env python3
"""Comprobacion de coherencia entre el README y los archivos de resultados.

Recorre las cifras que aparecen en el README y las compara una por una con
el archivo de resultados del que salen. No necesita instalar nada: solo la
biblioteca estandar de Python 3.

Uso:
    python3 verificar_resultados.py           (desde la carpeta del repositorio)
    python3 verificar_resultados.py /ruta/al/repo
"""
import gzip
import hashlib
import json
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()


def _cargar_por_ruta(nombre: str, ruta: Path):
    """Importa un modulo suelto sin pasar por el paquete, que exigiria torch."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo
SHA_CODIGO = "1fd13fa54e17f43e59ee4603ea70e327fc8089f33502d3abaaea5e5d33bd47e8"

ok_total = fallos = 0
def check(nombre, ok, detalle=""):
    global ok_total, fallos
    if ok:
        ok_total += 1
    else:
        fallos += 1
    print(f"  [{'OK ' if ok else '!! '}] {nombre}" + (f"   → {detalle}" if detalle else ""))

def titulo(t):
    print(f"\n{'─'*72}\n{t}\n{'─'*72}")

print(f"\n{'='*72}\nVERIFICACION DE LOS RESULTADOS — U3 Taller 2 · DQN sobre Acrobot-v1\n{'='*72}")
print(f"Carpeta: {RAIZ}")

# ───────────────────────────────────────────────── 1. archivos que deben estar
titulo("1. ¿Estan todos los archivos?")
ESPERADOS = [
    "README.md", "requirements.txt", ".gitignore",
    "src/acrobot_dqn/dqn_acrobot_v3.py",
    "src/acrobot_dqn/__init__.py",
    "src/acrobot_dqn/cli.py",
    "pyproject.toml",
    "notebooks/SAR_U3_Taller2_DQN_EJECUTADO.ipynb",
    "docs/GUIA_DE_LECTURA.md",
    "resultados/full_summary.json",
    "resultados/evaluation_metrics_100ep.json",
    "resultados/training_history_resumen.json",
    "resultados/training_history_1200ep.json.gz",
    "resultados/dqn_learning_curve_1200ep.png",
    "resultados/dqn_5_semillas.png",
    "resultados/dqn_acrobot_1200ep_state_dict.pt",
    "resultados/semillas/RESUMEN_5_SEMILLAS.json",
]
for rel in ESPERADOS:
    p = RAIZ / rel
    check(rel, p.exists(), f"{p.stat().st_size/1024:.0f} KB" if p.exists() else "NO ESTA")
for s in (43, 44, 45, 46):
    p = RAIZ / f"resultados/semillas/resultado_seed_{s}.json"
    check(f"resultados/semillas/resultado_seed_{s}.json", p.exists(),
          f"{p.stat().st_size/1024:.0f} KB" if p.exists() else "NO ESTA")

if not (RAIZ / "resultados/full_summary.json").exists():
    print("\n!! Faltan artefactos basicos: el repositorio esta incompleto.\n")
    sys.exit(1)

# ─────────────────────────────────────── 2. el codigo fuente no fue modificado
titulo("2. ¿El codigo fuente sigue intacto?")
src = RAIZ / "src/acrobot_dqn/dqn_acrobot_v3.py"
sha = hashlib.sha256(src.read_bytes()).hexdigest()
check("SHA-256 de dqn_acrobot_v3.py", sha == SHA_CODIGO,
      sha if sha == SHA_CODIGO else f"CAMBIO\n        esperado {SHA_CODIGO}\n        obtenido {sha}")

# ───────────────────────────────────────────────── 3. cifras contra artefactos
titulo("3. ¿Las cifras del README coinciden con los artefactos?")
readme = (RAIZ / "README.md").read_text(encoding="utf-8")
resumen = json.loads((RAIZ / "resultados/full_summary.json").read_text())
ev = json.loads((RAIZ / "resultados/evaluation_metrics_100ep.json").read_text())["returns"]
hist = json.loads((RAIZ / "resultados/training_history_resumen.json").read_text())
r = hist["returns"]
sem = json.loads((RAIZ / "resultados/semillas/RESUMEN_5_SEMILLAS.json").read_text())

def en_readme(txt):
    return txt in readme

media_ev = sum(ev) / len(ev)
sd_ev = st.stdev(ev)
u100 = r[-100:]

PRUEBAS = [
    ("retorno medio de evaluacion = -80,96", abs(media_ev + 80.96) < .005 and en_readme("−80,96"), f"{media_ev:.2f}"),
    ("100 episodios de evaluacion", len(ev) == 100, f"{len(ev)}"),
    ("100/100 terminaciones", resumen["evaluacion_greedy"]["tasa_terminacion_verdadera"] == 1.0, "1.00"),
    ("95 de 100 mejores que -100", sum(1 for x in ev if x > -100) == 95 and en_readme("95 / 100"), f"{sum(1 for x in ev if x > -100)}"),
    ("sd de evaluacion = 12,52", abs(sd_ev - 12.518) < .01, f"{sd_ev:.2f}"),
    ("1.200 episodios de entrenamiento", len(r) == 1200, f"{len(r)}"),
    ("media ultimos 100 = -85,68", abs(sum(u100)/100 + 85.68) < .005 and en_readme("−85,68"), f"{sum(u100)/100:.2f}"),
    ("mejor episodio = -60", max(r) == -60 and en_readme("−60"), f"{max(r):.0f}"),
    ("el mejor fue el episodio 760", r.index(max(r)) + 1 == 760 and en_readme("760"), f"{r.index(max(r))+1}"),
    ("54 episodios truncados", sum(hist["truncated"]) == 54, f"{sum(hist['truncated'])}"),
    ("ultimo truncado = episodio 350", max(i for i, t in enumerate(hist["truncated"]) if t) + 1 == 350 and en_readme("350"),
     f"{max(i for i,t in enumerate(hist['truncated']) if t)+1}"),
    ("158.451 pasos de entorno", sum(hist["lengths"]) == 158451 and en_readme("158.451"), f"{sum(hist['lengths'])}"),
    ("clipping en 6,00% de los pasos", abs(hist["grad_norms_resumen"]["pct_recortados"] - 6.0) < .02 and en_readme("6,00 %"),
     f"{hist['grad_norms_resumen']['pct_recortados']:.2f}%"),
    ("la loss SUBE de 0,0062 a 0,4807", hist["losses_resumen"]["por_sextos"][0] < hist["losses_resumen"]["por_sextos"][-1],
     f"{hist['losses_resumen']['por_sextos'][0]:.4f} → {hist['losses_resumen']['por_sextos'][-1]:.4f}"),
]
for n, ok, d in PRUEBAS:
    check(n, ok, d)

# ────────────────────────────────────────────────────── 4. las cinco semillas
titulo("4. ¿Las cinco semillas sostienen la conclusion?")
medias = [s["eval_medio"] for s in sem["semillas"]]
a = sem["agregado"]
SEM = [
    ("son 5 semillas", len(sem["semillas"]) == 5, f"{[s['seed'] for s in sem['semillas']]}"),
    ("las 5 superan el umbral de -100", all(m > -100 for m in medias), f"peor {min(medias):.2f}"),
    ("500/500 terminaciones verdaderas", a["terminaciones"] == 500, f"{a['terminaciones']}"),
    ("media de las medias = -79,90", abs(a["media_de_medias"] + 79.90) < .005 and en_readme("−79,90"), f"{a['media_de_medias']:.2f}"),
    ("sd ENTRE semillas = 1,82", abs(a["sd_entre_semillas"] - 1.82) < .005 and en_readme("1,82"), f"{a['sd_entre_semillas']:.2f}"),
    ("sd DENTRO de cada semilla = 15,47", abs(a["sd_dentro_promedio"] - 15.47) < .005 and en_readme("15,47"), f"{a['sd_dentro_promedio']:.2f}"),
    ("la variabilidad entre semillas es la menor", a["sd_entre_semillas"] < a["sd_dentro_promedio"],
     f"{a['sd_dentro_promedio']/a['sd_entre_semillas']:.1f} veces menor"),
    ("470 de 500 mejores que -100", a["mejores_que_-100"] == 470 and en_readme("470 / 500"), f"{a['mejores_que_-100']}"),
]
for n, ok, d in SEM:
    check(n, ok, d)

# ──────────────────────────────────────────────────── 5. el notebook ejecutado
titulo("5. ¿El notebook esta ejecutado de verdad?")
nb = json.loads((RAIZ / "notebooks/SAR_U3_Taller2_DQN_EJECUTADO.ipynb").read_text(encoding="utf-8"))
cod = [c for c in nb["cells"] if c["cell_type"] == "code"]
con_out = sum(1 for c in cod if c.get("outputs"))
errores = sum(1 for c in cod if any(o.get("output_type") == "error" for o in c.get("outputs", [])))
ejec = sum(1 for c in cod if c.get("execution_count") is not None)
fuente = "\n".join("".join(c["source"]) for c in nb["cells"])
salidas = json.dumps([c.get("outputs", []) for c in cod])
modo = re.search(r'RUN_MODE\s*=\s*"(\w+)"', fuente)
NB = [
    ("todas las celdas de codigo se ejecutaron", ejec == len(cod), f"{ejec}/{len(cod)}"),
    ("ninguna celda dio error", errores == 0, f"{errores} errores"),
    ("hay celdas con salidas guardadas", con_out > 0, f"{con_out} celdas con output"),
    ('esta en modo "full", no "smoke"', bool(modo) and modo.group(1) == "full",
     f'RUN_MODE = "{modo.group(1)}"' if modo else "no se encontro RUN_MODE"),
    ("entreno 1.200 episodios, no 10", "1200" in salidas, "visible en las salidas"),
    ("el texto no quedo con marcadores de trabajo", "PENDIENTE" not in fuente, "ninguno"),
]
for n, ok, d in NB:
    check(n, ok, d)

# ─────────────────────────────────────────── 6. lo que NO se debe afirmar
titulo("6. ¿El README evita afirmaciones que la evidencia no sostiene?")
PROHIBIDAS = ["nivel humano", "convergió", "es óptimo", "resultado óptimo", "totalmente robusto"]
for p in PROHIBIDAS:
    check(f'no dice "{p}"', p.lower() not in readme.lower())
check("declara terminated vs truncated", "truncated" in readme and "terminated" in readme)
check("cita el umbral oficial de -100", "−100" in readme or "-100" in readme)
check("cita la politica aleatoria de -500", "−500" in readme or "-500" in readme)

# ────────────────────────────────────────── 7. la historia cruda se puede abrir
titulo("7. ¿La evidencia cruda esta integra?")
try:
    with gzip.open(RAIZ / "resultados/training_history_1200ep.json.gz", "rt", encoding="utf-8") as f:
        crudo = json.load(f)
    check("el .gz se descomprime y es JSON valido", True,
          f"{len(crudo['losses'])} losses, {len(crudo['grad_norms'])} grad norms")
    check("los retornos crudos coinciden con el resumen", crudo["returns"] == r, "identicos")
except Exception as e:
    check("el .gz se descomprime", False, f"{type(e).__name__}: {e}")

titulo("8. El notebook y los artefactos cuentan lo mismo")
try:
    nbj = json.loads((RAIZ / "notebooks/SAR_U3_Taller2_DQN_EJECUTADO.ipynb").read_text(encoding="utf-8"))
    salidas = []
    for celda in nbj["cells"]:
        for o in celda.get("outputs", []):
            salidas.append("".join(o.get("text", [])))
    salidas = "\n".join(salidas)

    ev = json.loads((RAIZ / "resultados/evaluation_metrics_100ep.json").read_text(encoding="utf-8"))
    hist = json.loads((RAIZ / "resultados/training_history_resumen.json").read_text(encoding="utf-8"))

    for etiqueta, valor in [
        ("retorno medio de evaluacion", f'"mean_return": {ev["mean_return"]}'),
        ("SD de evaluacion", f'"sd_return": {ev["sd_return"]}'),
        ("pasos medios de evaluacion", f'"mean_steps": {ev["mean_steps"]}'),
        ("media de los ultimos 100 de entrenamiento",
         f'"train_mean_last_n": {round(sum(hist["returns"][-100:]) / 100, 8)}'),
    ]:
        check(f"el notebook imprime el mismo {etiqueta}", valor in salidas, valor.split(": ")[1])

    n_ep = len(hist["returns"])
    check("el notebook corrio en modo full", '"mode": "full"' in salidas, "full")
    check("el notebook corrio los mismos episodios", f'"train_episodes": {n_ep}' in salidas, f"{n_ep}")
except Exception as e:
    check("el notebook se puede contrastar", False, f"{type(e).__name__}: {e}")

titulo("9. Los rangos de observacion del README salen del artefacto")
try:
    rg = json.loads((RAIZ / "resultados/rangos_observacion.json").read_text(encoding="utf-8"))
    readme = (RAIZ / "README.md").read_text(encoding="utf-8")
    for clave in ("omega1", "omega2"):
        for pol in ("aleatoria", "entrenada"):
            lo, hi = rg[clave][pol]
            texto = f"[{lo:+.3f}, {hi:+.3f}]".replace("-", "\u2212").replace(".", ",")
            check(f"{clave} con politica {pol} coincide", texto in readme, texto)
except Exception as e:
    check("los rangos de observacion se pueden contrastar", False, f"{type(e).__name__}: {e}")

titulo("10. La ablacion de §10.2 dice lo que el README afirma")
try:
    resumen = json.loads((RAIZ / "resultados/ablacion/RESUMEN_ABLACION.json").read_text(encoding="utf-8"))
    hist = json.loads((RAIZ / "resultados/training_history_resumen.json").read_text(encoding="utf-8"))
    readme = (RAIZ / "README.md").read_text(encoding="utf-8")

    for s in ("42", "43", "44"):
        d = resumen["semillas"][s]
        texto = f"{d['evaluacion_retorno_medio']:.2f}".replace(".", ",").replace("-", "\u2212")
        check(f"el README reporta la ablacion de la semilla {s}", texto in readme, texto)

    ag = resumen["agregado"]
    check("la ablacion fracasa en una de las tres semillas",
          ag["peor_semilla"] < -100.0, f"peor semilla {ag['peor_semilla']:.2f}")
    check("la desviacion entre semillas de la ablacion es la que cita el README",
          f"{ag['sd_entre_semillas']:.2f}".replace(".", ",") in readme, f"{ag['sd_entre_semillas']:.2f}")
    check("las terminaciones de la ablacion son 288 de 300",
          ag["terminaciones_totales"] == 288, f"{ag['terminaciones_totales']}/300")

    correcto = [json.loads((RAIZ / "resultados/evaluation_metrics_100ep.json").read_text(encoding="utf-8"))["mean_return"]]
    for s in (43, 44):
        correcto.append(json.loads((RAIZ / f"resultados/semillas/resultado_seed_{s}.json").read_text(
            encoding="utf-8"))["evaluacion_greedy"]["retorno_medio"])
    sd_ok = st.stdev(correcto)
    check("la version correcta es mas estable entre semillas que la ablacion",
          sd_ok < ag["sd_entre_semillas"], f"{sd_ok:.2f} frente a {ag['sd_entre_semillas']:.2f}")

    # el 0,027 % de §10.2 sale de la corrida de la ABLACION, no de la correcta
    abl = json.loads((RAIZ / "resultados/ablacion/ablacion_seed_42.json").read_text(encoding="utf-8"))
    truncados = sum(abl["truncated"])
    pasos = sum(abl["lengths"])
    check("el 0,027 % que cita §10.2 sale de la corrida de la ablacion",
          abs(truncados / pasos * 100 - 0.027) < 0.001,
          f"{truncados}/{pasos:,} = {truncados/pasos:.4%}")
    check("§10.2 cita las 41 transiciones y los 154.568 pasos de esa corrida",
          truncados == 41 and pasos == 154_568, f"{truncados} y {pasos:,}")

    # y el 3,8 %, que es la cifra que la explicacion usa, se recalcula aqui.
    # Se carga el modulo por ruta y no por el paquete, para que esta comprobacion
    # funcione tambien en un Python sin torch instalado.
    contar = _cargar_por_ruta("minilotes", RAIZ / "src/acrobot_dqn/minilotes.py").contar
    ml = contar(RAIZ / "resultados/ablacion/ablacion_seed_42.json")
    check("el 3,8 % de minilotes afectados que cita §10.2 se reproduce",
          abs(ml["fraccion_de_minilotes"] * 100 - 3.8) < 0.1,
          f"{ml['minilotes_afectados']:,.0f} de {ml['minilotes']:,} = "
          f"{ml['fraccion_de_minilotes']:.1%}")
    check("§10.2 cita unos 5.900 minilotes de 153.569",
          abs(ml["minilotes_afectados"] - 5_861) < 5 and ml["minilotes"] == 153_569,
          f"{ml['minilotes_afectados']:,.0f} de {ml['minilotes']:,}")
    check("la figura de la ablacion existe",
          (RAIZ / "resultados/ablacion_truncacion.png").exists(), "presente")
except Exception as e:
    check("la ablacion se puede contrastar", False, f"{type(e).__name__}: {e}")

titulo("11. La comparacion con Double DQN de §10.3")
try:
    dd = json.loads((RAIZ / "resultados/double_dqn/RESUMEN_DOUBLE_DQN.json").read_text(encoding="utf-8"))
    readme = (RAIZ / "README.md").read_text(encoding="utf-8")

    for s in ("42", "43", "44"):
        texto = f"{dd['double_dqn']['retorno_por_semilla'][s]:.2f}".replace(".", ",").replace("-", "\u2212")
        check(f"el README reporta Double DQN en la semilla {s}", texto in readme, texto)

    check("Double DQN mejora en las tres semillas",
          dd["semillas_en_que_double_mejora"] == 3, "3 de 3")
    check("ninguna de las dos versiones deja episodios sin resolver",
          dd["dqn"]["terminaciones"] == 300 and dd["double_dqn"]["terminaciones"] == 300,
          "300/300 y 300/300")
    check("en el estado inicial las dos se quedan cortas",
          dd["dqn"]["sesgo_medio"] < 0 and dd["double_dqn"]["sesgo_medio"] < 0,
          f"{dd['dqn']['sesgo_medio']:.2f} y {dd['double_dqn']['sesgo_medio']:.2f}")

    sesgo = json.loads((RAIZ / "resultados/double_dqn/medicion_sesgo.json").read_text(encoding="utf-8"))
    for metodo in ("dqn", "double_dqn"):
        tramos = sesgo["agregado"][metodo]["por_tramo"]
        check(f"{metodo}: el sesgo cambia de signo con la distancia a la meta",
              tramos["0-19"]["sesgo_medio"] < 0 < tramos["100+"]["sesgo_medio"],
              f"{tramos['0-19']['sesgo_medio']:+.2f} cerca de la meta, "
              f"{tramos['100+']['sesgo_medio']:+.2f} lejos")
        texto = f"{tramos['100+']['sesgo_medio']:.2f}".replace(".", ",").replace("-", "\u2212")
        check(f"el README reporta el tramo lejano de {metodo}",
              texto in (RAIZ / "README.md").read_text(encoding="utf-8"), texto)

    # §10.3 publica los dos numeros de los que sale la resta, no solo la resta
    for metodo in ("dqn", "double_dqn"):
        tramos = sesgo["agregado"][metodo]["por_tramo"]
        check(f"{metodo}: cada tramo trae predicho, real y cota, no solo el sesgo",
              all({"valor_predicho", "retorno_real", "cota_superior_del_retorno_real"}
                  <= set(t) for t in tramos.values()), "los tres campos en los seis tramos")
        check(f"{metodo}: la resta publicada cuadra con sus dos sumandos",
              all(abs(t["valor_predicho"] - t["retorno_real"] - t["sesgo_medio"]) < 0.02
                  for t in tramos.values()), "coinciden hasta el redondeo")

    # el limite que §10.3 declara: en el tramo lejano la cota fuerza el signo positivo
    cota = sesgo["agregado"]["dqn"]["por_tramo"]["100+"]["cota_superior_del_retorno_real"]
    check("§10.3 declara la cota del retorno real del tramo lejano",
          f"{cota:.2f}".replace(".", ",").replace("-", "\u2212")
          in (RAIZ / "README.md").read_text(encoding="utf-8"), f"{cota:.2f}")
    check("la medicion del sesgo trae la advertencia sobre esa cota",
          "advertencia" in sesgo and "cota" in sesgo["advertencia"].lower(), "presente")
    check("§10.3 declara que el tramo lejano de Double DQN promedia dos semillas",
          sesgo["agregado"]["double_dqn"]["por_tramo"]["100+"]["corridas"] == 2
          and "dos** semillas" in (RAIZ / "README.md").read_text(encoding="utf-8"),
          "2 corridas, declarado")
    check("la medicion del sesgo cubre las seis corridas",
          len(sesgo["por_metodo_y_semilla"]) == 6, "6 corridas")
    check("estan los pesos de las seis corridas para recalcularlo",
          all((RAIZ / f"resultados/pesos/{n}.pt").exists()
              for n in ("dqn_seed_43", "dqn_seed_44", "double_dqn_seed_42",
                        "double_dqn_seed_43", "double_dqn_seed_44")), "presentes")
    check("Double DQN se queda mas corto que DQN, como dice §10.3",
          dd["double_dqn"]["sesgo_medio"] < dd["dqn"]["sesgo_medio"],
          f"{dd['double_dqn']['sesgo_medio']:.2f} frente a {dd['dqn']['sesgo_medio']:.2f}")
    check("el sesgo es coherente con V(s0) y el retorno real",
          all(abs(dd[m]["v_predicho_por_semilla"][s] - dd[m]["retorno_descontado_por_semilla"][s]
                  - dd[m]["sesgo_por_semilla"][s]) < 0.02
              for m in ("dqn", "double_dqn") for s in ("42", "43", "44")),
          "las seis filas cuadran")
    check("la figura del estado y las acciones de §2 existe",
          (RAIZ / "resultados/estado_y_acciones.png").exists(), "presente")
    check("los valores de la observacion dibujada quedan registrados",
          (RAIZ / "resultados/estado_y_acciones.json").exists(), "presente")
    check("la figura del entorno de §1 existe",
          (RAIZ / "resultados/ambiente_acrobot.png").exists(), "presente")
    check("la figura de §10.3 existe",
          (RAIZ / "resultados/double_dqn.png").exists(), "presente")
except Exception as e:
    check("la comparacion con Double DQN se puede contrastar", False, f"{type(e).__name__}: {e}")

titulo("11.1 La descomposicion de la varianza de §10.1")
try:
    # §10.1 no lee las tres dispersiones como componentes de varianza: el contraste
    # formal no distingue el efecto de estado de cero, y el README lo declara
    M = np.array([json.loads((RAIZ / "resultados/evaluation_metrics_100ep.json").read_text(
        encoding="utf-8"))["returns"]] + [json.loads(
        (RAIZ / f"resultados/semillas/resultado_seed_{s}.json").read_text(encoding="utf-8"))
        ["retornos_evaluacion"] for s in (43, 44, 45, 46)], dtype=float)
    fil, col, gran = M.mean(1), M.mean(0), M.mean()
    residuo = M - fil[:, None] - col[None, :] + gran
    ms_col = M.shape[0] * ((col - gran) ** 2).sum() / (M.shape[1] - 1)
    ms_int = (residuo ** 2).sum() / ((M.shape[0] - 1) * (M.shape[1] - 1))
    check("las tres dispersiones de §10.1 salen de la matriz 5x100",
          abs(st.stdev(fil) - 1.82) < 0.01 and abs(st.stdev(col) - 7.58) < 0.01
          and abs(float(np.std(residuo)) - 15.08) < 0.01,
          f"{st.stdev(fil):.2f} · {st.stdev(col):.2f} · {float(np.std(residuo)):.2f}")
    check("el efecto de estado no se distingue del residuo, como declara §10.1",
          abs(ms_col / ms_int - 1.0) < 0.1, f"F(99, 396) = {ms_col / ms_int:.2f}")
    check("y la dispersion de las medias por estado es la que produce el residuo solo",
          abs(float(np.std(residuo)) / 5 ** 0.5 - st.stdev(col)) < 1.0,
          f"{float(np.std(residuo)) / 5 ** 0.5:.2f} esperado frente a {st.stdev(col):.2f}")
except Exception as e:
    check("la descomposicion de §10.1 se puede contrastar", False, f"{type(e).__name__}: {e}")

titulo("12. El baseline tabular de §10.4")
try:
    tb = json.loads((RAIZ / "resultados/tabular/RESUMEN_TABULAR.json").read_text(encoding="utf-8"))
    readme = (RAIZ / "README.md").read_text(encoding="utf-8")

    for s in ("42", "43", "44"):
        texto = f"{tb['por_semilla'][s]['retorno_medio']:.2f}".replace(".", ",").replace("-", "\u2212")
        check(f"el README reporta el tabular de la semilla {s}", texto in readme, texto)

    check("el tabular tambien alcanza la meta en los 300 episodios",
          tb["agregado"]["terminaciones"] == 300, "300/300")
    check("ninguna semilla del tabular supera el umbral oficial",
          all(v["retorno_medio"] < -100 for v in tb["por_semilla"].values()),
          f"peor {min(v['retorno_medio'] for v in tb['por_semilla'].values()):.2f}")
    check("el DQN le saca mas del doble de ventaja",
          tb["dqn_mismas_semillas"]["retorno_medio"] / tb["agregado"]["retorno_medio"] < 0.5,
          f"{tb['dqn_mismas_semillas']['retorno_medio']:.2f} frente a {tb['agregado']['retorno_medio']:.2f}")
    barrido = json.loads((RAIZ / "resultados/tabular/barrido_hiperparametros.json").read_text(
        encoding="utf-8"))
    check("§10.4 declara que el barrido corrio con una sola semilla",
          "una sola semilla" in readme, "declarado")
    # §10.4 no se apoya en el barrido de una semilla: lo repite con tres
    tres = json.loads((RAIZ / "resultados/tabular/rejillas_tres_semillas.json").read_text(
        encoding="utf-8"))["rejillas"]
    readme = (RAIZ / "README.md").read_text(encoding="utf-8")
    por_rejilla = {r["rejilla"]: r for r in tres}
    check("la repeticion con tres semillas reproduce el barrido en la semilla 42",
          all(abs(r["retorno_por_semilla"]["42"]
                  - next(b["eval_retorno"] for b in barrido
                         if b["bins_angulo"] == int(r["rejilla"].split("x")[0])
                         and b["alpha"] == 0.1)) < 0.01 for r in tres),
          "las cuatro rejillas coinciden")
    for r in tres:
        texto = f"{r['retorno_medio']:.2f}".replace(".", ",").replace("-", "\u2212")
        check(f"§10.4 reporta la media de tres semillas de la rejilla {r['rejilla']}",
              texto in readme, texto)

    gruesa, elegida = por_rejilla["8x6"], por_rejilla["12x10"]
    check("con tres semillas las dos rejillas gruesas son indistinguibles",
          abs(gruesa["retorno_medio"] - elegida["retorno_medio"])
          < min(gruesa["sd_entre_semillas"], elegida["sd_entre_semillas"]),
          f"{gruesa['retorno_medio']:.2f} y {elegida['retorno_medio']:.2f}, "
          f"con SD {gruesa['sd_entre_semillas']:.2f} y {elegida['sd_entre_semillas']:.2f}")
    check("la cobertura no separa las dos rejillas gruesas, como dice §10.4",
          gruesa["cobertura_media"] / elegida["cobertura_media"] > 1.5
          and abs(gruesa["retorno_medio"] - elegida["retorno_medio"])
          < min(gruesa["sd_entre_semillas"], elegida["sd_entre_semillas"]),
          f"8x6 cubre {gruesa['cobertura_media'] / elegida['cobertura_media']:.1f} veces mas "
          f"y la diferencia de retorno es {abs(gruesa['retorno_medio'] - elegida['retorno_medio']):.2f}")
    check("las visitas por celda se dividen por 3,5 entre las dos gruesas sin mover el retorno",
          gruesa["visitas_por_celda_visitada"] / elegida["visitas_por_celda_visitada"] > 3.0
          and abs(gruesa["retorno_medio"] - elegida["retorno_medio"])
          < min(gruesa["sd_entre_semillas"], elegida["sd_entre_semillas"]),
          f"{gruesa['visitas_por_celda_visitada']:.0f} frente a "
          f"{elegida['visitas_por_celda_visitada']:.0f} visitas por celda, "
          f"{gruesa['retorno_medio']:.2f} frente a {elegida['retorno_medio']:.2f}")
    check("afinar la rejilla si empeora el retorno de forma consistente",
          all(tres[i]["retorno_medio"] > tres[i + 1]["retorno_medio"] for i in (1, 2)),
          " > ".join(f"{r['retorno_medio']:.2f}" for r in tres[1:]))
    term = [round(sum(r["terminacion_por_semilla"].values()) * 100) for r in tres]
    check("y hace que el tabular deje de resolver el entorno",
          min(term[:2]) > max(term[2:]) and max(term[2:]) < 300,
          " · ".join(f"{t}/300" for t in term))
    check("§10.4 reporta esas terminaciones",
          all(f"| {t} / 300 |" in readme for t in term), " · ".join(str(t) for t in term))

    # §10.4 declara que el tabular gasta mas pasos de entorno que el DQN
    pasos_tab = sum(json.loads((RAIZ / "resultados/tabular/tabular_seed_42.json").read_text(
        encoding="utf-8"))["lengths"])
    check("§10.4 declara los pasos de entorno del tabular y son mas que los del DQN",
          pasos_tab > 158_451 and f"{pasos_tab:,}".replace(",", ".")
          in (RAIZ / "README.md").read_text(encoding="utf-8"),
          f"{pasos_tab:,} frente a 158.451")

    check("la cobertura de la tabla es la que cita §10.4",
          abs(tb["agregado"]["cobertura_media"] - 10.9) < 0.05,
          f"{tb['agregado']['cobertura_media']} %")
    q = tb["presupuesto_quintuplicado"]
    check("quintuplicar el presupuesto no lleva el tabular al umbral",
          q["episodios"] == 6000 and q["retorno_medio"] < -100,
          f"{q['episodios']} episodios, {q['retorno_medio']:.2f}")
    check("el barrido de hiperparametros esta entregado",
          len(json.loads((RAIZ / "resultados/tabular/barrido_hiperparametros.json").read_text(
              encoding="utf-8"))) == 12, "12 configuraciones")
    check("la figura de §10.4 existe",
          (RAIZ / "resultados/tabular.png").exists(), "presente")
except Exception as e:
    check("el baseline tabular se puede contrastar", False, f"{type(e).__name__}: {e}")

# ─────────────────────────────────────────────────────────────── veredicto
print(f"\n{'='*72}")
print(f"RESULTADO: {ok_total} comprobaciones OK · {fallos} fallos")
if fallos == 0:
    print("\n  ✓ Cada cifra comprobada coincide con el archivo de resultados que la sostiene.")
    print("    Alcance: presencia de los archivos, metricas de entrenamiento y evaluacion,")
    print("    las cinco semillas, la integridad del historico comprimido y la coincidencia")
    print("    entre las salidas guardadas del notebook y los artefactos.")
else:
    print(f"\n  ✗ Hay {fallos} discrepancias, marcadas con !! en la salida de arriba.")
print(f"{'='*72}\n")
sys.exit(0 if fallos == 0 else 1)
