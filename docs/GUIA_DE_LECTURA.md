# Dónde está cada cosa

La actividad pide ocho contenidos en el README y cuatro descripciones del entorno. Esta tabla dice
en qué sección está cada uno y con qué archivo se puede contrastar, para que nadie tenga que leer
el documento entero buscando un punto concreto.

> **Estructura del repositorio:** `README.md` contiene el trabajo completo (10 secciones).
> Los experimentos de control extendidos —robustez con cinco semillas, ablación de truncación,
> Double DQN y baseline tabular— están en [`docs/ANALISIS_EXTENDIDO.md`](ANALISIS_EXTENDIDO.md).

---

## El entorno, punto por punto

Van en el orden con el que conviene empezar cualquier problema de refuerzo: primero el estado,
después las acciones, después la recompensa y por último la terminación.

| Lo que pide la actividad | Sección en README.md | Con qué se contrasta |
|---|---|---|
| Espacio de observaciones | §2, con figura | `Box(6,)` de tipo `float32`, con el rango declarado de las seis componentes y el medido con política aleatoria y con la política entrenada |
| Espacio de acciones | §2, con figura | `Discrete(3)` — torque −1, 0, +1 |
| Sistema de recompensas | §3 | −1 por paso salvo el paso terminal, que da 0; verificado sobre los 100 episodios de evaluación |
| Condiciones de terminación | §3 | `terminated` cuando el extremo supera la altura objetivo; `truncated` a los 500 pasos |
Las cuatro se pueden reproducir en vivo con `acrobot inspect`, que consulta el entorno instalado en
lugar de repetir lo que dice el README.

---

## Los ocho contenidos del README

| Contenido (criterio del taller) | Sección en README.md | Título exacto en el README |
|---|---|---|
| Descripción del ambiente | §1 | El ambiente |
| Acciones y observaciones | §2 y §3 | El estado y las acciones · La recompensa y la terminación |
| Flujo lógico del entrenamiento | §4 | Flujo del entrenamiento DQN |
| Particularidades del entorno | §5 | Particularidades del entorno |
| Explicación de la red neuronal | §6 y §7 | Arquitectura de la red neuronal · Hiperparámetros |
| Resultados del entrenamiento | §8 | Resultados del entrenamiento |
| Reflexión sobre los resultados | §9 | Reflexión sobre los resultados |
| Reflexión sobre lo que más costó | §10 | Reflexión sobre lo que más costó |

Los experimentos de control que profundizan en los resultados (robustez con cinco semillas, ablación de truncación, Double DQN y baseline tabular) están en `docs/ANALISIS_EXTENDIDO.md`.

---

## Las cifras y el archivo del que salen

> **Nota:** §1–§10 se refieren a `README.md`. §10.1–§10.4 se refieren a `docs/ANALISIS_EXTENDIDO.md`.

| Cifra | Valor | Archivo |
|---|---:|---|
| Retorno medio de evaluación | −80,96 | `resultados/evaluation_metrics_100ep.json` |
| Intervalo de confianza 95 % | [−83,41, −78,51] | `resultados/evaluation_metrics_100ep.json` |
| Umbral oficial del entorno | −100,0 | `env.spec.reward_threshold`, vía `acrobot inspect` |
| Política aleatoria, medida | −500,00 | `resultados/full_summary.json` · reproducible con `acrobot inspect` |
| Mejor episodio de entrenamiento | −60 (episodio 760) | `resultados/training_history_resumen.json` |
| Media de los últimos 100 episodios | −85,68 | `resultados/training_history_resumen.json` |
| Terminaciones en las cinco semillas | 500 / 500 | `resultados/semillas/RESUMEN_5_SEMILLAS.json` |
| Ablación de §10.2, media de las tres semillas | −97,85 | `resultados/ablacion/RESUMEN_ABLACION.json` |
| Ablación, peor semilla | −134,05 · 88/100 | `resultados/ablacion/evaluacion_seed_44.json` |
| Desviación entre semillas: correcto · ablación | 1,42 · 31,37 | `resultados/ablacion/RESUMEN_ABLACION.json` |
| Double DQN, media de las tres semillas | −77,12 | `resultados/double_dqn/RESUMEN_DOUBLE_DQN.json` |
| Sesgo en s₀: DQN · Double DQN | −6,03 · −11,96 | `resultados/double_dqn/medicion_sesgo.json` |
| Sesgo lejos de la meta (100+ pasos) | +9,53 · +11,25 | `resultados/double_dqn/medicion_sesgo.json` |
| Q-learning tabular, media de tres semillas | −195,32 | `resultados/tabular/RESUMEN_TABULAR.json` |
| Cobertura de la tabla | 1.566 celdas: 10,9 % de las 14.400 nominales, 26 % de las 6.048 alcanzables | `resultados/tabular/RESUMEN_TABULAR.json` |
| Las cuatro rejillas con tres semillas | −192,90 · −195,32 · −249,65 · −330,57 | `resultados/tabular/rejillas_tres_semillas.json` |
| Pasos de entorno: DQN · tabular | 158.451 · 388.550 | `resultados/training_history_resumen.json` · `resultados/tabular/tabular_seed_42.json` |
| Transiciones que el cambio de §10.2 afecta | 41 de 154.568 (0,027 %) | `resultados/ablacion/ablacion_seed_42.json` |
| Minilotes que llegan a contener alguna | 5.861 de 153.569 (3,8 %) | reproducible con `acrobot minilotes` |
| Desviación entre semillas · dentro de cada una | 1,82 · 15,47 | `resultados/semillas/RESUMEN_5_SEMILLAS.json` |
| Desviación entre las tres semillas de §10.2 y §10.3 | 1,42 | `resultados/ablacion/RESUMEN_ABLACION.json` |
| Parámetros de la red | 17.795 | contados sobre el modelo, no estimados |
| Episodios de entrenamiento | 1.200 | `resultados/training_history_resumen.json` |
| Pasos de entorno · actualizaciones de la red | 158.451 · 157.452 | `resultados/training_history_1200ep.json.gz` |

`acrobot verify` recorre estas correspondencias una por una, compara el texto del README contra los
archivos de resultados y comprueba además que las salidas guardadas del notebook digan lo mismo.

---

## Los tres componentes del código

La lógica del paquete se reparte en los tres componentes con los que se organiza un proyecto de
aprendizaje por refuerzo:

| Componente | Dónde vive | Qué hace |
|---|---|---|
| **Ambiente** | `acrobot inspect`, `acrobot rangos`, `acrobot estado`, `acrobot ambiente`, `acrobot demo` | consulta Acrobot-v1 en vivo: espacios, recompensa, terminación y rangos de observación; `acrobot demo` abre una ventana pygame con el agente entrenado jugando en tiempo real |
| **Agentes** | `dqn_acrobot_v3.py`, `ablacion.py`, `double_dqn.py`, `tabular.py` | las cuatro variantes que entrenamos; la primera es la del entregable y las otras tres existen para compararse con ella |
| **Evaluación** | `evaluate()`, `sesgo.py`, `minilotes.py`, `verificar_resultados.py` | mide la política congelada sobre semillas disjuntas, el sesgo de los valores Q, el alcance del error de la ablación (§10.2 — análisis extendido) y la coherencia entre el texto y los artefactos |

`cli.py` no es un cuarto componente: es la puerta de entrada que expone los tres desde la terminal.

## Los archivos del repositorio

| Archivo | Qué contiene |
|---|---|
| `README.md` | el documento principal del trabajo — 10 secciones que cubren todos los criterios de la actividad |
| `docs/ANALISIS_EXTENDIDO.md` | los cuatro experimentos de control: robustez con cinco semillas, ablación de truncación, Double DQN y baseline tabular |
| `docs/GUIA_DE_LECTURA.md` | este documento: dónde está cada contenido que pide la actividad |
| `notebooks/SAR_U3_Taller2_DQN_EJECUTADO.ipynb` | el mismo recorrido que el README pero celda a celda, con el código y las salidas guardadas de la corrida real |
| `ver_en_vivo.py` | script auxiliar para ver el agente jugando con ventana pygame (equivalente a `acrobot demo`) |
| `src/acrobot_dqn/dqn_acrobot_v3.py` | el algoritmo: red, buffer, política ε-greedy, objetivo de Bellman y bucle de entrenamiento |
| `src/acrobot_dqn/cli.py` | la interfaz de terminal `acrobot` — dieciséis subcomandos, incluido `demo` para ver el agente en tiempo real |
| `src/acrobot_dqn/ablacion.py` | la variante de §10.2 del análisis extendido: el mismo entrenamiento con una línea distinta |
| `src/acrobot_dqn/double_dqn.py` | Double DQN: el mismo bucle con otro objetivo de Bellman (§10.3 del análisis extendido) |
| `src/acrobot_dqn/figura_ablacion.py` | regenera la figura de la ablación (§10.2 — análisis extendido) desde los artefactos, sin reentrenar |
| `src/acrobot_dqn/sesgo.py` | mide el sesgo de los valores Q por distancia a la meta (§10.3 — análisis extendido) |
| `src/acrobot_dqn/minilotes.py` | cuenta en cuántas actualizaciones entra el error de la ablación (§10.2 — análisis extendido) |
| `src/acrobot_dqn/tabular.py` | Q-learning tabular, la alternativa sin red (§10.4 — análisis extendido) |
| `src/acrobot_dqn/figura_ambiente.py` | renderiza un episodio y dibuja la figura del entorno (§1 del README) |
| `src/acrobot_dqn/figura_estado.py` | dibuja el estado y las acciones sobre el mecanismo (§2 del README) |
| `src/acrobot_dqn/figura_double.py` | regenera la figura de Double DQN (§10.3 — análisis extendido) desde los artefactos, sin reentrenar |
| `src/acrobot_dqn/figura_tabular.py` | regenera la figura del tabular (§10.4 — análisis extendido) desde los artefactos, sin reentrenar |
| `src/acrobot_dqn/__init__.py` | expone la versión del paquete |
| `tests/test_algoritmo.py` | cuatro comprobaciones del algoritmo que corren en segundos |
| `resultados/` | pesos entrenados, historia completa, métricas de evaluación, rangos de observación, las cinco semillas y las figuras |
| `verificar_resultados.py` | las 133 comprobaciones que ejecuta `acrobot verify` |

El cuaderno escribe sus salidas en `artifacts/` mientras corre. Los archivos publicados en
`resultados/` —la carpeta que citan el README y `acrobot verify`— son los de la corrida del CLI.

El archivo del algoritmo tiene SHA-256 `1fd13fa5…47e8`. `acrobot verify` lo recalcula y lo compara,
de modo que cualquier cambio en el algoritmo se detecta antes de leer una sola cifra.

---

## Referencia cruzada con el código

| Concepto | Dónde en el código |
|---|---|
| Arquitectura de la red | `src/acrobot_dqn/dqn_acrobot_v3.py`, clase `QNetwork` |
| Replay buffer | `src/acrobot_dqn/dqn_acrobot_v3.py`, clase `ReplayBuffer` |
| Decaimiento de ε | `epsilon_by_step()` |
| Objetivo de Bellman y enmascaramiento con `terminated` | `optimize_step()` |
| Sincronización de la red objetivo cada 1.000 pasos | `train_dqn()` |
| Evaluación greedy con semillas disjuntas | `evaluate()` |
