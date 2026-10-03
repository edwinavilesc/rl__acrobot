# DQN sobre Acrobot-v1 — Análisis extendido

> 📋 Este documento contiene los experimentos de control del proyecto. El README principal con la descripción completa del trabajo está en [README.md](../README.md).

---

# DQN sobre Acrobot-v1

**Simulación y Aprendizaje por Refuerzo · Unidad 3 · Taller 2**
Maestría en Inteligencia Artificial, Universidad de La Sabana · MIA 2026-5
Profesor: Emilio Muñoz Pérez

**Grupo 1**

| Integrante | Correo |
|---|---|
| Saud Mauricio Gonzalez Rodriguez | saudgoro@unisabana.edu.co |
| Luis Alfonso Fajardo Andrade | luisfaan@unisabana.edu.co |
| Gloria Yaned Leguizamon Arboleda | glorialear@unisabana.edu.co |
| Ricardo Sandoval Chaves | ricardosach@unisabana.edu.co |
| Edwin Aviles Cogollo | edwinavco@unisabana.edu.co |
| Edgar Julian Mendez Ortegon | edgarmenor@unisabana.edu.co |
| Orly Alberto Alvao Barajas | orlyalba@unisabana.edu.co |

Un agente DQN que resuelve Acrobot-v1 y alcanza la meta en **100 de 100** episodios de evaluación
con semillas que nunca vio durante el entrenamiento, con un retorno medio de **−80,96**, por encima
del umbral de −100 que el propio entorno declara como criterio de resolución. Repetido con **cinco semillas**: las cinco superan el umbral, con **500 de 500** terminaciones verdaderas.

---

## 1. El ambiente

Acrobot es un péndulo doble subactuado. Dos eslabones unidos en cadena cuelgan de un punto fijo; el
motor está en la articulación **del medio**, no en la de arriba. El objetivo es levantar el extremo
libre hasta una altura determinada — concretamente, hasta que la punta del segundo eslabón supere
la altura del punto de anclaje en una unidad de longitud.

Lo que hace interesante al problema es que **el motor es demasiado débil para levantar el brazo de
un tirón**. No existe una secuencia de empujones en una sola dirección que resuelva el episodio. El
agente tiene que aprender a balancear: empujar en un sentido, dejar que la gravedad devuelva el
brazo, empujar en el otro en el momento justo, y acumular energía a lo largo de varias oscilaciones
hasta que el impulso baste. Es un problema de sincronía, no de fuerza.

![Un episodio del agente entrenado](resultados/ambiente_acrobot.png)

Los seis fotogramas son de un episodio real de nuestro agente, renderizado por Gymnasium. El brazo
parte colgando en reposo, gana amplitud oscilación tras oscilación y cruza la altura objetivo en el
paso 87. La traza de abajo mide esa misma altura a lo largo del episodio: sube a saltos y con
retrocesos, porque cada empujón solo sirve si llega cuando el brazo ya viene en esa dirección.

En cada paso el agente elige una de tres cosas: empujar en un sentido, empujar en el otro, o no
empujar —eso es lo que rotula cada fotograma como «torque +1», «torque −1» o «torque —», y §2 lo
define con precisión—. De los 87 pasos de este episodio el agente eligió empujar en los 87, 46 veces
en un sentido y 41 en el otro, y ni una sola vez se quedó quieto; el rótulo del último fotograma no
es una excepción a eso, sino el estado ya terminal, en el que no queda decisión que tomar.

El entorno viene de Sutton (1996) y está en `gymnasium.envs.classic_control`.

## 2. El estado y las acciones

Antes de elegir un método hay cuatro cosas que entender, y en este orden: **qué observa el agente**,
**qué puede hacer**, **qué recompensa recibe** y **cuándo se acaba un episodio**. Los dos primeros
están en esta sección y los dos siguientes en §3. No es una formalidad: el estado decide la entrada
de la red, las acciones deciden su salida, la recompensa es lo único que define qué significa
«bien», y la distinción entre terminar y ser interrumpido cambia la ecuación que se optimiza —como
acabamos midiendo en §10.2, donde confundirlas cuesta un modo de fallo entero.

Ninguno de los cuatro los tomamos de la documentación: los cuatro se consultan en vivo con `acrobot
inspect` sobre el entorno instalado, y esta sección reporta lo que devuelve.

**El estado primero**, porque es la entrada de la red y tiene que ser números. Estas son las seis
componentes que entrega el entorno y las tres acciones que el agente puede tomar, con los valores
reales de un instante concreto de la corrida:

![El estado y las acciones](resultados/estado_y_acciones.png)

**Espacio de observación:** `Box(6,)` de tipo `float32`. Las seis componentes, en orden:

| Índice | Magnitud | Rango declarado | Con política aleatoria | Con la política entrenada |
|---|---|---|---|---|
| 0 | cos θ₁ | [−1, 1] | [−0,719, +1,000] | [−1,000, +1,000] |
| 1 | sen θ₁ | [−1, 1] | [−1,000, +1,000] | [−1,000, +1,000] |
| 2 | cos θ₂ | [−1, 1] | [−1,000, +1,000] | [−1,000, +1,000] |
| 3 | sen θ₂ | [−1, 1] | [−1,000, +1,000] | [−1,000, +1,000] |
| 4 | ω₁ (velocidad angular del primer eslabón) | [−4π, 4π] ≈ ±12,566 | [−5,017, +4,914] | [−6,105, +6,684] |
| 5 | ω₂ (velocidad angular del segundo) | [−9π, 9π] ≈ ±28,274 | [−8,229, +8,191] | [−12,298, +11,621] |

Las dos columnas medidas salen de `acrobot rangos`, el comando que acompaña al código (§«Reproducir»),
que las recalcula: la aleatoria sobre 30 episodios con semillas 1000–1029, la entrenada sobre los
100 episodios de evaluación. Con `--save resultados/rangos_observacion.json` las escribe en el
archivo que este README cita, que es como se generó la copia publicada.

De la tabla se siguen dos consecuencias prácticas.

**Los ángulos vienen ya descompuestos en seno y coseno.** El entorno no entrega θ directamente, y
eso no es un capricho: θ es una variable circular, donde 0 y 2π son el mismo estado físico pero
números lejanísimos. Una red que recibiera θ tendría que aprender esa discontinuidad. Al recibir
(cos θ, sen θ) el estado se vuelve continuo y la red se ahorra el problema.

**No hizo falta preprocesar ni normalizar nada.** Cuatro de las seis componentes ya viven en
[−1, 1] por construcción. Las dos velocidades angulares declaran rangos de ±12,6 y ±28,3, y la
distribución que la red ve depende de qué política la genere: con política aleatoria el brazo casi
no se balancea y las velocidades se quedan en ±5,0 y ±8,2, mientras que con la política entrenada
—que balancea a propósito para ganar altura— llegan a ±6,7 y ±12,3. Incluso ese segundo rango, que
es el que importa, se queda holgadamente dentro de lo declarado y en una escala razonable para una
red con ReLU e inicialización estándar. Las alimentamos tal cual.

Medimos las dos distribuciones a propósito. Lo cómodo era decidir con los números de la política
aleatoria, que se obtienen antes de entrenar, y esos números subestiman en torno a un tercio lo que la red
termina recibiendo.

**Espacio de acción:** `Discrete(3)` — aplicar torque −1, 0 o +1 a la articulación central. Que sea
discreto es lo que hace el entorno compatible con DQN: el algoritmo necesita un `argmax` sobre un
conjunto finito de acciones.

**Las seis componentes de la observación son continuas**, y ahí está la decisión de fondo del
trabajo. Un método tabular necesita estados discretos, así que tendría que partir cada variable en
intervalos y perder lo que ocurre dentro de cada uno; una red toma los seis números tal cual. En
§10.4 entrenamos las dos alternativas y medimos cuánto cuesta esa diferencia, en lugar de dejarlo en
el argumento.

## 3. La recompensa y la terminación

Los otros dos puntos del orden que abre §2: qué recompensa recibe el agente, y cuándo se da por
acabado un episodio.

**Recompensa: −1 por cada paso mientras el brazo no alcance la altura objetivo; el paso que la
alcanza da 0.** Así está escrito en el entorno: `reward = -1.0 if not terminated else 0.0`. No hay
bonificación por llegar a la meta ni penalización extra por fallar. La única manera de ganar
recompensa es tardar menos.

La consecuencia es que **el retorno de un episodio que alcanza la meta es menos el número de pasos
que duró, más uno**, porque el último no descuenta; y el de un episodio que agota los 500 pasos sin
llegar es exactamente −500. Nuestros 100 episodios de evaluación cumplen la regla sin excepción: retorno medio −80,96 con
81,96 pasos de media, justo un punto de diferencia. Maximizar la recompensa y minimizar el tiempo
son entonces el mismo objetivo, y eso hace la métrica interpretable: −80,96 significa levantar el
brazo en unos 82 pasos.

Conviene señalar la excepción porque es fácil pasarla por alto. Si se mide la recompensa con una
política aleatoria —lo natural antes de entrenar— nunca se observa el 0: la política aleatoria no
llega a la meta en ninguno de 30 episodios, así que la muestra excluye por construcción el único
paso donde la regla cambia.

La señal es además **escasa en el sentido estricto**: mientras el agente no llegue a la meta, todos
los estados y todas las acciones reciben idéntica recompensa. No hay ninguna pista intermedia que
indique «vas mejorando». La única información sobre qué hacer llega cuando un episodio termina de
verdad antes de los 500 pasos.

**Terminación:**

- `terminated = True` — el extremo libre superó la altura objetivo. Es el final *real*: el estado
  siguiente no tiene valor futuro.
- `truncated = True` — se agotaron los 500 pasos (`env.spec.max_episode_steps = 500`). El episodio
  se corta por reloj, no porque el mundo se acabe.

**Esta distinción la tratamos explícitamente en el código, y es la decisión técnica que más peso
tuvo en el resultado.** El buffer guarda únicamente `terminated`:

```python
buf.push(obs, action, reward, nxt, terminated)          # truncated NO entra al buffer
...
target_q = r + gamma * next_q * (1 - terminated)        # solo el terminal real anula el bootstrap
```

Ese `(1 - terminated)` es lo que se llama **bootstrap**: estimar el valor del estado actual apoyándose
en el valor que la propia red asigna al estado siguiente. Cuando el episodio termina de verdad no hay
estado siguiente que valga nada, y el factor lo anula; en cualquier otro caso el valor futuro debe
contar.

El razonamiento es este: si la truncación anulara el bootstrap, el agente aprendería que el estado
del paso 500 vale cero, cuando en realidad es un estado normal del que se podría seguir. Esa
falsedad se propagaría hacia atrás por la trayectoria. Es una advertencia estándar en entornos con
límite temporal, y por eso escribimos el código así desde el principio.

Lo pusimos a prueba. En §10.2 entrenamos tres pares de agentes idénticos salvo en esa línea, y el
resultado matiza la advertencia: la versión rota aprende igual de bien en dos de las tres semillas y
fracasa en la tercera. El error no impide aprender; quita fiabilidad.

## 4. Flujo de entrenamiento DQN

El bucle de entrenamiento encadena siempre los mismos siete pasos:

1. la política ε-greedy elige una acción a partir de la observación actual;
2. `env.step(a)` la ejecuta y devuelve la recompensa, la observación siguiente y las dos banderas;
3. la transición `(s, a, r, s′, terminated)` entra al replay buffer;
4. se sortean 64 transiciones del buffer, de episodios distintos y de etapas distintas;
5. la red en línea predice `Q(s)[a]` y la red objetivo el `max_a′ Q_target(s′, a′)` del futuro;
6. se forma el objetivo `r + γ · max Q_target · (1 − terminated)` y se retropropaga la Huber loss;
7. cada 1.000 pasos la red objetivo copia los pesos de la red en línea.

Antes de entrar en cada pieza, conviene verlo entero: de dónde sale cada dato y a dónde va.

```
obs ──► política ε-greedy ──► env.step(a) ──► (s, a, r, s′, terminated) ──► ReplayBuffer
                                                                                │
                            ┌───────────────────────────────────────────────────┘
                            ▼
                   muestreo aleatorio de 64 transiciones
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
     Q_online(s)[a]              max_a′ Q_target(s′, a′)
              │                           │
              │                           ▼
              │              target = r + γ · max Q_target · (1 − terminated)
              ▼                           │
              └──────── Huber loss ◄──────┘
                            │
                   backward ──► clip_grad_norm(10.0) ──► Adam(lr=1e-3)
                            │
              cada 1.000 pasos: Q_target ← Q_online
```

Las cuatro piezas y por qué existen:

**Replay buffer (100.000 transiciones).** Las transiciones consecutivas de un episodio están
fuertemente correlacionadas: dos estados separados por un paso son casi el mismo estado. Entrenar
una red con lotes correlacionados viola el supuesto de muestras independientes y produce
actualizaciones sesgadas. El buffer rompe esa correlación muestreando al azar de todo el historial.
Además permite reutilizar cada transición muchas veces, lo que multiplica el aprovechamiento de la
experiencia.

**ε-greedy con decaimiento lineal (1,0 → 0,05 en 60.000 pasos).** Al principio el agente no sabe
nada y necesita explorar; al final ya tiene una política y explorar solo la degrada. El decaimiento
se cuenta **por paso del entorno, no por episodio**, lo cual tiene una consecuencia poco obvia: como
los episodios se acortan a medida que el agente mejora, los últimos 60.000 pasos cubren muchos más
episodios que los primeros. Por eso ε llegó a su piso de 0,05 en el episodio 213 —a los 60.059 pasos
acumulados— y no en un punto proporcional al total de episodios.

**Target network sincronizada cada 1.000 pasos.** Sin ella, la red persigue un objetivo que ella
misma define, y cada actualización mueve simultáneamente la predicción y el blanco. La copia
congelada da un objetivo estable durante mil pasos. Sincronizar por pasos en lugar de por episodios
es deliberado: en este entorno la longitud del episodio varía entre 500 y 70 pasos, así que
«cada 10 episodios» significaría cosas muy distintas al principio y al final.

**Actualización de Bellman con Huber loss.** El target es `r + γ·max Q_target(s′)·(1−terminated)`.
La pérdida es `smooth_l1_loss`, cuadrática para errores pequeños y lineal para los grandes: un
target ocasionalmente muy desviado no produce un gradiente desproporcionado. Combinada con el
recorte de norma a 10,0, mantiene el entrenamiento estable sin amortiguar las correcciones
pequeñas.

## 4.1 Alineación con el algoritmo presentado en clase

La sesión de Unidad 3 proyectó el **Algorithm 1 de Mnih** y separó explícitamente dos fases:
*Sampling* y *Training*. La implementación de este repositorio sigue esa misma estructura: inicializa
ReplayBuffer y dos redes, usa ε-greedy, almacena transiciones, toma minibatches aleatorios, calcula el
target con la red congelada y sincroniza la target cada cierto número de **pasos**.

| Material de clase | Implementación en este proyecto |
|---|---|
| Replay memory `D` de capacidad `N` | `ReplayBuffer(100000)` |
| Q online con pesos `θ` | `QNetwork(6, 3, 128)` |
| Q target con `θ⁻ = θ` | copia inicial exacta de `state_dict` |
| ε-greedy | `ε: 1,0 → 0,05` por pasos |
| minibatch aleatorio | `batch_size = 64` |
| `r + γ max Q_target` si no termina | mismo target, en API moderna con `(1-terminated)` |
| actualizar target cada `C` pasos | `C = 1000` pasos |

La diferencia terminológica importante es que el pseudocódigo histórico usa una sola noción de
“fin de episodio”, mientras Gymnasium separa `terminated` de `truncated`. En esta implementación solo
`terminated` anula el bootstrap; `truncated` detiene el bucle por límite temporal, pero no convierte
el estado en terminal del MDP.

## 5. Particularidades del ambiente y cómo condicionaron el diseño

**Recompensa constante y meta escasa.** Como todas las recompensas valen −1, la red no recibe
ninguna pista sobre el progreso hasta que un episodio termina de verdad. La terminación real es la
*única* señal cualitativamente distinta que existe en todo el entorno, y por eso la distinguimos de
la truncación en el código. Cuánto pesa esa distinción en el resultado es otra cuestión, y la
medimos en §10.2.

**Pero la exploración casi ciega sí encuentra la meta — y eso lo cambia todo.** **Diez de los
primeros cincuenta episodios alcanzaron la meta**, y el primero fue el episodio 11, cuando ε todavía
rondaba 0,9 y el agente apenas se distinguía del azar. Esa es la diferencia de fondo con MountainCar,
con el que Acrobot suele compararse.

Conviene señalar por qué la medición previa al entrenamiento daba 0 éxitos en 30 episodios: una
política completamente aleatoria empuja en direcciones que se cancelan, mientras que unas pocas
actualizaciones bastan para que el agente empiece a encadenar empujones en fase. El buffer contiene
trayectorias exitosas casi desde el arranque, y un buffer con éxitos contiene información sobre cómo
tener éxito. Nos ahorró cualquier técnica especial de exploración: ε-greedy simple bastó.

**Subactuación.** La solución obliga a una estrategia de balanceo en varias oscilaciones. Esto
implica que las decisiones buenas dependen de la **velocidad** tanto como de la posición — de ahí
que las dos componentes de velocidad angular del vector de observación sean imprescindibles y que
una red sin memoria explícita pueda resolverlo: el estado ya es markoviano.

**Horizonte corto y denso.** 500 pasos como máximo y un episodio resuelto ronda los 80. Con
γ = 0,99, el horizonte efectivo (≈ 1/(1−γ) = 100 pasos) cubre cómodamente un episodio completo. γ
no tuvimos que ajustarlo.

## 6. Arquitectura de la red y justificación

La red es la parte del diseño que más depende del entorno concreto, porque tiene que ajustarse a la
forma de la observación y al número de acciones. Esta es la que usamos:

```
Entrada  6 (estado continuo)
   │
Linear(6 → 128)   +  ReLU        896 parámetros
   │
Linear(128 → 128) +  ReLU     16.512 parámetros
   │
Linear(128 → 3)                  387 parámetros
   │
Salida   3 valores Q, uno por acción
                          ───────────────────
                   Total   17.795 parámetros
```

En §10.4 ponemos esta elección a prueba contra Q-learning tabular, que es la alternativa sin red.

**Por qué un MLP y no una red convolucional.** La observación es un vector de 6 números con
significado físico individual, no una imagen. No hay estructura espacial ni traslacional que
explotar, así que una convolución no aportaría nada y solo añadiría parámetros.

**Por qué dos capas ocultas de 128.** La función Q sobre este espacio no es lineal —depende de
productos entre ángulos y velocidades— así que una sola capa no basta. Dos capas ocultas dan
capacidad suficiente para esas interacciones sin que el número de parámetros se dispare. Como punto
de comparación externa: la implementación de referencia de Lapan para MountainCar en el capítulo 21
de *Deep Reinforcement Learning Hands-On* usa **una** capa oculta de 128 para un problema de
dimensión similar, así que dos con el mismo ancho es una elección holgada, no escasa.

**Por qué ReLU.** Es lo estándar, no satura para entradas positivas y es barata. La alternativa
clásica, `tanh`, satura y ralentiza el aprendizaje en las capas profundas.

**Por qué la salida no lleva activación.** Los valores Q son retornos *descontados* esperados. Con
γ = 0,99 y una recompensa de −1 por paso, el peor valor posible es −1/(1−γ) = −100, y en la red
entrenada caen entre −63,1 y +6,5. Una activación acotada entre 0 y 1 (sigmoide) o entre −1 y 1
(tanh) no podría representar esa escala, y softmax directamente cambiaría el significado de la
salida. En particular **no hay softmax**, y conviene decirlo porque es un error
frecuente: un DQN no produce una distribución de probabilidad
sobre acciones, produce una estimación de valor para cada una. La política sale del `argmax`, no de
un muestreo.

## 7. Hiperparámetros y por qué cada uno

Entrenar un DQN exige fijar de antemano una docena de valores que el problema no impone. Los
reunimos todos aquí, cada uno con la característica del entorno que lo justifica, para que ninguno
quede como un número suelto en el código.

| Parámetro | Valor | Razón |
|---|---:|---|
| `episodes` | 1.200 | Suficiente para estabilizar: la media móvil se asienta por encima de −90 desde el episodio 938 y en los últimos 300 solo oscila entre −90,9 y −83,2 (§9) |
| `gamma` | 0,99 | Horizonte efectivo ≈ 100 pasos, del orden del episodio resuelto (≈ 82) |
| `lr` | 1e-3 | Valor estándar de Adam para MLP pequeños; con 1e-4 el aprendizaje sería innecesariamente lento en una red de 17,8 k parámetros |
| `batch_size` | 64 | Compromiso entre varianza del gradiente y coste por paso; se actualiza en cada paso del entorno |
| `buffer_size` | 100.000 | Cubre 63 % de los 158.451 pasos: al final del entrenamiento se han descartado las 58.451 transiciones más antiguas, entre ellas los primeros éxitos. Lo que el buffer conserva es una ventana de las últimas ~100.000 transiciones, suficientemente ancha para mezclar episodios de etapas distintas |
| `learning_starts` | 1.000 | Entrenar con un buffer casi vacío produce lotes redundantes y sobreajuste inmediato |
| `target_update_steps` | 1.000 | Del orden de doce episodios resueltos: el blanco se mantiene fijo mientras el agente recorre varias trayectorias completas, y se refresca antes de que la política haya cambiado lo bastante como para que el objetivo quede obsoleto |
| `epsilon_start / end` | 1,0 → 0,05 | Exploración total al inicio; 5 % residual al final para no fijarse en un óptimo local |
| `epsilon_decay_steps` | 60.000 | ≈ 38 % del entrenamiento explorando activamente |
| `hidden` | 128 | Ver §6 |
| `max_grad_norm` | 10,0 | Ver §8: intervino en el 6,00 % de los pasos, régimen de seguridad sin distorsionar |
| `seed` | 42 | Reproducibilidad |
| `eval_episodes` | 100 | Muestra suficiente para un IC 95 % estrecho |

## 8. Resultados

Corrida única, semilla 42, CPU: **158.451 pasos de entorno y 157.452 actualizaciones de la red**.
La diferencia de 999 son los pasos iniciales en los que el buffer aún no había alcanzado las 1.000
transiciones que exige `learning_starts`, de modo que se recogió experiencia sin entrenar.

El reloj no es una propiedad del algoritmo. Esta misma corrida quedó registrada con 3.471 s en el
artefacto original y tardó 299 s al repetirla, con resultados idénticos elemento a elemento; las
cuatro semillas adicionales tardaron entre 290 y 302 s. No sabemos a qué atribuir la diferencia: los
dos artefactos declaran la misma plataforma, así que la diferencia está en las condiciones de
ejecución —carga de la máquina, hilos disponibles— y no en el algoritmo. Los pasos y las
actualizaciones, que sí son del algoritmo, no cambian entre corridas.

### Entrenamiento

| Métrica | Valor |
|---|---:|
| Mejor episodio | **−60,0** (episodio 760) |
| Media de los últimos 100 | **−85,68** |
| Mediana de los últimos 100 | −85,0 |
| Media global de los 1.200 | −131,09 |
| Alcanzan la meta (global) | 95,58 % |
| Alcanzan la meta (últimos 100) | **100 %** |
| Último episodio truncado | **350** |

De los 1.200 episodios, 1.147 alcanzan la meta y 54 agotan los 500 pasos. La suma da 1.201 porque
el episodio 47 hace las dos cosas: llega a la altura objetivo justo en el paso límite, de modo que
el entorno marca `terminated` y `truncated` a la vez. El código lo resuelve como corresponde
—guarda `terminated`, que es lo que define si hay valor futuro—, y por eso el episodio cuenta como
éxito.

### Evolución por bloques

| Bloque | Retorno medio | SD | Pasos medios | % meta |
|---|---:|---:|---:|---:|
| 1–200 | −290,68 | 148,30 | 291,4 | 76,0 % |
| 201–400 | −135,97 | 70,01 | 136,9 | 97,5 % |
| 401–600 | −98,73 | 23,05 | 99,7 | 100 % |
| 601–800 | −89,38 | 22,38 | 90,4 | 100 % |
| 801–1000 | −86,09 | 17,51 | 87,1 | 100 % |
| 1001–1200 | **−85,66** | **13,77** | 86,7 | 100 % |

La desviación estándar cae de 148,30 a 13,77. El agente no solo mejora su media: **deja de tener
episodios catastróficos**.

### Referencias para interpretar estos números

- **Política aleatoria: −500,00**, medida sobre 30 episodios (0 de 30 alcanzaron la meta).
- **Umbral oficial de Gymnasium: −100,0**, declarado por el propio entorno en
  `env.spec.reward_threshold`.

### Gradient clipping

Norma media 4,95, mediana 5,28, máxima 19,89, todas medidas **antes** del recorte: por eso la máxima
puede superar el umbral. **9.449 de 157.452 pasos (6,00 %)** lo superaron y fueron recortados a 10,0.
El clipping actúa sobre uno de cada dieciséis pasos, suficiente para contener los picos sin
amortiguar la corrección habitual.

### La loss sube mientras el agente mejora

Partimos los 157.452 pasos de optimización en seis tramos iguales; la tabla muestra el primero, el
tercero y el sexto, que bastan para ver la tendencia.

| Tramo de pasos | Loss media |
|---|---:|
| 0 – 26.242 | 0,0062 |
| 52.484 – 78.726 | 0,2155 |
| 131.210 – 157.452 | **0,4807** |

La loss se multiplica por 77 mientras el retorno pasa de −290 a −85. **No es un deterioro.** La loss
de un DQN mide el desacuerdo entre la red y un target que se mueve, no el desempeño del agente.

Al principio la red está inicializada cerca de cero y el objetivo vale apenas `r + γ·max Q ≈ −1`:
ambos términos son pequeños, el bootstrap todavía no ha propagado profundidad y acertar es trivial.
A medida que el agente aprende, los valores Q se separan dentro de la banda que γ = 0,99 permite —de
−100 a 0— y distinguir un estado bueno de uno mediocre se vuelve un problema de regresión
genuinamente más difícil. Una loss mayor sobre un problema más difícil no es un problema peor
resuelto, y por eso ninguna conclusión de este trabajo se apoya en la loss.

## 9. Curva de aprendizaje

La tabla por bloques resume mil doscientos episodios en seis filas. La curva los muestra todos, con
la media móvil de cincuenta encima para que la tendencia se lea sin ocultar la dispersión.

![Curva de aprendizaje](resultados/dqn_learning_curve_1200ep.png)

Retorno por episodio en azul claro, media móvil de 50 en naranja, y las dos referencias: el umbral
oficial de −100 (línea roja) y la política aleatoria de −500 (línea gris).

Cronología de los cruces, medidos sobre la media móvil de 50:

| Hito | Episodio |
|---|---:|
| Supera −400 (sostenido) | 83 |
| Supera −300 (sostenido) | 107 |
| Supera −200 (sostenido) | 142 |
| Supera −150 (sostenido desde 281) | 240 |
| **Supera el umbral de −100** (sostenido desde 548) | **492** |
| Supera −90 (sostenido desde 938) | 583 |

La columna da el episodio en que la media móvil de 50 cruza el nivel por primera vez; el número
entre paréntesis, cuando aparece, es el episodio a partir del cual ya no vuelve a bajar de él.

Las bandas azules que bajan hasta −500 en la primera mitad son los episodios truncados. Desaparecen
por completo después del **episodio 350**: a partir de ahí el agente llega siempre.

## 10. Evaluación del agente

Política puramente codiciosa (ε = 0), **100 episodios** con semillas **10.000–10.099**, disjuntas de
las de entrenamiento (42–1.241). El agente nunca vio esos estados iniciales.

| Métrica | Valor |
|---|---:|
| **Retorno medio** | **−80,96** |
| Mediana | −78,5 |
| Desviación estándar | 12,52 |
| **IC 95 % de la media** | **[−83,41, −78,51]** |
| Mínimo / máximo | −134 / −61 |
| p10 / p25 / p75 / p90 | −93 / −86 / −74 / −69 |
| Pasos medios | 81,96 |
| **Alcanzan la meta** | **100 / 100** |
| Mejores que el umbral −100 | **95 / 100** |

Tres cosas que este bloque demuestra y el entrenamiento por sí solo no:

**Resuelve el entorno según el criterio oficial.** −80,96 queda 19 puntos por encima de −100, y el
intervalo de confianza completo está por encima del umbral. No es un resultado apretado.

**Generaliza.** La evaluación con semillas nuevas da **mejor** resultado que el entrenamiento
(−80,96 frente a −85,68). No es aprendizaje adicional. La explicación evidente es que en
entrenamiento queda ε = 0,05 —una de cada veinte acciones sigue siendo aleatoria—, así que la
medimos: con los pesos finales y las mismas cien semillas de evaluación, treinta repeticiones con
ε = 0,05 dan **−83,52** frente al −80,96 greedy. Quitar el ruido vale **2,56 puntos** (error estándar
0,31), poco más de la mitad de la brecha de 4,72. El resto no lo atribuimos: los estados iniciales del
entrenamiento y los de la evaluación son distintos, y la media de los últimos cien episodios promedia
una red que todavía cambiaba. Lo que sí confirma la comparación es que el agente mejora al retirarle
la exploración, es decir que la política aprendida se sostiene sola.

**Falla poco y falla suave.** Solo 5 de 100 episodios quedan por debajo de −100, y los tres peores
son −134, −124 y −124. Ni un solo episodio truncado: incluso en su peor día, el agente llega.

## 10.1 Robustez: cinco semillas

Un solo entrenamiento demuestra que **este agente** resuelve el entorno. No demuestra que **el
método** lo resuelva: podría ser suerte de la inicialización. Para separar las dos cosas repetimos
la corrida completa con cinco semillas, cambiando **únicamente** `seed`. Los 100 episodios de
evaluación usan las mismas semillas 10.000–10.099 en las cinco, de modo que los cinco agentes se
enfrentan exactamente a los mismos estados iniciales.

| Semilla | Retorno medio | SD | IC 95 % | Mejores que −100 | Terminación | Train últimos 100 |
|---:|---:|---:|---|---:|---:|---:|
| 42 | −80,96 | 12,52 | [−83,41, −78,51] | 95/100 | 100 % | −85,68 |
| 43 | −79,33 | 14,54 | [−82,18, −76,48] | 91/100 | 100 % | −84,85 |
| 44 | −78,14 | 10,60 | [−80,22, −76,06] | 94/100 | 100 % | −84,59 |
| 45 | −82,52 | 29,00 | [−88,20, −76,84] | 95/100 | 100 % | −84,31 |
| 46 | −78,55 | 10,71 | [−80,65, −76,45] | 95/100 | 100 % | −85,75 |

**Agregado sobre 500 episodios de evaluación:**

| Métrica | Valor |
|---|---:|
| Media de las medias por semilla | **−79,90** |
| **Desviación estándar ENTRE semillas** | **1,82** |
| Desviación estándar DENTRO de cada semilla (promedio) | 15,47 |
| Rango de las medias | [−82,52, −78,14], amplitud 4,38 |
| IC 95 % de la media entre semillas (t₄) | [−82,16, −77,64] |
| **Terminaciones verdaderas** | **500 / 500** |
| Episodios mejores que −100 | 470 / 500 (94,0 %) |
| Semillas que superan el umbral oficial | **5 / 5** |

![Cinco semillas](resultados/dqn_5_semillas.png)

**Las cinco semillas son estadísticamente indistinguibles.** La mayor diferencia entre dos de ellas
es 4,38 puntos (semillas 44 y 45), con un error estándar pareado de 3,06: t = 1,43, que no alcanza
para separarlas. Dicho de otro modo, cambiar la inicialización de la red no cambia el resultado de
forma detectable con cien episodios por semilla.

Conviene no comparar directamente el 1,82 de entre semillas con el 15,47 de dentro de cada una,
porque son granularidades distintas: el primero es la desviación de cinco *medias* de 100 episodios y
el segundo la de episodios individuales. El error estándar de la media de una semilla es 15,47/10 ≈
1,55, prácticamente igual al 1,82 observado, que es justo lo que se esperaría si las cinco políticas
fueran equivalentes.

**Dónde está, entonces, la dispersión que queda.** La matriz de 5 semillas × 100 estados iniciales
permite mirarla por sus tres márgenes:

| Dispersión observada | Desviación |
|---|---:|
| entre las cinco medias por semilla | 1,82 |
| entre las cien medias por estado inicial | 7,58 |
| del residuo, semilla × estado | **15,08** |

La lectura cómoda de esa tabla sería «el estado inicial pesa cuatro veces más que la semilla», y es
justo la que no se sostiene. Las tres cifras son descriptivas, no componentes de varianza: **las
medias por estado se dispersan ya solo por el residuo**: con el denominador que le corresponde al
contraste, la desviación residual es 16,95, y al promediarla sobre cinco semillas deja 16,95/√5 =
7,58, exactamente el 7,58 observado. El contraste formal lo confirma: en un ANOVA de dos
vías sobre esta matriz, F(99, 396) = 1,00 con p = 0,49 para el efecto de estado. Con una observación
por celda no podemos separar un efecto de estado del residuo, y lo que los datos indican es que no
hay nada que separar.

Lo que sí queda en pie, y es lo que importa: **la dispersión es interacción**, cuánto le cuesta a
*esta* política *ese* estado concreto. Si viniera del entorno sin más, las cinco políticas encontrarían
difíciles los mismos estados y las medias por estado se dispersarían mucho más de lo que se dispersan;
no lo hacen. Cada entrenamiento produce una política con sus propios puntos débiles, y son distintos
entre sí aunque los promedios coincidan.

Las cinco curvas de la izquierda de la figura son casi indistinguibles a partir del episodio 600, y
las cinco cajas de la derecha están prácticamente alineadas.

**Una anomalía que conviene declarar.** La semilla 45 tiene SD 29,00, entre el doble y el triple que las otras cuatro (12,52 · 14,54 · 10,60 · 10,71). No es
inestabilidad: es **un único episodio de −345** entre cien. Quitándolo, la semilla 45 queda en
−79,87 con SD 11,81, exactamente en línea con el resto. Ese episodio **terminó igual** —no fue
truncado—, así que no es un fracaso sino un estado inicial desfavorable que costó 345 pasos en vez
de 80. Es el peor de los 500 episodios evaluados y aun así el agente llegó a la meta.

## 10.2 Ablación: ¿cuánto pesa distinguir truncación de terminación?

Los §3 y §4 defienden una decisión: que el bootstrap solo se anule cuando el episodio termina de
verdad, nunca cuando se corta por reloj. El argumento es correcto en abstracto, pero un argumento no
dice cuánto pesa. Para averiguarlo entrenamos un segundo agente **idéntico en todo salvo en una
línea**:

```python
# la versión del entregable
buf.push(obs, a, r, nxt, terminated)
# la versión de la ablación
buf.push(obs, a, r, nxt, terminated or truncated)
```

Misma arquitectura, mismos hiperparámetros, misma semilla, mismo orden de llamadas al generador
aleatorio. Lo único distinto es que en la segunda el agente aprende que el estado del paso 500 no
tiene futuro.

### Lo que vimos con una semilla

Con la semilla 42 las dos versiones son indistinguibles en lo que importa: −80,96 contra −80,62 en
evaluación y 100 de 100 terminaciones en ambas. Las trayectorias episodio a episodio se separan desde
el episodio 11, como era de esperar, pero sus medias móviles de 50 corren pegadas de principio a fin,
con una correlación de 0,986.

Buscamos la razón y la primera cuenta que hicimos fue esta: el cambio no afecta a todas las
transiciones de un episodio truncado, sino **solo a la última**, la única cuyo `truncated` vale
`True`. En la corrida de la ablación con semilla 42 hubo 41 episodios truncados, de modo que el error
toca 41 de las 154.568 transiciones recogidas: el **0,027 %**. Parecía suficiente explicación.

Y estaba mal planteada. Lo que decide cuánto daño hace una transición corrupta no es qué fracción
de la experiencia ocupa, sino **en cuántas actualizaciones entra**. Con lotes de 64 tomados al azar del
buffer vivo, recorrimos la corrida paso a paso calculando en cada actualización la probabilidad de
que el lote contuviera al menos una transición contaminada. La respuesta es **3,8 %: unos 5.900 de
los 153.569 minilotes**, ciento cuarenta veces más que la fracción de transiciones.

Esa corrección importa porque cambia la historia. Con el 0,027 % habría que invocar la casualidad
para explicar que una de tres semillas se rompa; con el 3,8 % no hace falta ninguna casualidad. El
error entra en uno de cada veintiséis gradientes, y que dos de tres corridas salgan indemnes deja de
ser un milagro para ser lo que uno esperaría de una perturbación pequeña pero frecuente.

Si hubiéramos parado en la primera corrida, la conclusión habría sido que la distinción da igual.

### Lo que apareció con tres

Por lo mismo que en §10.1 no nos conformamos con una corrida, repetimos la ablación con las semillas
43 y 44.

![Retorno de evaluación por semilla, las dos versiones](resultados/ablacion_truncacion.png)

| | Bootstrap correcto | Ablación |
|---|---:|---:|
| Semilla 42 | −80,96 · 100/100 | −80,62 · 100/100 |
| Semilla 43 | −79,33 · 100/100 | −78,87 · 100/100 |
| Semilla 44 | −78,14 · 100/100 | **−134,05 · 88/100** |
| Media de las tres | **−79,48** | −97,85 |
| Desviación entre semillas | **1,42** | **31,37** |
| Terminaciones, total | **300 / 300** | 288 / 300 |

Dos de las tres son indistinguibles. La tercera no supera el umbral oficial.

Lo que falla en la semilla 44 no es el entrenamiento: su media de los últimos 100 episodios es
−88,39 y no truncó ni una vez en los últimos 200 episodios. La curva de aprendizaje no delata nada.
El fallo aparece en la evaluación greedy, y es localizado: **88 de los 100 estados iniciales nuevos
se resuelven con normalidad —promedian −84,15— y 12 terminan en −500**, sin alcanzar la meta. El
agente aprendió una política que funciona casi siempre y se atasca en una minoría de configuraciones
iniciales que nunca vio.

### Qué nos deja

El error no impide aprender: **quita fiabilidad**. Con el bootstrap correcto, la desviación entre
semillas es de 1,42 y las tres corridas se parecen entre sí tanto como se parecen las cinco de
§10.1. Con el bootstrap roto, esa desviación se multiplica por veintidós y aparece un modo de fallo
que no existía.

Eso encaja con el 3,8 % de minilotes contaminados, pero con una precisión que el promedio esconde y
que conviene decir: **ese 3,8 % no está repartido de forma pareja**. Los episodios truncados se acaban
pronto —el último de la corrida es el 335, a un 48 % del entrenamiento— así que la tasa local cae del
**9,7 % en el primer sexto al 0,7 % en el último**, y la primera mitad concentra el 79 % del daño. Es
decir que la perturbación entra sobre todo mientras la red se está formando, que es cuando un valor
corrompido tiene más recorrido para propagarse. La mayoría de las veces el promedio del lote la
absorbe; cuando no, deja una región del espacio de estados con valores corrompidos, y esa región solo
se descubre al evaluar sobre estados iniciales que el agente no recorrió.

Mantuvimos la versión correcta, y ahora por una razón medida y no solo por semántica. Y nos llevamos
tres advertencias sobre nuestro propio método: una ablación de una sola corrida puede decir lo
contrario de la verdad; una curva de entrenamiento sana puede esconder un agente que falla el 12 % de
las veces; y una cuenta puede ser aritméticamente correcta y aun así medir lo que no es —el 0,027 %
de las transiciones recogidas no era falso, era irrelevante.

La ablación se reproduce con `acrobot ablacion --seed N`. Sus artefactos están en
`resultados/ablacion/`: la historia y la evaluación de cada semilla, más `RESUMEN_ABLACION.json`.

## 10.3 Double DQN: quitar el sesgo del máximo

Las dos secciones anteriores miden nuestro propio agente. Esta lo compara con una variante conocida
del algoritmo, para saber si lo que entregamos se queda corto frente a algo que está a dos líneas de
distancia.

El punto de partida es una debilidad documentada de DQN. El objetivo de Bellman usa
`max_a Q_target(s', a)`: la misma red elige cuál es la mejor acción y estima cuánto vale. Cuando las
estimaciones tienen ruido —y siempre lo tienen—, tomar el máximo de un conjunto de números ruidosos
devuelve de forma sistemática un valor demasiado alto, porque el máximo tiende a caer sobre el error
positivo más grande. Van Hasselt, Guez y Silver (2016) lo llaman sesgo de sobreestimación y proponen
separar las dos funciones:

```python
# DQN: la red objetivo elige y evalúa
next_q = target_net(s2).max(dim=1, keepdim=True).values

# Double DQN: la red en línea elige, la red objetivo evalúa
mejor_accion = q_net(s2).argmax(dim=1, keepdim=True)
next_q = target_net(s2).gather(1, mejor_accion)
```

Entrenamos las dos versiones con las semillas 42, 43 y 44, con la misma red, los mismos
hiperparámetros y el mismo orden de llamadas al generador aleatorio. Y medimos dos cosas distintas:
el desempeño, que es el retorno de los 100 episodios de evaluación, y el sesgo, que es lo que la red
predice para el estado inicial —`max_a Q(s₀,a)`— frente al retorno descontado que el agente obtiene
de verdad a partir de ese mismo estado.

![Desempeño y sesgo de las dos versiones](resultados/double_dqn.png)

| | DQN | Double DQN |
|---|---:|---:|
| Semilla 42 | −80,96 | **−78,42** |
| Semilla 43 | −79,33 | **−78,41** |
| Semilla 44 | −78,14 | **−74,52** |
| Media de las tres | −79,48 | **−77,12** |
| Terminaciones | 300 / 300 | 300 / 300 |
| Valor que la red predice para s₀ | −60,70 | −65,61 |
| Retorno descontado real desde s₀ | −54,67 | −53,65 |
| **Sesgo** (predicho − real) | **−6,03** | **−11,96** |

### Lo que muestra el desempeño

Double DQN mejora en las tres semillas, con una diferencia media de **2,36 puntos**. Tres semillas no
permiten afirmar nada sobre el método, pero los 300 episodios de evaluación **están pareados**: cada
estado inicial se evalúa con las dos versiones. Sobre esos 300 pares, la diferencia media es de
+2,36 con error estándar 1,01, es decir t = 2,33. Dicho con precisión: **este agente Double DQN es
mejor que este agente DQN**, con margen suficiente para no atribuirlo al azar de los estados
iniciales. Si Double DQN es mejor *como método* es otra pregunta, y para esa la unidad es la semilla
de entrenamiento: seguimos teniendo tres, y tres no bastan.

Dentro de esa mejora hay un detalle que la media esconde. En la semilla 43, Double DQN **pierde en
más estados de los que gana** —44 contra 55— y aun así termina con mejor promedio, porque gana en la
cola. No mejora el caso típico: recorta el caso malo. Es lo que cabría esperar de una corrección que
actúa sobre la sobreestimación.

### Lo que muestra el sesgo, y lo que no

Nuestra primera medición fue en el estado inicial, y dio un resultado desconcertante: **DQN se quedaba
corto por 6,03 puntos y Double DQN por 11,96**. Ninguno sobreestimaba, y el que existe para corregir
el optimismo era el más pesimista de los dos.

Antes de escribir una explicación, medimos el sesgo en **todos los estados que el agente visita**, no
solo en el primero, agrupándolos por cuántos pasos le faltan para llegar a la meta. Y publicamos los
dos números de los que sale la resta, porque la resta sola engaña:

| Pasos que faltan | Predicho | Real | Sesgo | n | Cota del real |
|---|---:|---:|---:|---:|---:|
| 0 – 19 | −15,16 | −8,51 | −6,65 | 5.700 | — |
| 20 – 49 | −33,02 | −28,32 | −4,70 | 9.000 | −17,38 |
| 50 – 79 | −49,92 | −46,16 | −3,76 | 7.774 | −38,89 |
| 80 – 89 | −55,22 | −56,51 | **+1,29** | 1.007 | −54,80 |
| 90 – 99 | −55,06 | −60,52 | **+5,46** | 390 | −59,12 |
| 100 o más | −56,44 | −65,97 | **+9,53** | 272 | −63,03 |

*DQN, media de las tres semillas. La cota es el mayor retorno descontado posible con esos pasos por
delante: −(1 − γⁿ⁻¹)/(1 − γ).*

**El sesgo cambia de signo, y la columna del valor predicho dice por qué.** De los 50 pasos en
adelante, lo que la red predice se aplana —−49,9, −55,2, −55,1, −56,4— mientras el retorno real sigue
cayendo: −46,2, −56,5, −60,5, −66,0. El cruce no lo produce una predicción que suba, sino una
predicción casi constante que el retorno real termina de atravesar. La red comprime el rango de la
función de valor: distingue bien los estados cercanos a la meta y deja de distinguir los lejanos.

**Y aquí está el límite de esta medición, que conviene decir antes que la interpretación.** El
retorno real está acotado por los pasos restantes: con cien pasos por delante no puede ser mejor que
−63,03. El valor mínimo que nuestras redes predicen sobre los estados visitados es −62,43, −62,16 y
−58,54 según la semilla. **En ese tramo el sesgo positivo estaba forzado por aritmética**: cualquier
red cuyo suelo quede por encima de la cota lo habría producido, con o sin el operador máximo. Así que
esta medida no identifica sobreestimación del máximo, y no la vamos a leer como tal.

Lo que sí sostiene: **el sesgo no es un número, tiene estructura**, y medirlo en el estado inicial
—que es lo cómodo— nos dio la respuesta contraria a la que da medirlo en toda la trayectoria.

Dos advertencias sobre la tabla misma. Los tramos lejanos reúnen pocos estados —272 y 206 sobre
decenas de miles—, así que sus valores son ruidosos. Y la fila «100 o más» de Double DQN promedia
**dos** semillas y no tres, porque en la semilla 44 el agente nunca se quedó tan lejos de la meta;
sus dos sumandos son +2,92 y +19,57, que difieren en un factor de siete. Comparar ese +11,25 con el
+9,53 de DQN sería comparar medias de distinto tamaño, y no lo hacemos.

### Lo que esta medición no identifica

La cantidad que medimos es una suma: el optimismo del máximo, el retraso de la red objetivo, el error
de aproximación, la convergencia incompleta y —como acabamos de ver— la cota que los propios pasos
restantes imponen al retorno real. Con un número por método no podemos separarlos.

La medición que sí aislaría el optimismo cuesta cinco líneas y no requiere reentrenar: sobre el mismo
lote de estados y **la misma red objetivo**, comparar `max_a Q_target(s,a)` contra
`Q_target(s, argmax_a Q_online(s,a))`. Esa diferencia es el sesgo de selección y nada más. No la
hicimos porque guardamos solo la red en línea al final del entrenamiento, y la red objetivo del
momento de la actualización no quedó registrada. Es lo primero que añadiríamos.

### Qué hacemos con esto

Mantenemos DQN como el agente del entregable. La mejora de Double DQN es consistente pero pequeña, y
el trabajo pedía implementar y entrenar un DQN, no elegir la mejor variante.

Lo que esta sección añade no es un agente distinto sino una manera de mirar: el sesgo de los valores
Q no se resume en una cifra, y medirlo en el estado inicial —que es lo cómodo— nos había dado la
conclusión contraria a la que sostienen los datos.

La comparación se reproduce con `acrobot double --seed N` y la medición del sesgo con `acrobot
sesgo`. Los artefactos están en `resultados/double_dqn/` y los pesos de las seis corridas en
`resultados/pesos/`, de modo que el sesgo se puede recalcular sin volver a entrenar.

## 10.4 El baseline tabular: ¿hacía falta la red?

El §6 justifica la red con un argumento: la función Q depende de productos entre ángulos y
velocidades, y una tabla no puede representar eso sin discretizar. Es el mismo tipo de argumento sin
medir que ya nos falló dos veces, así que lo medimos.

Q-learning tabular necesita estados discretos. La observación que entrega Gymnasium tiene seis
componentes, pero el estado físico real son cuatro: dos ángulos y dos velocidades. Los senos y
cosenos son una reparametrización que le conviene a una red y no a una tabla, así que recuperamos
los ángulos con `atan2` y discretizamos las cuatro variables. Todo lo demás es idéntico al DQN:
1.200 episodios, γ = 0,99, el mismo decaimiento de ε y las mismas semillas de evaluación.

Una advertencia antes de comparar: *el mismo número de episodios no es el mismo número de pasos*.
Como el tabular tarda más en llegar a la meta, sus episodios son más largos, y esos 1.200 episodios
le costaron **388.550 pasos de entorno** contra los **158.451** del DQN. La comparación que sigue le
da al tabular 2,5 veces más interacción, no menos, lo que hace la conclusión más fuerte y no más
débil. Con un matiz que también conviene decir: como ε decae por pasos y no por episodios, esos pasos
de más son casi todos greedy: el decaimiento de ε cubre el 15 % de los pasos del tabular contra el
38 % de los del DQN, y el tabular toca el piso de ε en su episodio 129. La ventaja que le damos es de
explotación, no de exploración.

**Le dimos una oportunidad honesta.** Un baseline mal ajustado que pierde no demuestra nada, así que
antes de comparar probamos doce configuraciones —cuatro rejillas de discretización por tres tasas de
aprendizaje— y nos quedamos con la mejor: rejilla 12×12×10×10, α = 0,1. Dos salvedades sobre esa
selección, que nos costó descubrir: el barrido corrió con **una sola semilla**, y evaluó cada
configuración sobre las mismas cien semillas con las que después reportamos el resultado. Más abajo
comprobamos cuáles de sus diferencias sobreviven a repetirlo.

![Tabular frente a DQN](resultados/tabular.png)

| | DQN | Q-learning tabular |
|---|---:|---:|
| Semilla 42 | −80,96 | −166,21 |
| Semilla 43 | −79,33 | −205,31 |
| Semilla 44 | −78,14 | −214,43 |
| Media de las tres | **−79,48** | **−195,32** |
| Terminaciones | 300 / 300 | **300 / 300** |
| Supera el umbral de −100 | 3 de 3 semillas | 0 de 3 |

### Lo que muestra la comparación

**El tabular resuelve Acrobot.** Alcanza la meta en los 300 episodios de evaluación, igual que el
DQN, y eso merece decirse antes que nada: el entorno no exige una red para ser resuelto. La
discretización de cuatro variables basta para aprender una política que funciona siempre.

**Y aun así le cuesta dos veces y media más llegar.** −195,32 contra −79,48, y ninguna de las tres semillas
alcanza el umbral de −100 con el que Gymnasium declara resuelto el entorno. Ninguna de las doce
configuraciones del barrido llegó tampoco: la mejor se quedó en −166,21.

**La tabla se queda mayormente vacía, y eso hay que contarlo con cuidado.** El agente visita
**1.566 celdas de media de las 14.400**, un 10,9 %. Pero ese denominador está inflado: buena parte
de la rejilla corresponde a combinaciones de velocidad que el sistema físico nunca alcanza. Mirando
qué bins se tocan en las tres corridas, de los 10 bins de ω₁ solo aparecen 7 y de los 10 de ω₂ solo
6, de modo que la tabla *alcanzable* tiene 6.048 celdas y la cobertura real ronda el **26 %**. La
cifra honesta es esa, no el 10,9 %.

**Nuestra primera explicación fue la cobertura, y no se sostuvo.** Escribimos que el tabular pierde
porque la tabla queda vacía: ante una celda en cero la política elige por desempate, sin información.
Es una explicación cómoda y el propio barrido parecía respaldarla. Pero el barrido tenía una sola
semilla, así que repetimos las cuatro rejillas con α = 0,1 en las semillas 42, 43 y 44 —las mismas
del resto de §10— antes de apoyarnos en él:

| Rejilla (α = 0,1) | Celdas | Cobertura | Visitas por celda | Retorno medio | SD entre semillas | Terminaciones |
|---|---:|---:|---:|---:|---:|---:|
| 8×8×6×6 | 2.304 | 18,1 % | 890 | −192,90 | 12,94 | 299 / 300 |
| **12×12×10×10** | **14.400** | **10,9 %** | **252** | **−195,32** | **25,62** | **300 / 300** |
| 16×16×12×12 | 36.864 | 8,7 % | 137 | −249,65 | 26,57 | 285 / 300 |
| 20×20×14×14 | 78.400 | 7,1 % | 85 | −330,57 | 45,20 | 252 / 300 |

*`resultados/tabular/rejillas_tres_semillas.json`. La corrida de la semilla 42 reproduce exactamente
las cifras del barrido original, que es lo que valida el procedimiento.*

Con tres semillas, la diferencia que habíamos usado como prueba **desaparece**. 8×6 promedia −192,90
y 12×10 promedia −195,32: 2,4 puntos de distancia frente a desviaciones entre semillas de 12,94 y
25,62, y en la semilla 44 la rejilla gruesa gana por 30 puntos. Con una sola semilla habíamos visto
19,8 puntos de ventaja para 12×10 y los habíamos leído como si significaran algo.

Eso deja sin sostén la explicación por cobertura: la rejilla que cubre 1,7 veces más de su tabla no
rinde peor, rinde igual. Y no la reemplazamos por otra. Se podría decir que el límite son las visitas
por celda —890, 251, 137, 85, que sí caen al afinar—, pero entre las dos rejillas donde esa cifra se
divide por tres y medio el retorno no cambia de forma medible, así que la explicación no se apoya en
nada que hayamos medido. **Cuál es el mecanismo no lo sabemos**, y preferimos dejarlo escrito así
antes que rellenarlo con una historia plausible.

**Lo que sí sobrevive a las tres semillas** es lo que pasa al afinar. Por encima de 12×10 el retorno
empeora de forma consistente y, más revelador, el tabular **deja de resolver el entorno**: 285 y 252
terminaciones de 300, contra 299 y 300 de las dos rejillas gruesas. Eso cierra la salida fácil —«con
más resolución la tabla funcionaría»—: con este presupuesto, afinarla la empeora.

**Tampoco es falta de presupuesto.** Corrimos la mejor configuración con **6.000 episodios** —cinco
veces más, 1,59 millones de pasos de entorno— y el resultado no mejora: −185,00 contra los −166,21
que esa misma semilla y rejilla dieron con 1.200 episodios, con la cobertura subiendo solo del 26 %
al 32 % de la tabla alcanzable. Es una sola corrida y la diferencia cabe dentro de la desviación
entre semillas, así que lo que sostiene no es que empeore, sino que **quintuplicar la experiencia no
acerca el tabular al umbral**. La política greedy concentra las visitas en la misma región, así que
las celdas nuevas son las que menos importan.

### Qué nos deja

La red no era indispensable para resolver el entorno, y sí lo fue para alcanzar el criterio de
resolución con este presupuesto. La distinción importa porque separa *poder resolver* de *resolver
bien*, y pone un número a cada uno.

Vale además como aviso sobre la comparación misma, y sobre nosotros. La rejilla más fina —20×20×14×14—
es la que peor rinde, y es la que un lector apurado habría elegido por ser la más detallada: un método
clásico no se evalúa con su configuración más ambiciosa sino con la que su presupuesto de datos puede
sostener. Pero el aviso de verdad es el otro: escribimos una explicación causal —la cobertura— apoyada
en un barrido de una sola semilla, y solo se cayó cuando lo repetimos. Es el mismo error que §10.2 nos
había enseñado a no cometer, cometido otra vez tres secciones después.

La comparación se reproduce con `acrobot tabular --seed N` y sus artefactos están en
`resultados/tabular/`: las tres tablas entrenadas, sus evaluaciones, el barrido completo, el coste por
celda de cada rejilla, la repetición de las cuatro rejillas con tres semillas y la corrida de 6.000
episodios.

## 11. Conclusiones

1. **Un DQN estándar resuelve Acrobot-v1** sin extensiones ni técnicas especiales de exploración:
   −80,96 de retorno medio en evaluación contra un umbral oficial de −100 y un suelo aleatorio de
   −500. En §10.3, añadir Double DQN mejora el resultado en algo más de dos puntos —en las tres
   semillas, aunque tres pares no basten para acotar la mejora—, de modo que la versión simple no se
   queda corta en este entorno.
2. **Distinguir `truncated` de `terminated` no decide si el agente aprende, sino si se puede
   confiar en él.** Entrenamos las dos versiones con tres semillas (§10.2): dos pares son
   indistinguibles y en el tercero la versión rota cae a −134,05 y falla 12 de 100 episodios de
   evaluación. La desviación entre semillas pasa de 1,42 a 31,37. El cambio afecta solo al último
   paso de cada episodio truncado, y esa transición entra en el 3,8 % de los minilotes, lo que explica
   las dos caras: casi siempre se diluye, y cuando no, el fallo no se ve en la curva de
   entrenamiento.
3. **La loss no es una medida de desempeño.** En esta corrida se multiplicó por 77 mientras el
   agente pasaba de −290 a −85, por las razones que explica §8. Las conclusiones se sostienen en el
   retorno de evaluación, la tasa de terminación y la curva.
4. **La estabilización se lee mejor en la varianza que en la media.** La media pasa de −89,38 a
   −85,66 entre los bloques 601–800 y 1001–1200 —una mejora modesta—, pero la SD cae de 22,38 a
   13,77. Lo que el agente gana en la segunda mitad del entrenamiento es sobre todo consistencia.
5. **El resultado no depende de la inicialización, pero las políticas no son intercambiables.**
   Cinco corridas con semilla distinta superan el umbral oficial, con 500 de 500 terminaciones
   verdaderas y una desviación entre las cinco medias de 1,82, que es lo que se esperaría por puro
   error de muestreo si las cinco políticas fueran equivalentes (§10.1). Eso permite decir *el método
   resuelve Acrobot* y no solo *este agente lo resuelve*. Lo que la descomposición de varianza añade
   es que la dispersión que queda **no** viene del entorno sin más: la de las medias por estado
   inicial (7,58) es la que produciría el residuo por sí solo, y el contraste formal no distingue un
   efecto de estado de cero (F(99, 396) = 1,00). Lo que queda es interacción: cada entrenamiento
   produce una política con sus propios puntos débiles, y son distintos entre sí aunque los promedios
   coincidan.
6. **El sesgo de los valores Q no es un número: tiene estructura.** Double DQN gana 2,36 puntos de
   media en las tres semillas (§10.3). Al medir el sesgo en toda la trayectoria y no solo en el
   estado inicial, la red resulta comprimir el rango de la función de valor: distingue bien los estados
   cercanos a la meta y se aplana en los lejanos, de modo que la diferencia con el retorno real cambia
   de signo con la distancia.
   No podemos atribuir ese cambio al optimismo del máximo, porque en los tramos lejanos la cota
   aritmética del retorno real fuerza el signo positivo. Lo que sí sostiene la medición es que medir
   en un solo punto —el estado inicial— daba la respuesta contraria.
7. **La red no era indispensable para resolver Acrobot, y sí para resolverlo bien.** Q-learning
   tabular alcanza la meta en los 300 episodios de evaluación —gastando dos veces y media más pasos de
   entorno que el DQN— pero promedia −195,32 frente a −79,48 y ninguna semilla llega al umbral
   oficial (§10.4). No sabemos nombrar el mecanismo: al repetir las cuatro rejillas con tres
   semillas, las dos más gruesas se vuelven indistinguibles entre sí pese a que una cubre 1,7 veces
   más tabla que la otra, así que la cobertura no explica la diferencia que nos importaba. Lo que sí sostienen las tres semillas es que el camino de salida obvio
   está cerrado: afinar la rejilla empeora el retorno y hace que el tabular deje de resolver el
   entorno, y quintuplicar los episodios no lo acerca al umbral.
8. **No afirmamos optimalidad ni convergencia en sentido estricto.** No hay cota inferior conocida
   para Acrobot con la que comparar el −60 del mejor episodio, y la media móvil sigue oscilando entre
   −90,9 y −83,2 en los últimos 300 episodios. Lo demostrado es *estabilización*, no convergencia.

**Lo que queda fuera.** El replay es uniforme y no probamos agregación por vecindad ni *tile
coding*, que son las representaciones intermedias entre una tabla y una red. Los dos experimentos de
§10.2 y §10.3 corrieron con tres semillas cada uno: alcanza para ver la dirección de los efectos, no
para acotarlos con un intervalo. Dueling DQN queda sin probar; con solo tres acciones, el margen que
cabe esperar de factorizar valor y ventaja es estrecho.

## 12. Reflexión sobre los resultados

Nuestra primera lectura de la curva fue la más obvia: mirar cuánto mejoraba la media del retorno.
Nos quedamos cortos. Hacia la mitad del entrenamiento el agente ya estaba cerca de su nivel final, y sin
embargo la dispersión seguía cayendo. Entre los bloques 601–800 y 1001–1200 la media apenas mejora
de −89,38 a −85,66, pero la desviación estándar baja de 22,38 a 13,77, y en los primeros 200
episodios era 148,30. La segunda mitad del entrenamiento no fue una repetición improductiva: el
agente pasó de **saber resolver Acrobot** a **resolverlo sin episodios catastróficos**. Desde ahí
empezamos a leer las curvas con dos preguntas separadas, cuánto mejora el desempeño y cuánto
disminuye su inestabilidad, y la segunda resultó ser la que más información daba.

La loss nos obligó a una distinción más incómoda. Creció mientras el retorno mejoraba, de 0,0062 a
0,4807, y viniendo del aprendizaje supervisado una loss creciente se lee casi automáticamente como
un problema. El §8 explica por qué aquí no lo es. Lo que nos llevamos no es esa excepción
memorizada sino el hábito que nos dejó: **una métrica no se interpreta por su nombre sino por la
cantidad que realmente mide.** Ese hábito es el que después nos hizo desconfiar de nuestras propias
afirmaciones.

La repetición con cinco semillas cambió la pregunta de fondo. Con una sola corrida podíamos afirmar
que *ese* agente había aprendido; con cinco, que el procedimiento es reproducible. Conviene precisar
qué varía entre ellas: la inicialización de los pesos y el ruido de la exploración, no la
distribución de estados iniciales, que se solapa casi por completo porque cada episodio se siembra
con `seed + episodio`. Eso no demuestra optimalidad ni convergencia universal, pero sí que el
resultado no depende de una inicialización afortunada.

Los dos experimentos de control cambiaron algo más de fondo, que es cómo leemos un argumento. En
§10.2 rompimos a propósito la decisión que el propio documento declaraba determinante, y resultó que
el agente aprende igual: lo que el error se lleva no es el aprendizaje sino la fiabilidad, y solo se
ve al evaluar sobre estados que el agente nunca recorrió. Y en §10.4 volvimos a hacerlo: escribimos que el tabular pierde por falta de cobertura, apoyados en
un barrido de una sola semilla, y la explicación se cayó cuando lo repetimos con tres. Esa sección
termina sin mecanismo, que es lo que los datos permiten.

En §10.3 fuimos a medir el sesgo que Double DQN existe
para corregir y nos equivocamos dos veces seguidas: primero midiéndolo donde era cómodo —el estado
inicial— y concluyendo que no existía, y después leyendo el cambio de signo como sobreestimación,
cuando en los tramos lejanos ese signo estaba forzado por la cota del retorno real. En los dos casos el mecanismo que
describe la teoría es real; lo que la teoría no dice es cuánto pesa en *este* entorno, con *esta*
configuración y *este* presupuesto de episodios. Esa distancia entre el mecanismo y su magnitud es,
para nosotros, el resultado más útil del trabajo.

El contraste con MountainCar fue el que más nos ordenó las ideas. Es el mismo algoritmo y casi los
mismos hiperparámetros, y el resultado es opuesto: allí el DQN se atasca, aquí resuelve el entorno.
Lo que separa los dos casos es si la exploración inicial llega a tropezar con la meta alguna vez.
En Acrobot, diez de los primeros cincuenta episodios llegaron, y el primero fue el episodio 11: el
buffer tuvo trayectorias exitosas desde el arranque. Un buffer sin éxitos no contiene información
sobre cómo tener éxito, por muchas transiciones que acumule. Eso reordenó nuestra manera de estimar
la dificultad de un entorno: lo que pesa es si el azar tropieza con la recompensa, más que lo
complicado que se vea el problema.

## 13. Reflexión sobre lo que más costó

Lo que más nos costó fue entender que `terminated` y `truncated` detienen el mismo bucle de Python
y significan cosas distintas para Bellman. Un episodio truncado en el paso 500 no implica que ese
estado tenga valor futuro cero; solo significa que un límite externo interrumpió la experiencia. La
lámina del Algorithm 1 que se proyectó en clase dice «si el episodio termina», y el pseudocódigo
original no necesitaba distinguir los dos casos porque Atari no truncaba por reloj; Gymnasium sí, y
ahí es donde la traducción exige criterio.

Lo difícil vino después, y no fue entender la distinción sino medir cuánto pesaba. Habíamos escrito
que era la decisión determinante del proyecto con un razonamiento que nos parecía sólido: si la
falsedad se propaga hacia atrás por toda la trayectoria, el daño tiene que ser grande. Ese
razonamiento saltaba de «el error existe» a «el error importa» sin pasar por la pregunta que decide
el asunto, que es cuánta experiencia toca.

Entrenar la versión equivocada nos corrigió tres veces. La primera corrida salió indistinguible de la
buena y estuvimos a punto de escribir que la distinción daba igual. Dos semillas más mostraron que
no: en una de las tres el agente falla 12 de 100 episodios de evaluación sin que la curva de
entrenamiento lo delate. Y la explicación que habíamos escrito para el primer resultado —41
transiciones contaminadas de 154.568, una proporción ínfima— resultó estar contando lo que no era:
lo que importa no es qué fracción de la experiencia ocupan sino en cuántos minilotes entran, y eso es el 3,8 %,
ciento cuarenta veces más.

Las tres veces el error fue el mismo: dar por terminada una medición antes de haber comprobado que
medía lo que la conclusión necesitaba. Exagerar el efecto desde un argumento, negarlo desde una sola
corrida, y explicarlo con una cuenta correcta sobre la magnitud equivocada.

La segunda dificultad fue no corregir el entrenamiento demasiado pronto. Al ver la loss crecer, la
reacción intuitiva era bajar el learning rate o dar por hecha una inestabilidad. Habría sido
intervenir sobre una corrida que estaba mejorando, guiados por una lectura equivocada de la
métrica. Antes de tocar un hiperparámetro tuvimos que volver a la ecuación, separar retorno de
error de regresión y preguntarnos qué medía cada serie. Esa disciplina —entender la métrica antes
de modificar nada— nos parece más transferible que cualquier valor concreto de `lr` o `batch_size`.

La tercera fue no dejar que el trabajo se partiera en dos versiones del mismo experimento. El
notebook existe para leerse de arriba abajo, con cada pieza a la vista en el orden en que se
explica, y por eso lleva el código escrito celda a celda; el paquete existe para ejecutarse desde la
terminal. Son dos presentaciones del mismo algoritmo, y ahí estaba el riesgo: que una se corrigiera
y la otra no.

Lo que hicimos fue correr las dos por separado con la misma semilla y comparar las salidas, no el
código: los 1.200 retornos de entrenamiento, los 157.452 valores de loss, los 157.452 de norma del
gradiente y los 100 retornos de evaluación coincidieron elemento a elemento. Esa comparación la
hicimos con las dos historias completas en memoria y **no queda reproducible desde el repositorio**,
porque el notebook solo guarda en sus salidas las cifras que imprime. Lo que `acrobot verify` sí
comprueba, cada vez que se ejecuta, es que los escalares que el notebook dejó impresos —retorno medio,
desviación, pasos medios, media de los últimos cien, modo y número de episodios— coincidan con los
archivos de `resultados/`. Es menos que la comparación original, y es lo que se puede sostener sin
volver a correr las dos vías.

---

## Reproducir: dos vías para lo mismo

El trabajo se puede recorrer de dos maneras y **las dos producen exactamente los mismos números**. El
CLI importa el algoritmo de `src/acrobot_dqn/dqn_acrobot_v3.py` sin tocarlo. El notebook lo reescribe
celda a celda a propósito, para que pueda leerse en el orden en que se explica; §13 cuenta cómo
comprobamos que las dos vías coinciden elemento a elemento, y qué parte de esa comprobación queda
automatizada en `acrobot verify`.

### Prerequisitos

**Python ≥ 3.11.** El proyecto se desarrolló con Python 3.11 sobre Linux y se ha verificado también
con Python 3.12 (Anaconda, Windows, GPU). No requiere GPU: la corrida canónica corrió en CPU; si hay
GPU disponible el código la detecta y la usa automáticamente.

Las cuatro dependencias son `gymnasium[classic-control]`, `torch`, `numpy` y `matplotlib`. Se
instalan solas con `pip install -e .`; no hace falta instalar nada a mano.

### Instalación

#### Linux / macOS

```bash
git clone <url-del-repositorio>
cd rl_acrobot
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

#### Windows — con Anaconda (recomendado)

Si ya tienes un entorno conda con PyTorch y Gymnasium:

```bat
:: Desde Anaconda Prompt
conda activate <nombre-del-entorno>
cd D:\ruta\al\repositorio\rl_acrobot
pip install -e .
acrobot version        :: confirma la instalación y muestra el dispositivo
```

Si no tienes un entorno, créalo desde Anaconda Prompt:

```bat
conda create -n rl_acrobot python=3.12
conda activate rl_acrobot
cd D:\ruta\al\repositorio\rl_acrobot
pip install -e .
acrobot version
```

#### Windows — con PowerShell sin Anaconda en el PATH

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
$env:PYTHONUTF8 = "1"   # evita errores de codificación con caracteres especiales
acrobot version
```

> Si usas un entorno conda cuyo `Scripts\` no está en el PATH del sistema, añádelo en la sesión
> actual antes de llamar a `acrobot`:
> ```powershell
> $env:PYTHONUTF8 = "1"
> $env:PATH = "C:\Users\<usuario>\anaconda3\envs\<entorno>\Scripts;" + $env:PATH
> ```
> Para que sea permanente, activa el entorno desde Anaconda Prompt (`conda activate <entorno>`) en
> lugar de PowerShell.

### Vía A — el notebook, para leer el razonamiento

```bash
pip install jupyterlab          # si aún no está instalado
jupyter lab notebooks/SAR_U3_Taller2_DQN_EJECUTADO.ipynb
```

`notebooks/SAR_U3_Taller2_DQN_EJECUTADO.ipynb` **ya viene ejecutado** con la corrida de 1.200
episodios y todas sus salidas guardadas: se lee sin correr nada y sin elegir kernel. Alterna
markdown, código, salida e interpretación, así que es la vía para entender *por qué* cada decisión.

### Vía B — la línea de comandos, para usarlo

Una vez instalado el paquete (`pip install -e .`), el comando `acrobot` queda disponible:

```bash
acrobot version                       # versiones y dispositivo (CPU o GPU)
acrobot inspect                       # audita Acrobot-v1: espacios, recompensa, umbral oficial
acrobot train --mode smoke            # 10 episodios, ~20 s: comprueba el pipeline
acrobot train --mode full --seed 42   # la corrida canónica de 1.200 episodios
acrobot evaluate --episodes 100       # evaluación greedy con semillas 10000-10099
acrobot plot --output curva.png       # regenera la curva desde la historia guardada
acrobot results                       # los resultados verificados de las cinco semillas
acrobot verify                        # comprueba el README contra los artefactos
acrobot rangos                        # rangos de observación con las dos políticas
acrobot ablacion                      # entrena la variante de §10.2 y la compara
acrobot ablacion --solo-figura        # regenera su figura sin reentrenar
acrobot double                        # entrena Double DQN y lo compara (§10.3)
acrobot sesgo                         # el sesgo por distancia a la meta, sin reentrenar
acrobot minilotes                     # en cuántos minilotes entra el error de §10.2
acrobot ambiente                      # renderiza un episodio y rehace la figura de §1
acrobot tabular                       # entrena Q-learning tabular y lo compara (§10.4)
acrobot estado                        # rehace la figura del estado y las acciones (§2)
acrobot demo                          # abre ventana pygame con el agente jugando en tiempo real
```

`acrobot train` arranca en modo `smoke` **a propósito**: verificar el pipeline antes de gastar una
corrida completa es más barato que descubrir un fallo con el entrenamiento ya avanzado.

#### `acrobot demo` — ver el agente en tiempo real

Abre una ventana de pygame con el péndulo animado y el agente aplicando su política entrenada.
Requiere los pesos en `resultados/dqn_acrobot_1200ep_state_dict.pt` (incluidos en el repositorio).

```bash
acrobot demo                             # 3 episodios · pausa 0,02 s/frame (por defecto)
acrobot demo --episodios 5               # cinco episodios seguidos
acrobot demo --pausa 0.05               # más lento, fácil de seguir visualmente
acrobot demo --pausa 0                  # máxima velocidad sin pausa entre frames
acrobot demo --seed 10042               # condición inicial distinta
acrobot demo --episodios 1 --seed 10010 # un episodio concreto
```

En consola muestra el retorno y el número de pasos al terminar cada episodio:

```
dispositivo : cuda
pesos       : dqn_acrobot_1200ep_state_dict.pt
episodios   : 3  |  semilla base : 10000
────────────────────────────────────────────────
  ep  1  |  retorno    -78  |  pasos  78  |  META ✓
  ep  2  |  retorno    -82  |  pasos  82  |  META ✓
  ep  3  |  retorno    -75  |  pasos  75  |  META ✓
────────────────────────────────────────────────
```

**Para salir:**
- **Ctrl + C** en la terminal — detiene inmediatamente y cierra la ventana.
- Cerrar la ventana con el botón **×** — el script termina al acabar el episodio actual.

### Qué vía usar

| Para… | Vía |
|---|---|
| seguir el razonamiento y ver los resultados sin ejecutar nada | notebook |
| comprobar que las cifras del README son ciertas | `acrobot verify` |
| auditar el entorno de forma independiente | `acrobot inspect` |
| reproducir el entrenamiento | `acrobot train --mode full` |
| **ver el agente jugando en tiempo real** | **`acrobot demo`** |
| reutilizar el agente desde otro script | `from acrobot_dqn import train_dqn, evaluate` |

**Entorno de la corrida reportada:** Python 3.11.15 · PyTorch 2.14.0 · Gymnasium 1.3.0 ·
NumPy 2.4.4 · CPU (x86_64, Linux).
Verificado también con: Python 3.12 · PyTorch 2.11.0+cu128 · GPU (Windows, CUDA 12.8).

> **Nota de reproducibilidad.** El código elige dispositivo automáticamente. En Mac con Apple
> Silicon seleccionará **mps**: `seed_everything` siembra `random`, `numpy`, `torch` y `torch.cuda`,
> pero no `torch.mps`, y las operaciones de punto flotante en mps y cpu no son bit a bit idénticas.
> Los números de una corrida en mps serán parecidos pero **no idénticos** a los de arriba con la
> misma semilla. Para reproducción exacta, forzar `DEVICE = torch.device("cpu")`.

## Contenido del repositorio

```
README.md                                    este documento
requirements.txt                             4 dependencias
pyproject.toml                               empaquetado y registro del comando `acrobot`
src/acrobot_dqn/dqn_acrobot_v3.py            implementación completa — el archivo verificado
src/acrobot_dqn/cli.py                       los dieciséis subcomandos
src/acrobot_dqn/ablacion.py                  la variante de §10.2, una línea distinta
src/acrobot_dqn/double_dqn.py                Double DQN, la variante de §10.3
src/acrobot_dqn/figura_ablacion.py           regenera la figura de §10.2 desde los artefactos
src/acrobot_dqn/sesgo.py                     mide el sesgo de los valores Q (§10.3)
src/acrobot_dqn/minilotes.py                 cuenta los minilotes que toca el error de §10.2
src/acrobot_dqn/tabular.py                   Q-learning tabular, la alternativa de §10.4
src/acrobot_dqn/figura_ambiente.py           renderiza el episodio de §1
src/acrobot_dqn/figura_estado.py             dibuja el estado y las acciones de §2
src/acrobot_dqn/figura_double.py             regenera la figura de §10.3 desde los artefactos
src/acrobot_dqn/figura_tabular.py            regenera la figura de §10.4 desde los artefactos
tests/test_algoritmo.py                      cuatro comprobaciones del algoritmo, sin entrenar
src/acrobot_dqn/__init__.py                  API del paquete
verificar_resultados.py                      las 133 comprobaciones de `acrobot verify`
notebooks/SAR_U3_Taller2_DQN_EJECUTADO.ipynb notebook con la corrida y sus salidas
resultados/
  dqn_learning_curve_1200ep.png              la curva de la corrida canónica (semilla 42)
  dqn_5_semillas.png                         las cinco curvas y los cinco boxplots de evaluación
  semillas/RESUMEN_5_SEMILLAS.json           el agregado de robustez
  semillas/resultado_seed_4[3-6].json        una corrida completa por semilla
  evaluation_metrics_100ep.json              los 100 retornos de evaluación
  training_history_resumen.json              historia por episodio: retornos, longitudes, terminales
  training_history_1200ep.json.gz            historia cruda: las 157.452 losses y grad norms
  dqn_acrobot_1200ep_state_dict.pt           pesos entrenados
  full_summary.json                          todas las métricas de este README, en un archivo
  rangos_observacion.json                    los rangos de §2 con las dos políticas
  ambiente_acrobot.png                       los seis fotogramas de §1 y la altura paso a paso
  estado_y_acciones.png                      el estado y las acciones de §2, sobre el mecanismo
  tabular.png                                DQN frente a Q-learning tabular (§10.4)
  ablacion_truncacion.png                    la evaluación de las dos versiones, tres semillas
  ablacion/ablacion_seed_4[2-4].json         la corrida de la ablación, una por semilla
  ablacion/ablacion_seed_4[2-4].json.gz      con las losses y las normas de gradiente
  ablacion/evaluacion_seed_4[2-4].json       su evaluación sobre 100 episodios
  ablacion/RESUMEN_ABLACION.json             el agregado de §10.2
  double_dqn.png                             desempeño y sesgo de las dos versiones de §10.3
  double_dqn/double_dqn_seed_4[2-4].json     la corrida de Double DQN, una por semilla
  double_dqn/evaluacion_seed_4[2-4].json     su evaluación sobre 100 episodios
  double_dqn/medicion_sesgo.json             el sesgo por distancia a la meta, seis corridas
  pesos/                                     los pesos de las otras cinco corridas
  tabular/tabla_seed_4[2-4].npy              las tres tablas Q entrenadas
  tabular/evaluacion_seed_4[2-4].json        su evaluación sobre 100 episodios
  tabular/barrido_hiperparametros.json       las doce configuraciones probadas
  tabular/rejillas_tres_semillas.json        las cuatro rejillas repetidas con tres semillas
  tabular/RESUMEN_TABULAR.json               el agregado de §10.4
  double_dqn/RESUMEN_DOUBLE_DQN.json         el agregado de §10.3
docs/GUIA_DE_LECTURA.md                      dónde está cada contenido que pide la actividad
```

## Referencias

- Mnih, V. et al. (2013). *Playing Atari with Deep Reinforcement Learning*. arXiv:1312.5602
- Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning*. Nature 518,
  529–533. — es el que introduce la red objetivo y el Algorithm 1 que seguimos en §4.
- van Hasselt, H., Guez, A. & Silver, D. (2016). *Deep Reinforcement Learning with Double
  Q-learning*. arXiv:1509.06461
- Wang, Z. et al. (2016). *Dueling Network Architectures for Deep Reinforcement Learning*.
  arXiv:1511.06581
- Lapan, M. (2020). *Deep Reinforcement Learning Hands-On* (2.ª ed.). Packt Publishing.
- Sutton, R. S. (1996). *Generalization in Reinforcement Learning: Successful Examples Using Sparse
  Coarse Coding*. NIPS 8. — origen del entorno Acrobot.
- Documentación de Gymnasium: https://gymnasium.farama.org/environments/classic_control/acrobot/
