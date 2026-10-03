"""Comprobaciones del algoritmo: las cuatro propiedades de las que dependen los resultados.

No entrenan nada. Cada una fija una decision que el README declara y que, si se
rompiera en silencio, invalidaria una seccion entera del trabajo.

Lo que estas pruebas NO cubren, y conviene decirlo: el error de §10.2 no vive en
`optimize_step` sino en el sitio de llamada -`buf.push(..., terminated)` frente a
`buf.push(..., terminated or truncated)`-. Como aqui las transiciones se insertan a
mano en el buffer, la prueba del enmascaramiento pasa igual con las dos versiones. Lo
que detecta un cambio en el archivo del algoritmo es el SHA-256 que comprueba
`verificar_resultados.py`, no estas cuatro pruebas.
"""
from dataclasses import replace

import numpy as np
import torch

from acrobot_dqn import Config, QNetwork, ReplayBuffer, epsilon_by_step
from acrobot_dqn.dqn_acrobot_v3 import optimize_step

CFG = Config()


def test_epsilon_decae_por_paso_y_toca_el_piso():
    """ε va de 1,0 a 0,05 en 60.000 pasos y se queda ahí (§4, §7)."""
    assert epsilon_by_step(0, CFG) == CFG.epsilon_start
    # la interpolacion lineal no cae en 0,05 exacto por aritmetica de punto flotante
    assert abs(epsilon_by_step(CFG.epsilon_decay_steps, CFG) - CFG.epsilon_end) < 1e-9
    assert abs(epsilon_by_step(10 * CFG.epsilon_decay_steps, CFG) - CFG.epsilon_end) < 1e-9
    medio = epsilon_by_step(CFG.epsilon_decay_steps // 2, CFG)
    assert abs(medio - (CFG.epsilon_start + CFG.epsilon_end) / 2) < 1e-9


def test_no_se_entrena_antes_de_learning_starts():
    """Con el buffer por debajo de `learning_starts`, no hay actualizacion (§7)."""
    cfg = replace(CFG, learning_starts=1000, batch_size=64)
    q = QNetwork(6, 3)
    tgt = QNetwork(6, 3)
    opt = torch.optim.Adam(q.parameters(), lr=cfg.lr)
    buf = ReplayBuffer(cfg.buffer_size)
    for _ in range(cfg.learning_starts - 1):
        buf.push(np.zeros(6, np.float32), 0, -1.0, np.zeros(6, np.float32), False)
    assert optimize_step(q, tgt, opt, buf, cfg) is None
    buf.push(np.zeros(6, np.float32), 0, -1.0, np.zeros(6, np.float32), False)
    assert optimize_step(q, tgt, opt, buf, cfg) is not None


def test_el_terminal_real_anula_el_bootstrap():
    """Con `terminated=True` el objetivo es exactamente r, sin valor futuro (§3, §10.2).

    Fija la mitad del asunto de §10.2: que la bandera de terminal, cuando llega,
    anule el valor futuro. La otra mitad -que esa bandera sea `terminated` y no
    `terminated or truncated`- se decide en el bucle de entrenamiento y esta prueba
    no la ve; vease el encabezado del archivo. Se comprueba por comportamiento: si se
    entrena solo con transiciones terminales de recompensa -1, Q converge a -1. Si el
    bootstrap no se anulara, convergeria hacia -1/(1-gamma) = -100.
    """
    cfg = replace(CFG, learning_starts=1, batch_size=8, lr=0.01)
    torch.manual_seed(0)
    q = QNetwork(6, 3)
    tgt = QNetwork(6, 3)
    tgt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=cfg.lr)
    buf = ReplayBuffer(cfg.buffer_size)

    estado = np.ones(6, np.float32)
    siguiente = np.full(6, 2.0, np.float32)
    for _ in range(64):
        buf.push(estado, 0, -1.0, siguiente, True)     # terminated = True
    for _ in range(600):
        optimize_step(q, tgt, opt, buf, cfg)

    with torch.no_grad():
        valor = float(q(torch.as_tensor(estado).unsqueeze(0))[0, 0])
    assert abs(valor - (-1.0)) < 0.05, f"Q convergio a {valor:.3f} y deberia valer -1"


def test_la_red_tiene_la_forma_que_declara_el_readme():
    """6 → 128 → 128 → 3 y 17.795 parametros (§6)."""
    red = QNetwork(6, 3, CFG.hidden)
    assert sum(p.numel() for p in red.parameters()) == 17_795
    assert red(torch.zeros(4, 6)).shape == (4, 3)
