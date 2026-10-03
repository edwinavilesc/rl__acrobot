"""DQN sobre Acrobot-v1 — U3 Taller 2.

El algoritmo vive en `dqn_acrobot_v3`; este paquete solo lo envuelve para poder usarlo
desde la terminal sin alterarlo.

    from acrobot_dqn import Config, train_dqn, evaluate
"""
from .dqn_acrobot_v3 import (  # noqa: F401
    DEVICE,
    Config,
    QNetwork,
    ReplayBuffer,
    choose_action,
    epsilon_by_step,
    evaluate,
    optimize_step,
    seed_everything,
    train_dqn,
)

__version__ = "1.0.0"
__all__ = ["Config", "QNetwork", "ReplayBuffer", "DEVICE", "train_dqn", "evaluate",
           "seed_everything", "epsilon_by_step", "choose_action", "optimize_step"]
