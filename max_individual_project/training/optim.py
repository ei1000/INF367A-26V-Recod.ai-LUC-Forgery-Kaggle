from __future__ import annotations


def set_optimizer_learning_rate(optimizer, learning_rate: float):
    if optimizer is None:
        return

    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate
