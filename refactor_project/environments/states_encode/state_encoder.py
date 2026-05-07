from typing import Any

import numpy as np
import torch

from refactor_project.config.config import Config
from refactor_project.rl.actions import OpType


def empty_state(L_max: int) -> np.ndarray:
    return np.zeros((5, L_max), dtype=np.int64)


def encode_action_in_state(state: np.ndarray, step: int, action: Any) -> np.ndarray:
    kind = action[0]
    state[:, step] = 0

    if kind == "ROT":
        _, ax, q, _ = action
        state[1, step] = q + 1
        state[2, step] = OpType.ROT.value
        state[3, step] = ax

    elif kind == "ENC":
        _, ax, q, feat = action
        state[1, step] = q + 1
        state[2, step] = OpType.ENC.value
        state[3, step] = ax
        state[4, step] = feat + 1

    elif kind == "CNOT":
        _, ci, tj, _ = action
        if ci != tj:
            state[0, step] = ci + 1
            state[1, step] = tj + 1
            state[2, step] = OpType.CNOT.value

    return state


def sanitize_architecture(arch: torch.Tensor, n_qubits: int) -> torch.Tensor:
    arch = arch.clone()
    L = int(arch.shape[1])

    for local in range(L):
        op = int(arch[2, local].item())
        c = int(arch[0, local].item())
        t = int(arch[1, local].item())
        ax = int(arch[3, local].item())
        f1 = int(arch[4, local].item())

        if t > n_qubits or c > n_qubits:
            arch[:, local] = 0
            continue

        if op == OpType.CNOT.value:
            if c == 0 or t == 0 or c == t:
                arch[:, local] = 0

        elif op == OpType.ROT.value:
            # stronger sanitization: target + axis must exist
            if (t == 0) or (ax == 0):
                arch[:, local] = 0

        elif op == OpType.ENC.value:
            # stronger sanitization: target + axis + feature must exist
            if (t == 0) or (ax == 0) or (f1 == 0):
                arch[:, local] = 0

    return arch


def state_to_vec(
    state: np.ndarray, last_metric: float, current_n_qubits: int, cfg: Config
) -> np.ndarray:
    flat = state.flatten().astype(np.float32)
    extras = np.zeros(2, dtype=np.float32)
    extras[0] = np.float32(last_metric)

    denom = max(1, (cfg.max_qubits - cfg.min_qubits))
    extras[1] = np.float32((current_n_qubits - cfg.min_qubits) / denom)

    task_ctx = np.asarray(
        getattr(cfg, "task_context", [0.0, 0.0, 0.0, 0.0, 0.0]),
        dtype=np.float32,
    )
    return np.concatenate([flat, extras, task_ctx])
