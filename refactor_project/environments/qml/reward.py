"""
reward.py
=========
RewardModule — extraído de QMLEnvEnd2End.compute_reward().

Recebe callables para n_qubits e depth_ref para não segurar
referência ao env inteiro.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable
import math
from typing import Callable

import numpy as np
import torch

from refactor_project.config.config import Config
from refactor_project.environments.qml.arch_util import budget_excess, count_ops, dead_qubit_count
from refactor_project.rl.actions import OpType


class RewardModule:
    def __init__(
        self,
        cfg: Config,
        n_qubits_getter: Callable[[], int],
        depth_ref_getter: Callable[[], float],
    ) -> None:
        self.cfg = cfg
        self._n_qubits = n_qubits_getter
        self._depth_ref = depth_ref_getter
        self._metric_weight = 1.0
        self.last_spec = 0.0
        self._ep_reward_sums: dict[str, float] = defaultdict(float)

    # ── public ───────────────────────────────────────────────────────────────

    def set_metric_weight(self, w: float) -> None:
        self._metric_weight = float(max(0.0, w))

    def compute(
        self,
        auc: float,
        sens: float,
        arch_mat: torch.Tensor,
        depth: int,
        cnot_count: int,
        action_key: Hashable,
        recent_actions: set,
        grace_qubits: dict[int, int] | None = None,  # FIX-A
    ) -> float:
        cfg = self.cfg

        if math.isnan(auc):
            auc = 0.0
        if math.isnan(sens):
            sens = 0.0

        auc01 = float(np.clip(float(auc), 0.0, 1.0))
        sens01 = float(np.clip(float(sens), 0.0, 1.0))
        last_spec = float(np.clip(float(self.last_spec), 0.0, 1.0))

        # proxy metric = Youden J + AUC  (idêntico ao original)
        youden = float(np.clip(sens01 + last_spec - 1.0, -1.0, 1.0))
        youden01 = float(0.5 * (youden + 1.0))
        proxy_metric = 0.5 * auc01 + 0.5 * youden01

        w = float(self._metric_weight)
        mscale = float(cfg.metric_shaping_scale)
        comp_metric = float(w * mscale * (proxy_metric - 0.5))

        # depth
        depth_raw = float(abs(depth))
        depth_ref = float(self._depth_ref())
        depth_norm = float(min(depth_raw / max(depth_ref, 1e-6), 1.0))
        comp_depth = float(-float(cfg.depth_penalty) * depth_norm)

        # circuit size
        n_rot = int(torch.sum(arch_mat[2, :] == OpType.ROT.value).item())
        comp_cnot = float(-cfg.cnot_penalty * float(cnot_count))
        comp_rot = float(-float(cfg.rot_penalty) * float(n_rot))

        # qubit cost
        denom = max(1, cfg.max_qubits - cfg.min_qubits)
        qnorm = (self._n_qubits() - cfg.min_qubits) / denom
        comp_qubit = float(-float(cfg.qubit_penalty * qnorm))

        reward = comp_metric + comp_depth + comp_cnot + comp_rot + comp_qubit

        # dead qubits + budget excess
        dead_q = dead_qubit_count(arch_mat, self._n_qubits(), grace_qubits=grace_qubits)
        counts = count_ops(arch_mat)
        excess = budget_excess(counts, cfg)
        comp_dead = float(-float(cfg.dead_qubit_penalty) * float(dead_q))
        comp_budget = float(
            -float(cfg.budget_penalty) * float(excess["ENC"] + excess["ROT"] + excess["CNOT"])
        )
        reward += comp_dead + comp_budget

        # anti-collapse spec floor
        spec_floor = float(cfg.spec_floor_shaping)
        if last_spec < spec_floor:
            pen = (
                float(cfg.spec_collapse_penalty)
                * float(spec_floor - last_spec)
                / max(spec_floor, 1e-6)
            )
            reward += float(-pen)
            self._ep_reward_sums["collapse"] += float(-pen)

        # repeat penalty
        comp_repeat = 0.0
        if action_key in recent_actions:
            comp_repeat = -abs(float(cfg.repeat_penalty))
            reward += comp_repeat

        # accumulate
        self._ep_reward_sums["metric"] += float(comp_metric)
        self._ep_reward_sums["depth"] += float(comp_depth)
        self._ep_reward_sums["cnot"] += float(comp_cnot)
        self._ep_reward_sums["qubit"] += float(comp_qubit)
        self._ep_reward_sums["rot"] += float(comp_rot)
        self._ep_reward_sums["dead"] += float(comp_dead)
        self._ep_reward_sums["budget"] += float(comp_budget)
        self._ep_reward_sums["repeat"] += float(comp_repeat)

        return float(reward)

    def flush_episode_reward_sums(self) -> dict:
        out = dict(self._ep_reward_sums)
        self._ep_reward_sums = defaultdict(float)
        return out

    def reset_episode(self) -> None:
        self._ep_reward_sums = defaultdict(float)
