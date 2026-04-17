from __future__ import annotations

from collections import deque
from dataclasses import replace
import time

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import DataLoader, TensorDataset

from refactor_project.config.config import Config
from refactor_project.config.util import get_thr_targets
from refactor_project.environments.qml.arch_util import (
    budget_excess,
    confusion_from_thr,
    count_ops,
    would_exceed_budget,
)
from refactor_project.environments.qml.proxy_trainer import ProxyPack, ProxyTrainer
from refactor_project.environments.qml.reward import RewardModule
from refactor_project.environments.states_encode.metric import bce_logits_loss_torch
from refactor_project.environments.states_encode.state_encoder import (
    empty_state,
    encode_action_in_state,
    sanitize_architecture,
)
from refactor_project.environments.states_encode.threshold import find_threshold
from refactor_project.features.features import (
    compute_saliency_importance_patches,
    compute_saliency_importance_pixels,
    init_feature_bank,
    make_patch_groups,
    patchify_mean_flat,
)
from refactor_project.model.model import CQV_End2End
from refactor_project.model.util.metrics import focal_loss_with_logits
from refactor_project.model.util.util import compute_pos_weight, init_head_bias_with_prevalence
from refactor_project.rl.actions import action_is_valid_for_qubits, build_action_list_superset
from refactor_project.util.util import RunningPctl

SEED_DELTA_CALIB = 99991


class QMLEnvEnd2End:
    def __init__(
        self,
        X_tr,
        Y_tr,
        X_val,
        Y_val,
        cfg: Config,
        logger,
        seed: int = 0,
        device: torch.device | str = "cpu",
    ):
        self.cfg = replace(cfg)
        self.seed = int(seed)
        self.logger = logger
        self.DEVICE = torch.device(device)

        self._last_depth_tape = None
        self._last_cnot_tape = None

        self.thr = 0.5
        self.last_ep_score = None
        self.last_spec = 0.0

        self.current_n_qubits = int(cfg.start_qubits)
        self._qubit_cooldown = 0

        # ── tensors ───────────────────────────────────────────────────────────
        self.X_tr = torch.as_tensor(X_tr, dtype=torch.float32, device=self.DEVICE)
        self.Y_tr = torch.as_tensor(Y_tr, dtype=torch.float32, device=self.DEVICE)
        self.X_val = torch.as_tensor(X_val, dtype=torch.float32, device=self.DEVICE)
        self.Y_val = torch.as_tensor(Y_val, dtype=torch.float32, device=self.DEVICE)

        if self.X_tr.dim() == 1:
            self.X_tr = self.X_tr.unsqueeze(1)
        if self.X_val.dim() == 1:
            self.X_val = self.X_val.unsqueeze(1)

        self.X_tr_raw = self.X_tr.detach().clone()
        self.X_val_raw = self.X_val.detach().clone()

        self._rng = np.random.default_rng(int(seed))

        # ── domain detection + patch groups ──────────────────────────────────
        self.patch_groups = None
        self.P = None

        input_dim_now = (
            int(self.X_tr.shape[1])
            if self.X_tr.dim() == 2
            else int(self.X_tr.view(self.X_tr.shape[0], -1).shape[1])
        )
        self.is_image_like = bool(input_dim_now == 784)

        if self.is_image_like and bool(cfg.use_patch_bank):
            self.patch_groups = make_patch_groups(28, 28, cfg.patch_size, cfg.patch_stride)
            self.P = int(len(self.patch_groups))
            self.logger.log_to_file(
                "patchify",
                f"[patch_groups] enabled is_image_like=1 P={self.P} patch={cfg.patch_size} stride={cfg.patch_stride}",
            )
        else:
            self.logger.log_to_file(
                "patchify",
                f"[patch_groups] disabled is_image_like={int(self.is_image_like)} "
                f"use_patch_bank={int(bool(cfg.use_patch_bank))} input_dim={input_dim_now}",
            )

        # ── feature bank sizing / schedule ────────────────────────────────────
        if bool(cfg.use_patch_bank) and (self.P is not None):
            max_k = int(min(int(cfg.feature_bank_size), int(self.P)))
            min_k = int(min(int(cfg.feature_bank_min_size), int(max_k)))
            self.feature_bank_size_eff = int(max_k)
            self.feature_bank_min_eff = int(min_k)
            sched = tuple(cfg.feature_bank_schedule)
            if len(sched) == 0:
                sched = (int(max_k),)
            self.feature_bank_schedule_eff = tuple(
                int(np.clip(int(k), int(min_k), int(max_k))) for k in sched
            )
        else:
            self.feature_bank_size_eff = int(cfg.feature_bank_size)
            self.feature_bank_min_eff = int(
                min(int(cfg.feature_bank_min_size), int(cfg.feature_bank_size))
            )
            sched = tuple(cfg.feature_bank_schedule)
            if len(sched) == 0:
                sched = (int(self.feature_bank_size_eff),)
            self.feature_bank_schedule_eff = tuple(
                int(
                    np.clip(
                        int(k), int(self.feature_bank_min_eff), int(self.feature_bank_size_eff)
                    )
                )
                for k in sched
            )

        if len(self.feature_bank_schedule_eff) == 0:
            self.feature_bank_schedule_eff = (int(cfg.feature_bank_size),)

        # ── compact patchify ──────────────────────────────────────────────────
        if (
            bool(cfg.use_patch_bank)
            and bool(cfg.patch_bank_compact_features)
            and (self.patch_groups is not None)
            and bool(self.is_image_like)
        ):
            Dtr = int(self.X_tr.shape[1])
            Dva = int(self.X_val.shape[1])
            if Dtr == 784 and Dva == 784:
                self.X_tr = patchify_mean_flat(self.X_tr, self.patch_groups, device=self.DEVICE)
                self.X_val = patchify_mean_flat(self.X_val, self.patch_groups, device=self.DEVICE)
                logger.log_to_file(
                    "patchify",
                    f"[compact] X_tr -> {tuple(self.X_tr.shape)}  X_val -> {tuple(self.X_val.shape)} (P={self.P})",
                )
            else:
                logger.log_to_file(
                    "patchify",
                    f"[skip compact] X already compact? X_tr={tuple(self.X_tr.shape)} X_val={tuple(self.X_val.shape)}",
                )

        # ── normalização ──────────────────────────────────────────────────────
        self._x_mean = None
        self._x_std = None
        if bool(cfg.normalize_inputs):
            eps = float(cfg.norm_eps)
            per_feature = bool(cfg.norm_per_feature)
            clip = float(cfg.norm_clip)
            use_tanh = bool(cfg.norm_tanh)

            with torch.no_grad():
                if per_feature:
                    mu = self.X_tr.mean(dim=0, keepdim=True)
                    sd = self.X_tr.std(dim=0, keepdim=True).clamp_min(eps)
                else:
                    mu = self.X_tr.mean()
                    sd = self.X_tr.std().clamp_min(eps)
                self._x_mean = mu
                self._x_std = sd
                self.X_tr = (self.X_tr - mu) / sd
                self.X_val = (self.X_val - mu) / sd
                if clip > 0.0:
                    self.X_tr = torch.clamp(self.X_tr, -clip, clip)
                    self.X_val = torch.clamp(self.X_val, -clip, clip)
                if use_tanh:
                    self.X_tr = torch.tanh(self.X_tr)
                    self.X_val = torch.tanh(self.X_val)

            self.logger.log_to_file(
                "norm",
                f"[norm] per_feature={per_feature} eps={eps} clip={clip} tanh={use_tanh} "
                f"X_tr={tuple(self.X_tr.shape)} X_val={tuple(self.X_val.shape)}",
            )

        # ── actions ───────────────────────────────────────────────────────────
        self.ACTIONS = build_action_list_superset(
            int(cfg.max_qubits),
            int(self.feature_bank_size_eff),
            allow_nop=bool(cfg.allow_nop),
        )
        self.N_ACTIONS = len(self.ACTIONS)

        # ── data loaders ──────────────────────────────────────────────────────
        self.tr_dl_full = DataLoader(
            TensorDataset(self.X_tr, self.Y_tr),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )
        self.tr_dl_small = self._make_small_stratified_loader(
            X=self.X_tr,
            Y=self.Y_tr,
            subset_size=int(cfg.inner_train_subset_size),
            batch_size=int(cfg.batch_size),
            seed=int(seed),
        )

        self.pos_weight = compute_pos_weight(self.Y_tr, self.DEVICE)
        self.prev_train = (self.Y_tr.detach().cpu().numpy() > 0.5).astype(int)

        # ── episode state ─────────────────────────────────────────────────────
        self._recent_actions_q = deque(maxlen=int(cfg.recent_actions_maxlen))
        self.recent_actions: set = set()
        self._terminal_arch_hashes: set = set()
        self._ep_count: int = 0
        self._terminal_thr_stars: list = []
        self.current_bank_k = int(self.feature_bank_size_eff)
        self.episode_times: list = []
        self._calib_cache: dict = {}
        self._arch_result_cache: dict = {}
        self._arch_result_cache_maxsize: int = 256

        # ── depth ref ─────────────────────────────────────────────────────────
        self._depth_p95 = RunningPctl(
            p=float(cfg.depth_ref_pctl),
            maxlen=int(cfg.depth_ref_buf),
        )
        self._depth_ref_cached = float(cfg.depth_ref_default)
        self._last_depth_raw = 0.0

        # ── modules ───────────────────────────────────────────────────────────
        self._reward = RewardModule(
            cfg=self.cfg,
            n_qubits_getter=lambda: self.current_n_qubits,
            depth_ref_getter=self._depth_ref,
        )

        self._trainer = ProxyTrainer(
            cfg=self.cfg,
            device=self.DEVICE,
            logger=self.logger,
            seed=int(seed),
            pos_weight=self.pos_weight,
            prev_train=self.prev_train,
            model_builder=self._build_model,
            eval_metrics=self._eval_metrics,
            predict_lp_cal=self._predict_logits_probs_capped_train,
            predict_lp=self._predict_logits_probs_capped,
            calib_tensors=self._get_calib_tensors,
            make_dl_small=self._make_dl_small_for_seed,
            X_val=self.X_val,
            Y_val=self.Y_val,
            find_threshold=find_threshold,
            get_thr_targets=get_thr_targets,
            init_head_bias_with_prevalence=init_head_bias_with_prevalence,
            focal_loss_with_logits=focal_loss_with_logits,
            bce_logits_loss_torch=bce_logits_loss_torch,
        )

        self.feature_bank = None
        self.reset()
        self._metric_weight = 1.0

    # ── public API ────────────────────────────────────────────────────────────

    def set_metric_weight(self, w: float) -> None:
        self._metric_weight = float(max(0.0, w))
        self._reward.set_metric_weight(w)

    def set_current_bank_k(self, k: int) -> None:
        lo = int(self.feature_bank_min_eff)
        hi = int(self.feature_bank_size_eff)
        k = int(np.clip(int(k), lo, hi))
        self.current_bank_k = k
        self.logger.log_to_file(
            "feature_bank", f"[feature_bank k] current_bank_k={self.current_bank_k}"
        )

    def flush_episode_reward_sums(self) -> dict:
        return self._reward.flush_episode_reward_sums()

    def valid_action_mask(self) -> np.ndarray:
        mask = np.array(
            [
                action_is_valid_for_qubits(a, self.current_n_qubits, self.current_bank_k)
                for a in self.ACTIONS
            ],
            dtype=bool,
        )
        if not self.cfg.hard_block_budget:
            return mask
        arch = torch.tensor(self.state, dtype=torch.int64, device=self.DEVICE)
        counts = count_ops(arch)
        for i, a in enumerate(self.ACTIONS):
            if mask[i] and would_exceed_budget(counts, a, self.cfg):
                mask[i] = False
        return mask

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        self.step_idx = 0
        self.state = empty_state(self.cfg.L_max)
        self.last_auc = 0.5
        self.last_sens = 0.0
        self.last_spec = 0.0
        self.last_proxy_score = 0.5
        self.thr = float(self.cfg.thr_init)
        self._last_depth_tape = None
        self._last_cnot_tape = None
        self.current_n_qubits = int(self.cfg.start_qubits)
        self._qubit_cooldown = 0

        if self._ep_count > 0:
            self._recent_actions_q.clear()
            self.recent_actions.clear()

        self._terminal_thr_stars = []
        self._ep_count += 1
        self._reward.reset_episode()
        self._reward.last_spec = 0.0

        X_bank = self.X_tr.detach().cpu().numpy().astype(np.float32)
        if self.feature_bank is None:
            y_np = (self.Y_tr.detach().cpu().numpy().reshape(-1) > 0.5).astype(np.int32)
            self.feature_bank = init_feature_bank(
                X_bank,
                y_np,
                k_max=int(self.feature_bank_size_eff),
                mode=str(self.cfg.feature_bank_update),
                seed=int(self.seed),
                use_patch_bank=bool(
                    self.cfg.use_patch_bank
                    and (self.patch_groups is not None)
                    and self.is_image_like
                ),
                patch_groups=(
                    self.patch_groups
                    if (self.patch_groups is not None and self.is_image_like)
                    else None
                ),
            )
            self.logger.log_to_file(
                "feature_bank",
                f"[init] mode={self.cfg.feature_bank_update} k={len(self.feature_bank)} use_patch_bank={self.cfg.use_patch_bank}",
            )
        return self.state.copy()

    # ── step ──────────────────────────────────────────────────────────────────

    def step(self, action_idx: int):
        t0 = time.perf_counter()
        action = self.ACTIONS[action_idx]
        cfg = self.cfg

        def _info_base(**extra):
            youden = float(np.clip(self.last_sens + self.last_spec - 1.0, -1.0, 1.0))
            info = {
                "auc_val": float(self.last_auc),
                "sens_val": float(self.last_sens),
                "spec_val": float(self.last_spec),
                "youden_val": float(youden),
                "steps": int(self.step_idx),
                "thr": float(self.thr),
            }
            info.update(extra)
            return info

        arch_before = torch.tensor(self.state, dtype=torch.int64, device=self.DEVICE)
        counts_before = count_ops(arch_before)

        # 1) hard budget blocking
        if cfg.hard_block_budget:
            saturated = (
                counts_before["ENC"] >= cfg.ENC_budget
                and counts_before["ROT"] >= cfg.ROT_budget
                and counts_before["CNOT"] >= cfg.CNOT_budget
            )
            if saturated:
                return (
                    self.state.copy(),
                    0.0,
                    True,
                    _info_base(
                        depth=int(self.step_idx),
                        cnot=int(counts_before["CNOT"]),
                        budget_saturated=True,
                        counts=counts_before,
                        step_time_s=float(time.perf_counter() - t0),
                    ),
                )
            if would_exceed_budget(counts_before, action, cfg):
                return (
                    self.state.copy(),
                    -float(cfg.budget_penalty),
                    False,
                    _info_base(
                        depth=int(self.step_idx),
                        cnot=int(counts_before["CNOT"]),
                        budget_blocked=True,
                        counts=counts_before,
                        action=action,
                        step_time_s=float(time.perf_counter() - t0),
                    ),
                )

        # 2) qubit cooldown
        if self._qubit_cooldown > 0:
            self._qubit_cooldown -= 1

        # 3) qubit actions
        kind = action[0]
        if kind == "ADD_QUBIT":
            if self._qubit_cooldown == 0 and self.current_n_qubits < cfg.max_qubits:
                self.current_n_qubits += 1
                self._qubit_cooldown = int(cfg.qubit_change_cooldown)
                reward = -0.01
            else:
                reward = -0.02
            return (
                self.state.copy(),
                float(reward),
                False,
                _info_base(
                    depth=0,
                    cnot=0,
                    n_qubits=int(self.current_n_qubits),
                    step_time_s=float(time.perf_counter() - t0),
                ),
            )

        if kind == "REMOVE_QUBIT":
            if self._qubit_cooldown == 0 and self.current_n_qubits > cfg.min_qubits:
                self.current_n_qubits -= 1
                self._qubit_cooldown = int(cfg.qubit_change_cooldown)
                reward = -0.005
            else:
                reward = -0.02
            return (
                self.state.copy(),
                float(reward),
                False,
                _info_base(
                    depth=0,
                    cnot=0,
                    n_qubits=int(self.current_n_qubits),
                    step_time_s=float(time.perf_counter() - t0),
                ),
            )

        # 4) validação
        if not action_is_valid_for_qubits(action, self.current_n_qubits, self.current_bank_k):
            return (
                self.state.copy(),
                -0.02,
                False,
                _info_base(depth=0, cnot=0, n_qubits=int(self.current_n_qubits)),
            )

        # 5) ENC: map bank index -> feature global
        if kind == "ENC":
            ax, q, b = action[1], action[2], action[3]
            if b is None:
                return (
                    self.state.copy(),
                    -0.02,
                    False,
                    {"bad_action": action, "reason": "ENC b is None"},
                )
            if int(b) < 0 or int(b) >= int(self.current_bank_k):
                raise RuntimeError(f"ENC index out of range: b={b} bank_k={self.current_bank_k}")
            feat_global = (
                int(b)
                if (self.feature_bank is None or len(self.feature_bank) == 0)
                else int(self.feature_bank[int(b)])
            )
            input_dim = int(self.X_tr.shape[1])
            if feat_global < 0 or feat_global >= input_dim:
                raise RuntimeError(
                    f"ENC feature out of range: feat_global={feat_global} input_dim={input_dim} "
                    f"(bank_k={self.current_bank_k}, b={b})"
                )
            action = ("ENC", ax, q, feat_global)

        # 6) NOP
        if kind == "NOP":
            min_k = int(cfg.min_steps_before_nop)
            if self.step_idx < min_k:
                return (
                    self.state.copy(),
                    -float(cfg.nop_penalty),
                    False,
                    _info_base(
                        nop_blocked=True,
                        min_steps_before_nop=int(min_k),
                        depth=int(self.step_idx),
                        cnot=int(counts_before["CNOT"]),
                        n_qubits=int(self.current_n_qubits),
                        step_time_s=float(time.perf_counter() - t0),
                    ),
                )
            return (
                self.state.copy(),
                0.0,
                True,
                _info_base(
                    depth=int(self.step_idx),
                    cnot=int(counts_before["CNOT"]),
                    n_qubits=int(self.current_n_qubits),
                    step_time_s=float(time.perf_counter() - t0),
                ),
            )

        # 7) aplica ação
        encode_action_in_state(self.state, self.step_idx, action)
        self.step_idx += 1

        # 8) step barato — usa métricas cacheadas
        arch = torch.tensor(self.state, dtype=torch.int64, device=self.DEVICE)
        auc, sens = float(self.last_auc), float(self.last_sens)

        if bool(cfg.depth_use_proxy_when_skip):
            counts_now = count_ops(arch)
            n_ops = int(counts_now["ENC"] + counts_now["ROT"] + counts_now["CNOT"])
            depth = int(max(1, round(cfg.proxy_depth_per_op * n_ops)))
            cnot_count = int(cfg.proxy_cnot_per_cnot * counts_now["CNOT"])
        else:
            depth = int(self.step_idx)
            cnot_count = int(count_ops(arch)["CNOT"])

        action_key = tuple(action)
        self._reward.last_spec = float(self.last_spec)
        reward = self._reward.compute(
            auc, sens, arch, int(depth), int(cnot_count), action_key, self.recent_actions
        )

        if len(self._recent_actions_q) == self._recent_actions_q.maxlen:
            old = self._recent_actions_q.popleft()
            self.recent_actions.discard(old)
        self._recent_actions_q.append(action_key)
        self.recent_actions.add(action_key)

        done = bool(self.step_idx >= int(cfg.L_max))
        counts_now = count_ops(arch)
        excess_now = budget_excess(counts_now, cfg)

        return (
            self.state.copy(),
            float(reward),
            bool(done),
            {
                "auc_val": float(auc),
                "sens_val": float(sens),
                "steps": int(self.step_idx),
                "depth": int(depth),
                "cnot": int(cnot_count),
                "thr": float(self.thr),
                "counts": counts_now,
                "excess": excess_now,
                "n_qubits": int(self.current_n_qubits),
                "step_time_s": float(time.perf_counter() - t0),
            },
        )

    # ── terminal_evaluate ─────────────────────────────────────────────────────

    def terminal_evaluate(self):
        t0 = time.perf_counter()
        _arch_cache_key = hash(self.state.tobytes())

        # cache hit
        if _arch_cache_key in self._arch_result_cache:
            _cached = self._arch_result_cache[_arch_cache_key]
            self.last_auc, self.last_sens, self.last_spec = (
                float(_cached[0]),
                float(_cached[1]),
                float(_cached[2]),
            )
            self.thr = float(_cached[3])
            try:
                self.logger.log_to_file(
                    "terminal_splits", f"[arch_cache_HIT] auc={_cached[0]:.4f}"
                )
            except Exception:
                pass
            return _cached

        arch = torch.tensor(self.state, dtype=torch.int64, device=self.DEVICE)
        phase = str(self.cfg.phase).lower()

        try:
            _yv = (self.Y_val.detach().cpu().numpy().reshape(-1) > 0.5).astype(np.int32)
            _nv = int(len(_yv))
            self.logger.log_to_file(
                "terminal_splits",
                f"[N_val] total={_nv} pos={int(_yv.sum())} neg={_nv - int(_yv.sum())} "
                f"val_frac={self.cfg.val_frac_search:.2f} percent_search={self.cfg.percent_search}",
            )
        except Exception:
            pass

        thr_config = self._build_thr_config(phase)

        try:
            packs, proxy_scores = self._trainer.run_multi_seed(arch, phase, thr_config)
        except Exception as e:
            try:
                self.logger.log_to_file(
                    "thr", f"[terminal_evaluate][EXCEPTION] {type(e).__name__}: {e}"
                )
            except Exception:
                pass
            counts = count_ops(arch)
            tinfo_err = {"error": f"{type(e).__name__}: {e}"}
            self.last_auc, self.last_sens, self.last_spec = 0.5, 0.0, 0.0
            self.thr = float(self.cfg.thr_init)
            return 0.5, 0.0, 0.0, self.thr, int(self.step_idx), int(counts["CNOT"]), tinfo_err

        proxy_arr = np.asarray(proxy_scores, dtype=float)
        std_proxy = float(np.std(proxy_arr)) if len(proxy_arr) > 1 else 0.0

        agg = str(self.cfg.proxy_aggregation).lower().strip()
        if agg == "min":
            best_idx = int(np.argmin(proxy_arr))
        elif agg == "quantile":
            q = float(self.cfg.proxy_quantile)
            q_val = float(np.quantile(proxy_arr, q))
            best_idx = int(np.argmin(np.abs(proxy_arr - q_val)))
        else:
            best_idx = int(np.argmax(proxy_arr))

        best: ProxyPack = packs[best_idx]

        thr_stars_all = [pk.thr_star for pk in packs]
        thr_std = (
            float(np.std(np.asarray(thr_stars_all, dtype=float)))
            if len(thr_stars_all) > 1
            else 0.0
        )
        self._terminal_thr_stars = thr_stars_all

        if bool(self.cfg.log_thr_stability):
            try:
                self.logger.log_to_file(
                    "thr_stability",
                    f"[thr_stability] n_seeds={len(packs)} agg={agg} "
                    f"thr_stars={[f'{t:.3f}' for t in thr_stars_all]} "
                    f"thr_std={thr_std:.4f} "
                    f"proxy_scores={[f'{s:.4f}' for s in proxy_scores]} "
                    f"std_proxy={std_proxy:.4f} best_idx={best_idx}",
                )
            except Exception:
                pass

        self.last_auc = float(best.auc)
        self.last_sens = float(best.sens)
        self.last_spec = float(best.spec)
        self.thr = float(best.thr_star)
        self._reward.last_spec = float(best.spec)

        counts = count_ops(arch)
        depth_t = int(self.step_idx)
        cnot_t = int(counts["CNOT"])

        tinfo = dict(best.thr_info)
        tinfo["std_proxy"] = float(std_proxy)
        tinfo["thr_std"] = float(thr_std)
        tinfo["n_seeds"] = int(len(packs))
        tinfo["collapse_dbg"] = best.collapse_dbg

        self._terminal_arch_hashes.add(hash(self.state.tobytes()))

        try:
            self.logger.log_to_file(
                "thr",
                f"[terminal] phase={phase} auc={self.last_auc:.4f} sens={self.last_sens:.4f} "
                f"spec={self.last_spec:.4f} thr*={self.thr:.3f} std_proxy={std_proxy:.4f} "
                f"n_seeds={len(packs)} time={time.perf_counter() - t0:.2f}s",
            )
            _yv2 = (self.Y_val.detach().cpu().numpy().reshape(-1) > 0.5).astype(np.int32)
            _nv2 = int(len(_yv2))
            _max_off = (self.cfg.proxy_n_seeds - 1) * self.cfg.proxy_seed_delta
            self.logger.log_to_file(
                "terminal_splits",
                f"[split_summary] N_val={_nv2}(pos={int(_yv2.sum())},neg={_nv2 - int(_yv2.sum())}) "
                f"val_frac={self.cfg.val_frac_search:.2f} percent_search={self.cfg.percent_search} "
                f"calib_seed_delta={self.cfg.calib_seed_delta} max_proxy_offset={_max_off} "
                f"delta_gap={self.cfg.calib_seed_delta - _max_off}  target:N_val>=400,gap>>0",
            )
            self.logger.log_to_file(
                "episode",
                f"[episode] phase={phase} steps={int(self.step_idx)} "
                f"auc={self.last_auc:.4f} sens={self.last_sens:.4f} spec={self.last_spec:.4f} "
                f"thr*={self.thr:.3f} "
                f"calib_bce={best.calib_bce:.6f} val_bce={best.val_bce:.6f} "
                f"std_proxy={std_proxy:.4f} n_seeds={len(packs)}",
            )
        except Exception:
            pass

        _result = (
            float(self.last_auc),
            float(self.last_sens),
            float(self.last_spec),
            float(self.thr),
            int(depth_t),
            int(cnot_t),
            tinfo,
        )
        if len(self._arch_result_cache) >= int(self._arch_result_cache_maxsize):
            try:
                del self._arch_result_cache[next(iter(self._arch_result_cache))]
            except Exception:
                pass
        self._arch_result_cache[_arch_cache_key] = _result
        return _result

    # ── model builder ─────────────────────────────────────────────────────────

    def _build_model(self, arch_mat: torch.Tensor, diff_method: str = "adjoint") -> CQV_End2End:
        cfg = self.cfg
        arch_mat = sanitize_architecture(arch_mat, self.current_n_qubits)
        model = CQV_End2End(
            arch_mat=arch_mat,
            n_qubits=self.current_n_qubits,
            enc_lambda=float(cfg.enc_lambda),
            diff_method=str(diff_method),
            input_dim=int(self.X_tr.shape[1]),
            enc_affine_mode=str(cfg.enc_affine_mode),
            enc_alpha_init=float(cfg.enc_alpha_init),
            enc_beta_init=float(cfg.enc_beta_init),
            enc_beta_max=float(cfg.enc_beta_max),
            use_batched_qnode=bool(cfg.use_batched_qnode),
            vqc_theta_init_std=float(cfg.vqc_theta_init_std),
        ).to(self.DEVICE)

        if bool(cfg.freeze_enc_params):
            if hasattr(model, "enc_alpha_raw"):
                model.enc_alpha_raw.requires_grad_(False)
            if hasattr(model, "enc_beta_raw"):
                model.enc_beta_raw.requires_grad_(False)
            self.logger.log_to_file(
                "enc_params",
                f"[FIXED ENC] enc_alpha_raw e enc_beta_raw congelados "
                f"(alpha_init={cfg.enc_alpha_init}, beta_init={cfg.enc_beta_init})",
            )

        phase = str(cfg.phase).lower()
        if phase.startswith("final"):
            model.set_clamp_logits(True, clamp_value=float(cfg.final_logit_clamp))
            model.logit_scale_eval_only = False
            model.set_logit_scale_trainable(True)
        else:
            model.set_clamp_logits(False)
            model.logit_scale_eval_only = False
            model.set_logit_scale_trainable(
                bool(cfg.search_logit_scale_trainable), value=float(cfg.search_logit_scale)
            )
            model.logit_scale_min = float(cfg.search_logit_scale_min)
            model.logit_scale_max = float(cfg.search_logit_scale_max)

        return model

    # ── prediction helpers ────────────────────────────────────────────────────

    @torch.no_grad()
    def _predict_probs_capped(self, model, X: torch.Tensor, cap: int) -> tuple[np.ndarray, int]:
        model.eval()
        n = max(1, int(min(int(cap), int(X.shape[0]))))
        logits = model(X[:n].detach())
        probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
        return probs, n

    @torch.no_grad()
    def _predict_logits_probs_capped(
        self, model, X: torch.Tensor, cap: int
    ) -> tuple[np.ndarray, np.ndarray, int]:
        model.eval()
        n = max(1, int(min(int(cap), int(X.shape[0]))))
        logits_t = model(X[:n].detach())
        logits = logits_t.detach().cpu().numpy().reshape(-1)
        lc = np.clip(logits, -12.0, 12.0)
        probs = 1.0 / (1.0 + np.exp(-lc))
        return logits, probs, n

    @torch.no_grad()
    def _predict_logits_probs_capped_train(self, model, X: torch.Tensor, cap: int):
        model.eval()
        n = max(1, int(min(int(cap), int(X.shape[0]))))
        logits_t = model(X[:n].detach())
        logits = logits_t.detach().cpu().numpy().reshape(-1)
        abs_logits = np.abs(logits)
        p95 = np.percentile(abs_logits, 95)
        T = max(1.0, p95 / 8.0)
        logits_T = logits / T
        probs = 1.0 / (1.0 + np.exp(-logits_T))
        return logits, logits_T, probs, n

    # ── eval metrics ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def _eval_metrics(
        self,
        model,
        X: torch.Tensor,
        Y: torch.Tensor,
        thr: float | None = None,
    ) -> tuple[float, float, float]:
        model.eval()
        probs, n = self._predict_probs_capped(model, X, cap=int(self.cfg.val_loss_cap))
        yt_full = (Y.detach().cpu().numpy().reshape(-1) > 0.5).astype(int)
        yt = yt_full[:n]
        n_pos = int(yt.sum())
        n_neg = int(len(yt) - n_pos)

        if n_pos == 0 or n_neg == 0:
            auc = 0.5
        else:
            try:
                auc = float(roc_auc_score(yt, probs))
                if not np.isfinite(auc):
                    auc = 0.5
            except Exception:
                auc = 0.5

        thr_use = float(thr) if thr is not None else float(self.thr)
        if (float(probs.max()) - float(probs.min())) < 1e-3:
            thr_use = 0.5

        tp, fp, tn, fn = confusion_from_thr(yt, probs, thr_use)
        sens = float(tp) / float(max(n_pos, 1))
        spec = float(tn) / float(max(n_neg, 1))
        return float(auc), float(sens), float(spec)

    # ── saliency rescore ──────────────────────────────────────────────────────

    def maybe_rescore_feature_bank_saliency(self, arch_use: np.ndarray) -> None:
        if str(self.cfg.feature_bank_update).lower() != "saliency":
            return
        if self.X_tr is None or self.Y_tr is None:
            return

        model = self._build_model(
            torch.tensor(arch_use, dtype=torch.int64, device=self.DEVICE),
            diff_method="backprop",
        )

        if bool(self.cfg.use_patch_bank):
            if self.patch_groups is None or not bool(self.is_image_like):
                return
            sal = compute_saliency_importance_patches(
                model,
                self.X_tr,
                self.Y_tr,
                top_k=int(self.feature_bank_size_eff),
            )
            self.feature_bank = np.asarray(sal[: int(self.feature_bank_size_eff)], dtype=np.int64)
            return

        sal = compute_saliency_importance_pixels(
            model,
            self.X_tr,
            self.Y_tr,
            top_k=int(self.feature_bank_size_eff),
        )
        self.feature_bank = np.asarray(sal[: int(self.feature_bank_size_eff)], dtype=np.int64)

    # ── reinit theta ──────────────────────────────────────────────────────────

    def _reinit_vqc_theta(self) -> None:
        theta_init_std = float(self.cfg.vqc_theta_init_std)
        arch = torch.tensor(self.state, dtype=torch.int64, device=self.DEVICE)
        _tmp = self._build_model(arch, diff_method="backprop")
        if hasattr(_tmp, "theta"):
            with torch.no_grad():
                torch.nn.init.normal_(_tmp.theta, mean=0.0, std=theta_init_std)
            self.logger.log_to_file(
                "rl",
                f"[_reinit_vqc_theta] theta reinit std={theta_init_std:.4f} shape={list(_tmp.theta.shape)}",
            )
        del _tmp
        self._arch_result_cache.clear()

    # ── private helpers ───────────────────────────────────────────────────────

    def _depth_ref(self) -> float:
        ref = float(self._depth_p95.value(default=float(self._depth_ref_cached)))
        ref_min = float(self.cfg.depth_ref_min)
        ref = float(max(ref, ref_min))
        self._depth_ref_cached = float(ref)
        return float(ref)

    def _get_calib_tensors(self, cap: int, seed_for_cap: int) -> tuple[torch.Tensor, torch.Tensor]:
        key = (int(cap), int(seed_for_cap))
        if key in self._calib_cache:
            return self._calib_cache[key]

        y = (self.Y_tr.detach().cpu().numpy().reshape(-1) > 0.5).astype(np.int32)
        n = int(y.shape[0])
        cap_eff = int(min(max(2, cap), n))
        idx_pos = np.where(y == 1)[0]
        idx_neg = np.where(y == 0)[0]

        rng = np.random.default_rng(int(seed_for_cap))
        rng.shuffle(idx_pos)
        rng.shuffle(idx_neg)
        k_pos = int(min(len(idx_pos), cap_eff // 2))
        k_neg = int(min(len(idx_neg), cap_eff - k_pos))

        sel = []
        if k_pos > 0:
            sel.append(idx_pos[:k_pos])
        if k_neg > 0:
            sel.append(idx_neg[:k_neg])
        if not sel:
            sel_idx = rng.choice(np.arange(n), size=cap_eff, replace=False)
        else:
            sel_idx = np.concatenate(sel, axis=0)
            if sel_idx.size < cap_eff:
                rem = cap_eff - int(sel_idx.size)
                pool = np.setdiff1d(np.arange(n), sel_idx, assume_unique=False)
                if pool.size > 0:
                    sel_idx = np.concatenate(
                        [sel_idx, rng.choice(pool, size=min(rem, pool.size), replace=False)]
                    )
        rng.shuffle(sel_idx)

        sel_t = torch.as_tensor(sel_idx, device=self.DEVICE, dtype=torch.long)
        X_cal = self.X_tr.index_select(0, sel_t)
        Y_cal = self.Y_tr.index_select(0, sel_t)
        self._calib_cache[key] = (X_cal, Y_cal)

        try:
            ycal = (Y_cal.detach().cpu().numpy().reshape(-1) > 0.5).astype(np.int32)
            pos = int(ycal.sum())
            tot = int(ycal.shape[0])
            self.logger.log_to_file(
                "terminal_splits",
                f"[N_calib] total={tot} pos={pos} neg={tot - pos} "
                f"seed_for_cap={seed_for_cap} calib∩val_overlap=0(structural:X_tr⊥X_val)",
            )
        except Exception:
            pass
        return X_cal, Y_cal

    def _make_dl_small_for_seed(self, seed_offset: int) -> DataLoader:
        return self._make_small_stratified_loader(
            X=self.X_tr,
            Y=self.Y_tr,
            subset_size=int(self.cfg.inner_train_subset_size),
            batch_size=int(self.cfg.batch_size),
            seed=int(self.seed + seed_offset),
        )

    def _make_small_stratified_loader(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        subset_size: int,
        batch_size: int,
        seed: int = 0,
    ) -> DataLoader:
        y = (Y.detach().cpu().numpy().reshape(-1) > 0.5).astype(np.int32)
        idx_pos = np.where(y == 1)[0]
        idx_neg = np.where(y == 0)[0]
        rng = np.random.default_rng(int(seed))
        subset_size = int(max(2, min(subset_size, len(y))))
        k_pos = min(len(idx_pos), subset_size // 2)
        k_neg = min(len(idx_neg), subset_size - k_pos)
        sel_pos = (
            rng.choice(idx_pos, k_pos, replace=False) if k_pos > 0 else np.array([], dtype=int)
        )
        sel_neg = (
            rng.choice(idx_neg, k_neg, replace=False) if k_neg > 0 else np.array([], dtype=int)
        )
        sel = np.concatenate([sel_pos, sel_neg])
        if len(sel) < 2:
            sel = rng.choice(np.arange(len(y)), subset_size, replace=False)
        rng.shuffle(sel)
        Xs = X[torch.as_tensor(sel, device=X.device)]
        Ys = Y[torch.as_tensor(sel, device=Y.device)]
        ds = TensorDataset(Xs, Ys)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
        self.logger.log_to_file("speed", f"[inner subset] size={len(ds)} batch={batch_size}")
        return dl

    def _build_thr_config(self, phase: str) -> dict:
        cfg = self.cfg
        is_final = phase.startswith("final")

        if is_final:
            sens_target = float(cfg.final_sens_target)
            spec_min = float(cfg.final_thr_spec_min)
            fpr_max = float(cfg.final_thr_fpr_max)
            lam_spec = float(cfg.final_thr_lam_spec)
            lam_fpr = float(cfg.final_thr_lam_fpr)
            lam_sens = float(cfg.final_thr_lam_sens)
        else:
            sens_target = float(cfg.search_sens_target)
            spec_min = float(cfg.search_thr_spec_min)
            fpr_max = float(cfg.search_thr_fpr_max)
            lam_spec = float(cfg.search_thr_lam_spec)
            lam_fpr = float(cfg.search_thr_lam_fpr)
            lam_sens = float(cfg.search_thr_lam_sens)
            try:
                self.logger.log_to_file(
                    "thr",
                    f"[phase] phase={phase} targets sens/spec/fpr={sens_target:.3f}/{spec_min:.3f}/{fpr_max:.3f}",
                )
            except Exception:
                pass

        thr_mode = str(cfg.thr_mode).lower().strip()
        if thr_mode not in ("soft", "hard"):
            thr_mode = "soft"
        thr_policy = str(cfg.thr_policy).lower().strip()
        if thr_policy not in ("f2", "sens", "youden"):
            thr_policy = "f2"

        if thr_policy == "sens":
            lam_sens *= float(cfg.thr_policy_sens_mult)
            lam_spec *= float(cfg.thr_policy_spec_mult)
            lam_fpr *= float(cfg.thr_policy_fpr_mult)
        elif thr_policy == "youden":
            lam_spec *= float(cfg.thr_policy_youden_spec_mult)
            lam_sens *= float(cfg.thr_policy_youden_sens_mult)
            lam_fpr *= float(cfg.thr_policy_fpr_mult)

        return {
            "is_final": is_final,
            "thr_mode": thr_mode,
            "sens_target": sens_target,
            "spec_min": spec_min,
            "fpr_max": fpr_max,
            "lam_spec": lam_spec,
            "lam_fpr": lam_fpr,
            "lam_sens": lam_sens,
        }

    # ── compute_reward (mantido para compatibilidade com chamadas externas) ───

    def compute_reward(self, auc, sens, arch_mat, depth, cnot_count, action_key):
        self._reward.last_spec = float(self.last_spec)
        return self._reward.compute(
            auc, sens, arch_mat, depth, cnot_count, action_key, self.recent_actions
        )
