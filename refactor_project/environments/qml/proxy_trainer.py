"""
proxy_trainer.py
================
ProxyTrainer — extrai _one_proxy_run() de QMLEnvEnd2End.

Toda a lógica de warmup de head, proxy-train, recenter, collapse-check
e retry está aqui. O env chama apenas run_multi_seed().
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from refactor_project.config.config import Config
from refactor_project.environments.qml.arch_util import (
    collapse_health,
    p95_p5,
    rates_from_thr,
    safe_stats,
)

# ─────────────────────────────────────────────────────────────────────────────
# Resultado de uma execução de proxy
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ProxyPack:
    auc: float = 0.5
    sens: float = 0.0
    spec: float = 0.0
    thr_star: float = 0.5
    calib_bce: float = float("nan")
    val_bce: float = float("nan")
    calib_sens: float = 0.0
    calib_spec: float = 0.0
    calib_fpr: float = 1.0
    thr_info: dict = field(default_factory=dict)
    collapse_dbg: dict = field(default_factory=dict)
    model: Any = None  # nn.Module com melhor state_dict carregado


# ─────────────────────────────────────────────────────────────────────────────
# ProxyTrainer
# ─────────────────────────────────────────────────────────────────────────────


class ProxyTrainer:
    """
    Encapsula _one_proxy_run() e _run_proxy_train() (retry boost).

    O env passa:
      - model_builder  : callable(arch_mat, diff_method) -> nn.Module
      - eval_metrics   : callable(model, X_val, Y_val, thr) -> (auc, sens, spec)
      - predict_lp_cal : callable(model, X, cap) -> (logits_raw, logits_T, probs, n)
      - predict_lp     : callable(model, X, cap) -> (logits, probs, n)
      - calib_tensors  : callable(phase, seed_offset) -> (X_cal, Y_cal)
      - make_dl_small  : callable(seed_offset) -> DataLoader
      - X_val, Y_val   : tensors (mantidos no env)
    """

    def __init__(
        self,
        cfg: Config,
        device: torch.device,
        logger,
        seed: int,
        pos_weight,  # torch.Tensor | None  (do env)
        prev_train: np.ndarray,
        # callables injetados pelo env
        model_builder,
        eval_metrics,
        predict_lp_cal,
        predict_lp,
        calib_tensors,
        make_dl_small,
        X_val: torch.Tensor,
        Y_val: torch.Tensor,
        # helpers de threshold (já existentes no seu codebase)
        find_threshold,
        get_thr_targets,
        init_head_bias_with_prevalence,
        focal_loss_with_logits,
        bce_logits_loss_torch,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.logger = logger
        self.seed = int(seed)

        self.pos_weight = pos_weight
        self.prev_train = prev_train

        # callables
        self._model_builder = model_builder
        self._eval_metrics = eval_metrics
        self._predict_lp_cal = predict_lp_cal
        self._predict_lp = predict_lp
        self._calib_tensors = calib_tensors
        self._make_dl_small = make_dl_small
        self.X_val = X_val
        self.Y_val = Y_val

        self._find_threshold = find_threshold
        self._get_thr_targets = get_thr_targets
        self._init_head_bias_with_prevalence = init_head_bias_with_prevalence
        self._focal_loss_with_logits = focal_loss_with_logits
        self._bce_logits_loss_torch = bce_logits_loss_torch

        self._ema_head_loss: float | None = None
        self._ema_proxy_loss: float | None = None
        self.thr: float = 0.5

    # ── API pública ───────────────────────────────────────────────────────────

    def run_multi_seed(
        self,
        arch: torch.Tensor,
        phase: str,
        thr_config: dict,  # saída de _build_thr_config()
    ) -> tuple[list[ProxyPack], list[float]]:
        """
        Executa proxy runs para todos os seed offsets definidos no cfg.
        Retorna (packs, proxy_scores).
        """
        cfg = self.cfg
        use2 = bool(cfg.proxy_two_seeds)
        delta = int(cfg.proxy_seed_delta)
        n_seeds_cfg = int(cfg.proxy_n_seeds) if use2 else 1
        n_seeds_cfg = max(1, n_seeds_cfg)
        seed_offsets = [int(i * delta) for i in range(n_seeds_cfg)]

        packs = [
            self._one_proxy_run(arch, phase, thr_config, seed_offset=int(so))
            for so in seed_offsets
        ]

        proxy_scores = []
        for pk in packs:
            auc01_ = float(np.clip(pk.auc, 0.0, 1.0))
            sens_ = float(np.clip(pk.sens, 0.0, 1.0))
            spec_ = float(np.clip(pk.spec, 0.0, 1.0))
            youden_ = float(np.clip(sens_ + spec_ - 1.0, -1.0, 1.0))
            youden01_ = 0.5 * (youden_ + 1.0)
            proxy_scores.append(0.5 * auc01_ + 0.5 * youden01_)

        return packs, proxy_scores

    # ── core: uma run de proxy (= _one_proxy_run original) ───────────────────

    def _one_proxy_run(
        self,
        arch: torch.Tensor,
        phase: str,
        thr_config: dict,
        seed_offset: int,
    ) -> ProxyPack:
        cfg = self.cfg
        torch.manual_seed(int(self.seed + seed_offset))
        np.random.seed(int(self.seed + seed_offset))

        is_final = phase.startswith("final")

        # ── knobs de fase ─────────────────────────────────────────────────────
        search_head_only = bool(cfg.search_head_only) and (not is_final)
        search_vqc_batches_override = int(cfg.search_inner_train_batches_vqc_override)

        if is_final:
            term_diff = str(cfg.final_terminal_diff_method)
            n_head = int(cfg.final_inner_train_batches_head)
            n_vqc_batches = int(cfg.final_inner_train_batches_vqc)
            n_epochs = int(cfg.final_inner_epochs_classif)
            head_epochs = int(cfg.final_head_epochs)
            lr_head_eff = float(cfg.lr_head)
            wd_head_bias = float(cfg.wd_head)
        else:
            term_diff = str(cfg.search_terminal_diff_method)
            n_head = int(cfg.search_inner_train_batches_head)
            n_vqc_batches = int(cfg.search_inner_train_batches_vqc)
            n_epochs = int(cfg.search_inner_epochs_classif)
            head_epochs = int(cfg.search_head_epochs)
            lr_head_eff = float(cfg.lr_head)
            wd_head_bias = float(cfg.wd_head)

        n_head = max(0, n_head)
        n_vqc_batches = max(0, n_vqc_batches)
        n_epochs = max(1, n_epochs)
        head_epochs = max(1, head_epochs)

        use_pos_weight_search = bool(cfg.search_use_pos_weight)
        pos_weight_eff = self.pos_weight if (is_final or use_pos_weight_search) else None

        allow_bias_learn_search = bool(cfg.search_allow_bias_learn)

        # ── calib tensors ─────────────────────────────────────────────────────
        cap = int(cfg.final_calib_cap if is_final else cfg.search_calib_cap)
        seed_for_cap = int(self.seed) + int(cfg.calib_seed_delta) + int(seed_offset)
        X_cal, Y_cal = self._calib_tensors(cap, seed_for_cap)

        # ── build model ───────────────────────────────────────────────────────
        model = self._model_builder(arch, diff_method=term_diff)

        if is_final:
            self._init_head_bias_with_prevalence(model, self.prev_train)
        else:
            self._init_head_bias_with_prevalence(model, self.prev_train, force_p=0.5)
            torch.nn.init.normal_(model.head.weight, mean=0.0, std=0.02)

        if search_head_only:
            self._freeze_vqc(model)

        dl_small = self._make_dl_small(seed_offset)

        # ── 1) warmup head ────────────────────────────────────────────────────
        self._warmup_head(
            model,
            dl_small,
            pos_weight_eff,
            n_head,
            head_epochs,
            lr_head_eff,
            wd_head_bias,
            is_final,
            allow_bias_learn_search,
            phase,
            seed_offset,
        )

        # ── recenter logits após warmup ───────────────────────────────────────
        if (not is_final) and bool(cfg.search_recenter_logits):
            self._recenter_logits(model, X_cal, seed_offset)

        # ── 2) proxy train ────────────────────────────────────────────────────
        if is_final:
            self._unfreeze_all(model)
            if model.head.bias is not None:
                model.head.bias.requires_grad_(True)
        else:
            if search_head_only:
                self._freeze_vqc(model)
                if model.head.bias is not None:
                    model.head.bias.requires_grad_(True)
            if int(search_vqc_batches_override) >= 0:
                n_vqc_batches = int(search_vqc_batches_override)
            else:
                self._unfreeze_all(model)
                if model.head.bias is not None:
                    model.head.bias.requires_grad_(True)

        opt = self._build_optimizer(model, is_final, lr_head_eff, wd_head_bias, search_head_only)

        self._proxy_train_loop(
            model,
            opt,
            dl_small,
            pos_weight_eff,
            n_epochs,
            n_vqc_batches,
            n_head,
            search_head_only,
            is_final,
            phase,
            seed_offset,
        )

        # separability log
        self._log_separability(model, X_cal, phase, seed_offset)

        # ── retry config ──────────────────────────────────────────────────────
        do_retry = bool(cfg.collapse_retry)
        max_retry = int(cfg.collapse_max_retry)
        base_lr_vqc = float(cfg.lr_vqc)
        base_lr_head = float(cfg.lr_head)
        lr_scale = float(cfg.collapse_lr_scale)
        boost_epochs = int(cfg.collapse_boost_epochs)
        boost_mult = int(cfg.collapse_boost_batches_mult)
        boost_head_b = (max(1, n_head // 2)) if n_head > 0 else 0

        # ── retry loop ────────────────────────────────────────────────────────
        fail_streak_clinical = 0
        fail_streak_auc_floor = 0
        best_state: dict | None = None
        best_pack_d: dict | None = None

        for attempt in range(int(max_retry) + 1):
            model.eval()

            logits_raw, logits_tr, probs_tr, n = self._predict_lp_cal(
                model, X_cal, cap=int(min(int(cfg.val_loss_cap), int(X_cal.shape[0])))
            )
            yt_full = (Y_cal.detach().cpu().numpy().reshape(-1) > 0.5).astype(np.int32)
            yt = yt_full[:n]

            # calib BCE
            try:
                with torch.no_grad():
                    model.eval()
                    logits_t = model(X_cal[:n].detach())
                    calib_loss = self._bce_logits_loss_torch(
                        logits_t, Y_cal[:n].detach(), pos_weight=pos_weight_eff
                    )
            except Exception:
                calib_loss = float("nan")

            # recenter before threshold
            if (not is_final) and bool(cfg.search_recenter_before_thr):
                logits_tr, probs_tr, yt = self._recenter_before_thr(
                    model, X_cal, Y_cal, logits_tr, probs_tr, yt_full, attempt
                )
                n = len(yt)

            # health log
            self._log_health(logits_tr, probs_tr, yt, seed_offset, attempt)

            # threshold calibration
            thr_star, thr_info, sens_c, spec_c, fpr_c = self._calibrate_threshold(
                yt, probs_tr, logits_tr, thr_config
            )
            self.thr = float(thr_star)

            # val metrics
            auc, sens, spec = self._eval_metrics(model, self.X_val, self.Y_val, thr=self.thr)

            # val BCE
            try:
                with torch.no_grad():
                    model.eval()
                    capv = int(cfg.val_loss_cap)
                    nv = int(min(capv, int(self.X_val.shape[0])))
                    logits_v = model(self.X_val[:nv].detach())
                    val_loss = self._bce_logits_loss_torch(
                        logits_v, self.Y_val[:nv].detach(), pos_weight=pos_weight_eff
                    )
            except Exception:
                val_loss = float("nan")

            # streaks
            viable_count = (
                int(thr_info.get("viable_count", 0)) if isinstance(thr_info, dict) else 0
            )
            auc_val = float(auc)
            auc_eps = float(cfg.collapse_auc_eps)
            clinical_fail = (abs(auc_val - 0.5) <= auc_eps) and (viable_count == 0)

            auc_floor = float(cfg.search_auc_floor)
            is_search_now = phase.startswith("search")
            auc_low = bool(is_search_now and np.isfinite(auc_val) and (auc_val < auc_floor))
            logit_margin = float(p95_p5(np.asarray(logits_tr, dtype=float)))
            prob_std_now = float(np.std(np.asarray(probs_tr, dtype=float)))
            lmm = float(cfg.collapse_logit_margin_min)
            psw = float(cfg.collapse_prob_std_weak)
            lgt = float(cfg.collapse_logit_gate_mult) * lmm
            auc_floor_bad_sep = bool(
                auc_low
                and (
                    (np.isfinite(logit_margin) and logit_margin < lgt)
                    or (np.isfinite(prob_std_now) and prob_std_now < psw)
                )
            )

            fail_streak_clinical = (fail_streak_clinical + 1) if clinical_fail else 0
            fail_streak_auc_floor = (fail_streak_auc_floor + 1) if auc_floor_bad_sep else 0

            collapsed, cdbg = self._collapse_decision(
                logits_tr,
                probs_tr,
                auc_val,
                viable_count,
                fail_streak_clinical,
                fail_streak_auc_floor,
                phase,
            )

            # keep best
            if (
                (best_pack_d is None)
                or (float(auc) > float(best_pack_d["auc"]))
                or (
                    abs(float(auc) - float(best_pack_d["auc"])) < 1e-12
                    and (float(spec), float(sens))
                    > (float(best_pack_d["spec"]), float(best_pack_d["sens"]))
                )
            ):
                best_pack_d = {
                    "auc": float(auc),
                    "sens": float(sens),
                    "spec": float(spec),
                    "thr_star": float(thr_star),
                    "calib_sens": float(sens_c),
                    "calib_spec": float(spec_c),
                    "calib_fpr": float(fpr_c),
                    "thr_info": (thr_info if isinstance(thr_info, dict) else {}),
                    "collapse_dbg": dict(cdbg),
                    "calib_bce": float(calib_loss),
                    "val_bce": float(val_loss),
                }
                best_state = deepcopy(model.state_dict())

            try:
                self.logger.log_to_file(
                    "thr_debug",
                    f"[collapse-check] seed_off={seed_offset} attempt={attempt} "
                    f"collapsed={int(cdbg['collapsed'])} reason={cdbg['reason']} "
                    f"logit_p95_p5={float(cdbg['logit_p95_p5']):.4f} prob_std={float(cdbg['prob_std']):.4f} "
                    f"auc={auc_val:.4f} viable={viable_count}",
                )
            except Exception:
                pass

            if (not do_retry) or (not collapsed) or (attempt >= int(max_retry)):
                break

            # retry boost
            n_epochs2 = int(n_epochs + boost_epochs)
            n_batches2 = int(max(1, n_vqc_batches * boost_mult))
            self._run_proxy_train_boost(
                model,
                dl_small,
                pos_weight_eff,
                lr_vqc=float(base_lr_vqc * lr_scale),
                lr_head=float(base_lr_head),
                n_epochs_run=n_epochs2,
                n_batches_run=n_batches2,
                n_head_batches=boost_head_b,
                wd_head_bias=wd_head_bias,
                search_head_only=search_head_only,
                is_final=is_final,
                seed_offset=seed_offset,
            )

        assert best_pack_d is not None
        if best_state is not None:
            model.load_state_dict(best_state)

        self._log_head_norm(model, seed_offset)

        best_pack_d["model"] = model
        return ProxyPack(**{k: v for k, v in best_pack_d.items()})

    # ── helpers: freeze / unfreeze ────────────────────────────────────────────

    @staticmethod
    def _freeze_vqc(model) -> None:
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.head.parameters():
            p.requires_grad_(True)

    @staticmethod
    def _unfreeze_all(model) -> None:
        for p in model.parameters():
            p.requires_grad_(True)

    # ── helpers: param groups ─────────────────────────────────────────────────

    @staticmethod
    def _split_head_params(head) -> tuple[list, list]:
        w, b = [], []
        for p in head.parameters():
            (b if p.ndim == 1 else w).append(p)
        return w, b

    def _build_optimizer(
        self,
        model,
        is_final: bool,
        lr_head: float,
        wd_head_bias: float,
        search_head_only: bool,
    ) -> torch.optim.Optimizer:
        cfg = self.cfg
        seen: set = set()

        def _uniq(ps):
            out = []
            for p in ps:
                if p is None or id(p) in seen:
                    continue
                seen.add(id(p))
                out.append(p)
            return out

        lr_enc = float(cfg.lr_enc)
        lr_th = float(cfg.lr_theta)
        wd_head = float(cfg.wd_head)
        wd_enc = float(cfg.final_wd_enc if is_final else cfg.search_wd_enc)
        wd_th = float(cfg.final_wd_theta if is_final else cfg.search_wd_theta)

        head_w, head_b = self._split_head_params(model.head)
        head_w = _uniq(head_w)
        head_b = _uniq(head_b)

        enc_params = _uniq(
            [
                p
                for name in ("enc_alpha_raw", "enc_beta_raw")
                if hasattr(model, name) and isinstance(getattr(model, name), torch.nn.Parameter)
                for p in [getattr(model, name)]
            ]
        )
        theta_params = _uniq(
            [model.theta]
            if hasattr(model, "theta") and isinstance(model.theta, torch.nn.Parameter)
            else []
        )

        groups = []
        if hasattr(model, "logit_scale") and model.logit_scale.requires_grad:
            groups.append(
                {"params": _uniq([model.logit_scale]), "lr": lr_head, "weight_decay": 0.0}
            )
        if head_w:
            groups.append({"params": head_w, "lr": lr_head, "weight_decay": wd_head})
        if head_b:
            groups.append({"params": head_b, "lr": lr_head, "weight_decay": wd_head_bias})

        if not (search_head_only and not is_final):
            if enc_params:
                groups.append({"params": enc_params, "lr": lr_enc, "weight_decay": wd_enc})
            if theta_params:
                groups.append({"params": theta_params, "lr": lr_th, "weight_decay": wd_th})

        if not groups:
            groups = [
                {
                    "params": list(model.parameters()),
                    "lr": float(cfg.lr_vqc),
                    "weight_decay": float(cfg.wd_vqc),
                }
            ]

        return torch.optim.Adam(groups)

    # ── helpers: warmup head ──────────────────────────────────────────────────

    def _warmup_head(
        self,
        model,
        dl,
        pos_weight_eff,
        n_head,
        head_epochs,
        lr_head,
        wd_head_bias,
        is_final,
        allow_bias_learn_search,
        phase,
        seed_offset,
    ) -> None:
        cfg = self.cfg
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.head.parameters():
            p.requires_grad_(True)
        if model.head.bias is not None:
            if (not is_final) and bool(allow_bias_learn_search):
                model.head.bias.requires_grad_(True)
            else:
                model.head.bias.requires_grad_(False)

        hw, hb = self._split_head_params(model.head)
        groups = []
        if hw:
            groups.append({"params": hw, "lr": lr_head, "weight_decay": float(cfg.wd_head)})
        if hb:
            groups.append({"params": hb, "lr": lr_head, "weight_decay": wd_head_bias})
        opt_head = torch.optim.Adam(groups)
        crit = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_eff)

        model.train()
        for _he in range(int(head_epochs)):
            for bi, (xb, yb) in enumerate(dl):
                xb, yb = self._fix_batch(xb, yb, model)
                logits_b = model(xb)
                loss = crit(logits_b, yb)
                opt_head.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.head.parameters(), max_norm=1.0)
                opt_head.step()

                if bool(cfg.log_loss_per_iter):
                    log_every = int(cfg.loss_log_every)
                    if (bi % log_every) == 0 or bi == 0:
                        loss_f = float(loss.detach().cpu().item())
                        self._ema_head_loss = self._ema(
                            self._ema_head_loss, loss_f, float(cfg.loss_ema_alpha)
                        )
                        self.logger.log_to_file(
                            "loss_iter",
                            f"[head][iter] seed_off={seed_offset} he={_he} bi={bi} "
                            f"loss={loss_f:.6f} ema={self._ema_head_loss:.6f}",
                        )

                if (not is_final) and model.head.bias is not None:
                    bmax = float(cfg.search_head_bias_clamp)
                    with torch.no_grad():
                        model.head.bias.data.clamp_(-bmax, bmax)

                if (bi + 1) >= n_head:
                    break

    # ── helpers: recenter logits ──────────────────────────────────────────────

    def _recenter_logits(self, model, X_cal: torch.Tensor, seed_offset: int) -> None:
        cfg = self.cfg
        try:
            with torch.no_grad():
                model.eval()
                logits_c, _, n_c = self._predict_lp(
                    model,
                    X_cal,
                    cap=int(min(int(cfg.recenter_cap), int(X_cal.shape[0]))),
                )
                if (
                    n_c > 0
                    and hasattr(model, "head")
                    and getattr(model.head, "bias", None) is not None
                ):
                    mu = float(np.mean(np.asarray(logits_c, dtype=float)))
                    model.head.bias.data.sub_(float(mu))
                    self.logger.log_to_file(
                        "head", f"[recenter] seed_off={seed_offset} mu_logit={mu:.6f}"
                    )
        except Exception:
            pass

    def _recenter_before_thr(
        self,
        model,
        X_cal,
        Y_cal,
        logits_tr: np.ndarray,
        probs_tr: np.ndarray,
        yt_full: np.ndarray,
        attempt: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.cfg
        yt = yt_full

        try:
            if hasattr(model, "head") and model.head.bias is not None:
                mu = float(np.mean(np.asarray(logits_tr, dtype=float)))
                p95_abs = float(np.percentile(np.abs(np.asarray(logits_tr, dtype=float)), 95))
                sat_thr = float(cfg.collapse_saturation_abslogit_p95_thr)
                is_sat = bool(np.isfinite(p95_abs) and p95_abs >= sat_thr)
                mu_min = float(cfg.search_recenter_mu_min)
                mu_max = float(cfg.search_recenter_mu_clamp)
                damp = float(cfg.search_recenter_mu_damp)

                if (not is_sat) and (abs(mu) >= mu_min):
                    mu = float(np.clip(mu, -mu_max, mu_max))
                    delta = float(damp * mu)
                    bmax = float(cfg.search_head_bias_clamp)
                    with torch.no_grad():
                        model.head.bias.data.sub_(delta)
                        model.head.bias.data.clamp_(-bmax, bmax)
                    _, logits_tr, probs_tr, n = self._predict_lp_cal(
                        model,
                        X_cal,
                        cap=int(min(int(cfg.val_loss_cap), int(X_cal.shape[0]))),
                    )
                    yt = yt_full[:n]
                    self.logger.log_to_file(
                        "head",
                        f"[recenter_before_thr_safe] attempt={attempt} mu={mu:.6f} "
                        f"damp={damp:.3f} delta={delta:.6f} p95_abs={p95_abs:.3f}",
                    )
                else:
                    self.logger.log_to_file(
                        "head",
                        f"[recenter_before_thr_skip] attempt={attempt} mu={mu:.6f} "
                        f"mu_min={mu_min:.3f} p95_abs={p95_abs:.3f} is_sat={int(is_sat)}",
                    )
        except Exception:
            pass
        return logits_tr, probs_tr, yt

    # ── helpers: proxy train loop ─────────────────────────────────────────────

    def _proxy_train_loop(
        self,
        model,
        opt,
        dl,
        pos_weight_eff,
        n_epochs,
        n_vqc_batches,
        n_head,
        search_head_only,
        is_final,
        phase,
        seed_offset,
    ) -> None:
        cfg = self.cfg
        for ep in range(int(n_epochs)):
            epoch_losses = []
            model.train()
            for bi, (xb, yb) in enumerate(dl):
                xb, yb = self._fix_batch(xb, yb, model)
                logits = model(xb)

                if bool(cfg.use_focal):
                    loss = self._focal_loss_with_logits(
                        logits,
                        yb,
                        alpha=float(cfg.focal_alpha),
                        gamma=float(cfg.focal_gamma),
                        reduction="mean",
                        pos_weight=pos_weight_eff,
                    )
                else:
                    loss = F.binary_cross_entropy_with_logits(
                        logits.view(-1),
                        yb.view(-1),
                        pos_weight=pos_weight_eff,
                    )

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.head.parameters(), max_norm=float(cfg.clip_head)
                )
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.clip_all))
                opt.step()

                try:
                    epoch_losses.append(float(loss.detach().cpu().item()))
                except Exception:
                    pass

                if bool(cfg.log_loss_per_iter):
                    log_every = int(cfg.loss_log_every)
                    if (bi % log_every) == 0 or bi == 0:
                        loss_f = float(loss.detach().cpu().item())
                        self._ema_proxy_loss = self._ema(
                            self._ema_proxy_loss, loss_f, float(cfg.loss_ema_alpha)
                        )
                        self.logger.log_to_file(
                            "loss_iter",
                            f"[proxy][iter] seed_off={seed_offset} ep={ep} bi={bi} "
                            f"loss={loss_f:.6f} ema={self._ema_proxy_loss:.6f}",
                        )

                if (not is_final) and model.head.bias is not None:
                    bmax = float(cfg.search_head_bias_clamp)
                    with torch.no_grad():
                        model.head.bias.data.clamp_(-bmax, bmax)

                if search_head_only and not is_final:
                    default_steps = int(max(int(n_head), 2 * len(dl)))
                    head_steps = default_steps
                    if (bi + 1) >= int(head_steps):
                        break
                else:
                    if (bi + 1) >= n_vqc_batches:
                        break

            if bool(cfg.log_loss_per_epoch) and epoch_losses:
                m = float(np.mean(epoch_losses))
                s = float(np.std(epoch_losses))
                self.logger.log_to_file(
                    "loss_epoch",
                    f"[proxy][epoch] seed_off={seed_offset} ep={ep} "
                    f"loss_mean={m:.6f} loss_std={s:.6f} n_batches={len(epoch_losses)}",
                )

    # ── helpers: retry boost ──────────────────────────────────────────────────

    def _run_proxy_train_boost(
        self,
        model,
        dl,
        pos_weight_eff,
        lr_vqc,
        lr_head,
        n_epochs_run,
        n_batches_run,
        n_head_batches,
        wd_head_bias,
        search_head_only,
        is_final,
        seed_offset,
    ) -> None:
        cfg = self.cfg

        # optional head warmup
        if int(n_head_batches) > 0:
            for p in model.parameters():
                p.requires_grad_(False)
            for p in model.head.parameters():
                p.requires_grad_(True)

            hw, hb = self._split_head_params(model.head)
            opt_h = torch.optim.Adam(
                [
                    {"params": hw, "lr": float(lr_head), "weight_decay": float(cfg.wd_head)},
                    {"params": hb, "lr": float(lr_head), "weight_decay": float(wd_head_bias)},
                ]
            )
            crit_h = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_eff)
            model.train()
            for bi, (xb, yb) in enumerate(dl):
                xb, yb = self._fix_batch(xb, yb, model)
                loss = crit_h(model(xb), yb)
                opt_h.zero_grad(set_to_none=True)
                loss.backward()
                with torch.no_grad():
                    if hasattr(model, "head") and model.head.weight.grad is not None:
                        g_norm = model.head.weight.grad.norm().item()
                        try:
                            self.logger.log_to_file(
                                "head_grad",
                                f"[head_grad] seed_off={seed_offset} grad_norm={g_norm:.6f}",
                            )
                        except Exception:
                            pass
                torch.nn.utils.clip_grad_norm_(
                    model.head.parameters(), max_norm=float(cfg.clip_head)
                )
                opt_h.step()
                if (bi + 1) >= int(n_head_batches):
                    break

        # full train
        if (not is_final) and search_head_only:
            for p in model.parameters():
                p.requires_grad_(False)
            for p in model.head.parameters():
                p.requires_grad_(True)
            if model.head.bias is not None:
                model.head.bias.requires_grad_(True)
        else:
            for p in model.parameters():
                p.requires_grad_(True)

        lr_enc2 = float(lr_vqc)
        lr_th2 = float(lr_vqc)
        wd_enc2 = float(cfg.final_wd_enc if is_final else cfg.search_wd_enc)
        wd_th2 = float(cfg.final_wd_theta if is_final else cfg.search_wd_theta)

        seen2: set = set()

        def _u2(ps):
            out = []
            for p in ps:
                if p is None or id(p) in seen2:
                    continue
                seen2.add(id(p))
                out.append(p)
            return out

        hw2, hb2 = self._split_head_params(model.head)
        hw2 = _u2(hw2)
        hb2 = _u2(hb2)
        enc2 = _u2(
            [
                getattr(model, n)
                for n in ("enc_alpha_raw", "enc_beta_raw")
                if hasattr(model, n) and isinstance(getattr(model, n), torch.nn.Parameter)
            ]
        )
        theta2 = _u2(
            [model.theta]
            if hasattr(model, "theta") and isinstance(model.theta, torch.nn.Parameter)
            else []
        )

        groups2 = []
        if hw2:
            groups2.append(
                {"params": hw2, "lr": float(lr_head), "weight_decay": float(cfg.wd_head)}
            )
        if hb2:
            groups2.append(
                {"params": hb2, "lr": float(lr_head), "weight_decay": float(wd_head_bias)}
            )
        if not ((not is_final) and search_head_only):
            if enc2:
                groups2.append({"params": enc2, "lr": lr_enc2, "weight_decay": wd_enc2})
            if theta2:
                groups2.append({"params": theta2, "lr": lr_th2, "weight_decay": wd_th2})
        if not groups2:
            groups2 = [
                {
                    "params": list(model.parameters()),
                    "lr": float(cfg.lr_vqc),
                    "weight_decay": float(cfg.wd_vqc),
                }
            ]

        opt2 = torch.optim.Adam(groups2)
        clip_all = float(cfg.clip_all)
        for _ in range(int(n_epochs_run)):
            model.train()
            for bi, (xb, yb) in enumerate(dl):
                xb, yb = self._fix_batch(xb, yb, model)
                logits = model(xb)
                if bool(cfg.use_focal):
                    loss = self._focal_loss_with_logits(
                        logits,
                        yb,
                        alpha=float(cfg.focal_alpha),
                        gamma=float(cfg.focal_gamma),
                        reduction="mean",
                        pos_weight=pos_weight_eff,
                    )
                else:
                    loss = F.binary_cross_entropy_with_logits(
                        logits.view(-1), yb.view(-1), pos_weight=pos_weight_eff
                    )
                opt2.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.head.parameters(), max_norm=float(cfg.clip_head)
                )
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_all)
                opt2.step()
                if (bi + 1) >= int(n_batches_run):
                    break
                if (not is_final) and model.head.bias is not None:
                    bmax = float(cfg.search_head_bias_clamp)
                    with torch.no_grad():
                        model.head.bias.data.clamp_(-bmax, bmax)

    # ── helpers: threshold calibration ───────────────────────────────────────

    def _calibrate_threshold(
        self,
        yt: np.ndarray,
        probs_tr: np.ndarray,
        logits_tr: np.ndarray,
        thr_config: dict,
    ) -> tuple[float, dict, float, float, float]:
        cfg = self.cfg
        thr_mode = thr_config["thr_mode"]
        thr_min = float(cfg.thr_min)
        thr_max = float(cfg.thr_max)
        y_pos = int(yt.sum())
        y_tot = int(len(yt))
        single_class = (y_pos == 0) or (y_pos == y_tot)

        if single_class:
            thr_star = 0.5
            thr_info = {
                "t": 0.5,
                "mode": "single_class_calib",
                "viable_count": 0,
                "pmin": float(np.min(probs_tr)) if probs_tr.size else float("nan"),
                "pmax": float(np.max(probs_tr)) if probs_tr.size else float("nan"),
                "prange": float(np.max(probs_tr) - np.min(probs_tr))
                if probs_tr.size
                else float("nan"),
                "note": f"yt_single_class pos={y_pos}/{y_tot}",
            }
            sens_c, spec_c, fpr_c = rates_from_thr(yt, probs_tr, 0.5)
            try:
                self.logger.log_to_file(
                    "thr",
                    f"[thr] single_class_calib pos={y_pos}/{y_tot} "
                    f"sens={sens_c:.4f} spec={spec_c:.4f} fpr={fpr_c:.4f}",
                )
            except Exception:
                pass
            return thr_star, thr_info, sens_c, spec_c, fpr_c

        is_final = thr_config["is_final"]
        _grid = int(cfg.grid_size if is_final else cfg.search_grid_size)

        thr_star, thr_info = self._find_threshold(
            yt,
            probs_tr,
            mode=thr_mode,
            sens_target=float(thr_config["sens_target"]),
            fpr_max=float(thr_config["fpr_max"]),
            spec_min=float(thr_config["spec_min"]),
            grid_size=_grid,
            lam_spec=float(thr_config["lam_spec"]),
            lam_fpr=float(thr_config["lam_fpr"]),
            lam_sens=float(thr_config["lam_sens"]),
            logits=logits_tr,
            logger=self.logger,
            return_info=True,
            saturation_abslogit_p95_thr=float(cfg.collapse_saturation_abslogit_p95_thr),
            saturation_fallback_thr=float(cfg.saturation_fallback_thr),
        )
        thr_star = float(np.clip(float(thr_star), thr_min, thr_max))
        sens_c, spec_c, fpr_c = rates_from_thr(yt, probs_tr, thr_star)
        return thr_star, thr_info, sens_c, spec_c, fpr_c

    # ── helpers: collapse decision ────────────────────────────────────────────

    def _collapse_decision(
        self,
        logits_np: np.ndarray,
        probs_np: np.ndarray,
        auc_val: float | None,
        viable_count: int | None,
        fail_streak_clinical: int,
        fail_streak_auc_floor: int,
        phase: str,
    ) -> tuple[bool, dict]:
        cfg = self.cfg
        logits_np = np.asarray(logits_np, dtype=float).reshape(-1)
        probs_np = np.asarray(probs_np, dtype=float).reshape(-1)

        logit_margin = p95_p5(logits_np)
        prob_std = float(np.std(probs_np)) if probs_np.size else float("nan")
        logit_margin_min = float(cfg.collapse_logit_margin_min)
        prob_std_weak = float(cfg.collapse_prob_std_weak)
        auc_eps = float(cfg.collapse_auc_eps)
        streak_k = int(cfg.collapse_fail_streak_k)

        collapsed_strong = bool(np.isfinite(logit_margin) and logit_margin < logit_margin_min)

        auc_near_05 = (
            auc_val is not None and np.isfinite(auc_val) and abs(float(auc_val) - 0.5) <= auc_eps
        )
        no_viable = viable_count is not None and int(viable_count) == 0
        clinical_fail = bool(auc_near_05 and no_viable)

        logit_gate = float(cfg.collapse_logit_gate_mult)
        logit_gate_thr = logit_gate * logit_margin_min
        collapsed_weak = bool(
            np.isfinite(prob_std)
            and prob_std < prob_std_weak
            and fail_streak_auc_floor >= streak_k
            and np.isfinite(logit_margin)
            and logit_margin < logit_gate_thr
        )

        is_search = phase.startswith("search")
        auc_floor = float(cfg.search_auc_floor)
        auc_floor_streak = int(cfg.search_auc_floor_streak)
        auc_low = bool(
            is_search
            and auc_val is not None
            and np.isfinite(auc_val)
            and float(auc_val) < auc_floor
        )
        auc_floor_fail = bool(
            auc_low
            and fail_streak_auc_floor >= auc_floor_streak
            and (
                (np.isfinite(logit_margin) and logit_margin < logit_gate_thr)
                or (np.isfinite(prob_std) and prob_std < prob_std_weak)
            )
        )

        collapsed = bool(
            collapsed_strong
            or (clinical_fail and fail_streak_clinical >= streak_k)
            or collapsed_weak
            or auc_floor_fail
        )

        if collapsed:
            if collapsed_strong:
                reason = "logit_margin"
            elif auc_floor_fail:
                reason = f"auc_floor<{auc_floor:.3f}"
            elif clinical_fail and fail_streak_clinical >= streak_k:
                reason = "auc~0.5+viable=0_streak"
            elif collapsed_weak:
                reason = "prob_std_weak+streak+logit_gate"
            else:
                reason = "collapsed_other"
        else:
            reason = f"warn_auc_floor<{auc_floor:.3f}" if auc_low else "ok"

        return collapsed, {
            "collapsed": int(collapsed),
            "reason": str(reason),
            "logit_p95_p5": float(logit_margin) if np.isfinite(logit_margin) else float("nan"),
            "prob_std": float(prob_std) if np.isfinite(prob_std) else float("nan"),
            "auc": (None if auc_val is None else float(auc_val)),
            "viable_count": (None if viable_count is None else int(viable_count)),
            "fail_streak_clinical": int(fail_streak_clinical),
            "fail_streak_auc_floor": int(fail_streak_auc_floor),
            "auc_floor": (float(auc_floor) if is_search else None),
            "auc_low": bool(auc_low),
        }

    # ── helpers: logging ──────────────────────────────────────────────────────

    def _log_separability(self, model, X_cal, phase, seed_offset) -> None:
        try:
            model.eval()
            logits_dbg, probs_dbg, n_dbg = self._predict_lp(
                model,
                X_cal,
                cap=int(min(int(self.cfg.val_loss_cap), int(X_cal.shape[0]))),
            )
            logit_margin = float(p95_p5(np.asarray(logits_dbg, dtype=float)))
            prob_std = float(np.std(np.asarray(probs_dbg, dtype=float)))
            self.logger.log_to_file(
                "separability",
                f"[sep] phase={phase} seed_off={seed_offset} n={n_dbg} "
                f"logit_p95_p5={logit_margin:.4f} prob_std={prob_std:.4f} "
                f"p(min/mean/max)={float(np.min(probs_dbg)):.4f}/"
                f"{float(np.mean(probs_dbg)):.4f}/{float(np.max(probs_dbg)):.4f}",
            )
            if hasattr(model, "_dbg_raw_logits_std") and hasattr(model, "_dbg_scaled_logits_std"):
                ls_val = float(getattr(model, "_dbg_logit_scale_value", float("nan")))
                raw_std = float(getattr(model, "_dbg_raw_logits_std", float("nan")))
                scl_std = float(getattr(model, "_dbg_scaled_logits_std", float("nan")))
                self.logger.log_to_file(
                    "logit_scale",
                    f"[logit_scale] phase={phase} seed_off={seed_offset} "
                    f"scale={ls_val:.4f} raw_logits.std={raw_std:.6f} scaled_logits.std={scl_std:.6f}",
                )
        except Exception:
            pass

    def _log_health(self, logits_tr, probs_tr, yt, seed_offset, attempt) -> None:
        cfg = self.cfg
        try:
            ls = safe_stats(np.asarray(logits_tr, dtype=float))
            ps = safe_stats(np.asarray(probs_tr, dtype=float))
            lrm = float(cfg.collapse_range_min)
            lsm = float(cfg.collapse_std_min)
            hL = collapse_health(np.asarray(logits_tr, dtype=float), range_min=lrm, std_min=lsm)
            psm = float(cfg.collapse_prob_std_min_for_healthlog)
            hP = collapse_health(np.asarray(probs_tr, dtype=float), range_min=lrm, std_min=psm)
            self.logger.log_to_file(
                "health",
                f"[train-cap health seed_off={seed_offset} attempt={attempt}] n={len(yt)} "
                f"logits(min/mean/max)={ls['min']:.4f}/{ls['mean']:.4f}/{ls['max']:.4f} range={ls['range']:.4f} "
                f"| Lcollapsed={int(hL['collapsed'])} p95-p5={hL['p95_p5']:.4f} std={hL['std']:.4f} "
                f"|| probs(min/mean/max)={ps['min']:.4f}/{ps['mean']:.4f}/{ps['max']:.4f} range={ps['range']:.6f} "
                f"| Pcollapsed={int(hP['collapsed'])} p95-p5={hP['p95_p5']:.4f} std={hP['std']:.4f}",
            )
        except Exception:
            pass

    def _log_head_norm(self, model, seed_offset) -> None:
        cfg = self.cfg
        try:
            with torch.no_grad():
                if hasattr(model, "head") and hasattr(model.head, "weight"):
                    w_norm = model.head.weight.norm().item()
                    b_norm = (
                        model.head.bias.norm().item()
                        if (hasattr(model.head, "bias") and model.head.bias is not None)
                        else 0.0
                    )
                    self.logger.log_to_file(
                        "head",
                        f"[head] seed_off={seed_offset} weight_norm={w_norm:.4f} bias_norm={b_norm:.4f}",
                    )
                    d = max(1, int(model.head.weight.numel()))
                    thr = float(cfg.dead_head_norm_thr) * math.sqrt(d)
                    if w_norm < thr:
                        self.logger.log_to_file(
                            "head",
                            f"[head][WARN] small head norm: ||w||={w_norm:.4f} < {thr:.4f} (d={d})",
                        )
        except Exception:
            pass

    # ── misc ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _fix_batch(xb, yb, model):
        xb = xb.detach()
        xb.requires_grad_(False)
        if xb.dim() == 0:
            xb = xb.view(1, 1)
        elif xb.dim() == 1:
            xb = xb.unsqueeze(0) if xb.numel() == model.input_dim else xb.view(-1, 1)
        elif xb.dim() > 2:
            xb = xb.view(xb.shape[0], -1)
        if yb.dim() == 0:
            yb = yb.view(1, 1)
        elif yb.dim() == 1:
            yb = yb.view(-1, 1)
        return xb, yb

    @staticmethod
    def _ema(current: float | None, new_val: float, alpha: float) -> float:
        if current is None:
            return float(new_val)
        return (1.0 - alpha) * current + alpha * new_val
