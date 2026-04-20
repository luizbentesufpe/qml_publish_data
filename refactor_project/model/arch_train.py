from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from refactor_project.config.config import Config
from refactor_project.config.util import get_thr_targets
from refactor_project.environments.states_encode.state_encoder import sanitize_architecture
from refactor_project.environments.states_encode.threshold import _rates_from_thr, find_threshold
from refactor_project.model.model import CQV_End2End
from refactor_project.model.util.metrics import eval_metrics_final, focal_loss_with_logits
from refactor_project.model.util.util import compute_pos_weight, init_head_bias_with_prevalence
from refactor_project.util.save_image_article import save_circuit_image_paper
from refactor_project.util.util import Logger


def train_final_model_end2end(
    arch_mat: torch.Tensor,
    n_qubits: int,
    X_trval,
    Y_trval,
    X_te,
    Y_te,
    cfg: Config,
    logger: Logger,
    device: torch.device | str,
    noise: bool = False,
):
    # FINAL phase: strict/clinical threshold constraints
    try:
        cfg.phase = "final"
    except Exception:
        pass

    sens_tgt, spec_min, fpr_max = get_thr_targets(cfg)
    DEVICE = torch.device(device)
    arch_mat = sanitize_architecture(arch_mat, int(n_qubits)).to(DEVICE)

    model = CQV_End2End(
        arch_mat=arch_mat,
        n_qubits=n_qubits,
        enc_lambda=float(cfg.enc_lambda),
        diff_method="adjoint",
        input_dim=int(np.asarray(X_trval).shape[1]),
        enc_affine_mode=str(cfg.enc_affine_mode),
        enc_alpha_init=float(cfg.enc_alpha_init),
        enc_beta_init=float(cfg.enc_beta_init),
        enc_beta_max=float(cfg.enc_beta_max),
        use_batched_qnode=bool(cfg.use_batched_qnode),
        vqc_theta_init_std=float(cfg.vqc_theta_init_std),
        noise=noise,
        noise_p=cfg.noise_p,
    ).to(DEVICE)

    if bool(cfg.freeze_enc_params):
        if hasattr(model, "enc_alpha_raw"):
            model.enc_alpha_raw.requires_grad_(False)
        if hasattr(model, "enc_beta_raw"):
            model.enc_beta_raw.requires_grad_(False)
        logger.log_to_file(
            "enc_params", "[FIXED ENC] enc_alpha_raw e enc_beta_raw congelados (treino final)"
        )

    model.set_logit_scale_trainable(trainable=True)

    Y_trval_t = torch.tensor(Y_trval, dtype=torch.float32, device=DEVICE)
    Y_trval_np = (Y_trval_t.detach().cpu().numpy() > 0.5).astype(np.int32)

    init_head_bias_with_prevalence(model, Y_trval_np)
    pos_weight = compute_pos_weight(Y_trval_t, DEVICE)
    rng = np.random.default_rng(int(cfg.thr_calib_seed))
    ybin = (np.asarray(Y_trval).reshape(-1) > 0.5).astype(np.int32)
    idx = np.arange(len(ybin))

    # stratified split
    idx_pos = idx[ybin == 1]
    idx_neg = idx[ybin == 0]
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)

    calib_frac = float(cfg.thr_calib_frac)
    n_cal_pos = int(max(1, round(calib_frac * len(idx_pos))))
    n_cal_neg = int(max(1, round(calib_frac * len(idx_neg))))

    cal_idx = np.concatenate([idx_pos[:n_cal_pos], idx_neg[:n_cal_neg]])
    fit_idx = np.setdiff1d(idx, cal_idx)

    X_fit, Y_fit = np.asarray(X_trval)[fit_idx], np.asarray(Y_trval)[fit_idx]
    X_cal, Y_cal = np.asarray(X_trval)[cal_idx], np.asarray(Y_trval)[cal_idx]

    X_fit_t = torch.tensor(X_fit, dtype=torch.float32, device=DEVICE)
    Y_fit_t = torch.tensor(Y_fit, dtype=torch.float32, device=DEVICE)

    dl = DataLoader(
        TensorDataset(X_fit_t, Y_fit_t),
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    # warmup head only (few batches, publication-stable)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.head.parameters():
        p.requires_grad_(True)

    model.logit_scale.requires_grad_(False)

    if model.head.bias is not None:
        model.head.bias.requires_grad_(False)

    opt_head = torch.optim.Adam(model.head.parameters(), lr=1e-3)
    crit_warm = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model.train()

    n_head_batches = int(max(1, cfg.inner_train_batches_head))

    for bi, (xb, yb) in enumerate(dl):
        logits = model(xb)
        loss = crit_warm(logits, yb)
        opt_head.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.head.parameters(), max_norm=1.0)
        opt_head.step()
        if (bi + 1) >= n_head_batches:
            break

    for p in model.parameters():
        p.requires_grad_(True)

    if bool(cfg.freeze_enc_params):
        if hasattr(model, "enc_alpha_raw"):
            model.enc_alpha_raw.requires_grad_(False)
        if hasattr(model, "enc_beta_raw"):
            model.enc_beta_raw.requires_grad_(False)

    if model.head.bias is not None:
        model.head.bias.requires_grad_(True)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.final_lr_vqc)
    # final training
    for ep in range(cfg.final_epochs):
        model.train()
        for xb, yb in dl:
            logits = model(xb)
            if bool(cfg.use_focal):
                loss = focal_loss_with_logits(
                    logits,
                    yb,
                    alpha=float(cfg.focal_alpha),
                    gamma=float(cfg.focal_gamma),
                    reduction="mean",
                    pos_weight=pos_weight,
                )
            else:
                loss = F.binary_cross_entropy_with_logits(
                    logits.view(-1), yb.view(-1), pos_weight=pos_weight
                )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            if (not cfg.phase.startswith("final")) and (model.head.bias is not None):
                bmax = float(cfg.head_bias_abs_max)
                with torch.no_grad():
                    model.head.bias.data.clamp_(-bmax, bmax)

        if ep % 3 == 0:
            # calibrate thr* on train (trval) each check
            model.eval()
            Xtr_t = torch.tensor(X_fit, dtype=torch.float32, device=DEVICE)
            Ytr_t = torch.tensor(Y_fit, dtype=torch.float32, device=DEVICE)

            logits_tr = model(Xtr_t).detach().cpu().numpy().reshape(-1)
            probs_tr = (1.0 / (1.0 + np.exp(-logits_tr))).astype(np.float32).reshape(-1)

            ytr = (Ytr_t.detach().cpu().numpy().reshape(-1) > 0.5).astype(np.int32)
            thr_star = find_threshold(
                ytr,
                probs_tr,
                mode=str(cfg.thr_mode),
                sens_target=float(sens_tgt),
                grid_size=cfg.grid_size,
                fpr_max=float(fpr_max),
                spec_min=float(spec_min),
                lam_spec=float(cfg.thr_lam_spec),
                lam_fpr=float(cfg.thr_lam_fpr),
                lam_sens=float(cfg.thr_lam_sens),
                logits=logits_tr,
                logger=logger,
                return_info=False,
            )
            sens_r, spec_r, fpr_r = _rates_from_thr(ytr, probs_tr, float(thr_star))
            auc_m, sens_m = eval_metrics_final(model, X_fit, Y_fit, thr=thr_star, device=DEVICE)

            logger.log_to_file(
                "final_train",
                f"[ep={ep:02d}] thr*={thr_star:.3f} AUC_trval={auc_m:.4f} SENS={sens_r:.4f} SPEC={spec_r:.4f} FPR={fpr_r:.4f}",
            )

    # --- calibrate thr* on X_cal/Y_cal ---
    model.eval()
    Xcal_t = torch.tensor(X_cal, dtype=torch.float32, device=DEVICE)
    Ycal_t = torch.tensor(Y_cal, dtype=torch.float32, device=DEVICE)
    logits_cal = model(Xcal_t).detach().cpu().numpy().reshape(-1)
    # probs_cal  = torch.sigmoid(torch.tensor(logits_cal)).numpy().reshape(-1)
    probs_cal = (1.0 / (1.0 + np.exp(-logits_cal))).astype(np.float32).reshape(-1)
    ycal = (Ycal_t.detach().cpu().numpy().reshape(-1) > 0.5).astype(np.int32)

    thr_star = find_threshold(
        ycal,
        probs_cal,
        mode=str(cfg.thr_mode),
        sens_target=float(sens_tgt),
        grid_size=cfg.grid_size,
        fpr_max=float(fpr_max),
        spec_min=float(spec_min),
        lam_spec=float(cfg.thr_lam_spec),
        lam_fpr=float(cfg.thr_lam_fpr),
        lam_sens=float(cfg.thr_lam_sens),
        logits=logits_cal,
        logger=logger,
        return_info=False,
    )
    auc_te, sens_te = eval_metrics_final(model, X_te, Y_te, thr=thr_star, device=DEVICE)
    logger.log_to_file(
        "final_test", f"[TEST] thr*={thr_star:.3f} AUC={auc_te:.4f} SENS@thr*={sens_te:.4f}"
    )
    try:
        with torch.no_grad():
            _enc = {}
            if hasattr(model, "enc_alpha_raw"):
                _a = F.softplus(model.enc_alpha_raw.detach().cpu()) + 1e-6
                _b = torch.tanh(model.enc_beta_raw.detach().cpu()) * float(
                    getattr(model, "enc_beta_max", 1.0)
                )
                _enc["alpha"] = _a.numpy()  # shape depende do enc_affine_mode
                _enc["beta"] = _b.numpy()
                _enc["mode"] = str(getattr(model, "enc_affine_mode", "unknown"))
                _enc["thr_star"] = float(thr_star)
                _enc["auc_te"] = float(auc_te)
                _enc["sens_te"] = float(sens_te)
                if hasattr(model, "theta"):
                    _enc["theta"] = model.theta.detach().cpu().numpy()
                _enc["head_weight"] = model.head.weight.detach().cpu().numpy()
                _enc["head_bias"] = model.head.bias.detach().cpu().numpy()

            _pt_path = Path(str(logger.log_dir)) / "enc_params_final.pt"
            torch.save(_enc, str(_pt_path))
            logger.log_to_file(
                "enc_params",
                f"[SAVED] enc_params_final.pt | "
                f"alpha_mean={float(_enc['alpha'].mean()):.4f} "
                f"alpha_std={float(_enc['alpha'].std()):.4f} "
                f"beta_mean={float(_enc['beta'].mean()):.4f}",
            )
    except Exception as _e:
        logger.log_to_file("enc_params", f"[WARN] could not save enc_params: {_e}")

    try:
        safe_prefix = "final_circuit"
        out_img = logger.log_dir / f"circuit_{safe_prefix}_nq{n_qubits}_posttrain.png"
        _draw_model = CQV_End2End(
            arch_mat=arch_mat,
            n_qubits=n_qubits,
            enc_lambda=float(cfg.enc_lambda),
            diff_method="backprop",  # default.qubit: único device com draw_mpl
            input_dim=int(np.asarray(X_trval).shape[1]),
            enc_affine_mode=str(cfg.enc_affine_mode),
            enc_alpha_init=float(cfg.enc_alpha_init),
            enc_beta_init=float(cfg.enc_beta_init),
            enc_beta_max=float(cfg.enc_beta_max),
            use_batched_qnode=False,  # single-sample para o draw
            vqc_theta_init_std=float(cfg.vqc_theta_init_std),
        ).to("cpu")
        # Copia pesos treinados (theta, enc_alpha_raw, enc_beta_raw, head, logit_scale)
        _draw_model.load_state_dict(model.cpu().state_dict(), strict=False)
        save_circuit_image_paper(
            _draw_model,
            x_ref=np.asarray(X_trval)[0],
            out_path=str(out_img),
            title=f"Circuito VQC (nq={n_qubits}) — pós-treino",
        )
        logger.log_to_file("circuit", f"[SAVED] {out_img}")
        del _draw_model
    except Exception as e:
        logger.log_to_file("circuit", f"[WARN] could not save posttrain circuit: {e}")

    return auc_te, sens_te, float(thr_star)
