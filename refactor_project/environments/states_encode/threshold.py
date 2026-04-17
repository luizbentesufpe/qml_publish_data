import numpy as np

from refactor_project.environments.qml.arch_util import safe_stats


def compute_f2_from_probs(y_true: np.ndarray, probs: np.ndarray, thr: float) -> float:
    y_pred = (probs >= thr).astype(np.int32)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    beta2 = 4.0
    denom = beta2 * prec + rec
    if denom <= 0:
        return 0.0
    return (1 + beta2) * prec * rec / denom


def _confusion_from_thr(
    y_true: np.ndarray, probs: np.ndarray, thr: float
) -> tuple[int, int, int, int]:
    y = y_true.reshape(-1).astype(np.int32)
    p = probs.reshape(-1).astype(np.float32)
    yp = (p >= float(thr)).astype(np.int32)

    tp = int(((y == 1) & (yp == 1)).sum())
    fp = int(((y == 0) & (yp == 1)).sum())
    tn = int(((y == 0) & (yp == 0)).sum())
    fn = int(((y == 1) & (yp == 0)).sum())
    return tp, fp, tn, fn


def _rates_from_thr(
    y_true: np.ndarray, probs: np.ndarray, thr: float
) -> tuple[float, float, float]:
    tp, fp, tn, fn = _confusion_from_thr(y_true, probs, thr)
    n_pos = tp + fn
    n_neg = tn + fp
    sens = tp / max(n_pos, 1)
    spec = tn / max(n_neg, 1)
    fpr = fp / max(n_neg, 1)
    return float(sens), float(spec), float(fpr)


def _balanced_acc_from_thr(y_true: np.ndarray, probs: np.ndarray, thr: float) -> float:
    sens, spec, _ = _rates_from_thr(y_true, probs, thr)
    return float(0.5 * (sens + spec))


def find_threshold(
    y_true,
    probs,
    mode="soft",
    sens_target=0.85,
    spec_min=0.80,
    fpr_max=0.10,
    grid_size=201,
    lam_spec=2.0,
    lam_fpr=2.0,
    lam_sens=1.0,
    logits=None,
    logger=None,
    return_info=False,
    saturation_abslogit_p95_thr: float | None = None,
    saturation_fallback_thr: float | None = None,
):
    """
    Threshold selection.

    mode="hard":
    - if exists viable -> pick best F2
    - else -> fallback BA (old behavior)

    mode="soft" (recommended for collapsed scores / RL proxy):
    maximize:
        J(t) = F2(t)
            - lam_spec * max(0, spec_min - spec(t))
            - lam_fpr  * max(0, fpr(t) - fpr_max)
            - lam_sens * max(0, sens_target - sens(t))

    This avoids 'viable=0 => BA lottery' and prevents thr* degenerating to 0.0/0.05.


    """
    y = np.asarray(y_true).reshape(-1).astype(int)
    p = np.asarray(probs).reshape(-1).astype(float)

    # defensive
    if p.size == 0:
        t = 0.5
        info = {"t": float(t), "note": "empty_probs", "viable_count": 0, "collapse_pen": 0.0}
        return (float(t), info) if bool(return_info) else float(t)

    # detect collapse (your exact symptom)
    pmin = float(np.min(p))
    pmax = float(np.max(p))
    prange = float(pmax - pmin)

    if prange < 1e-3:
        # stable, non-extreme fallback
        t = 0.5
        if (
            logits is not None
            and saturation_abslogit_p95_thr is not None
            and saturation_fallback_thr is not None
        ):
            p95_abs = float(np.percentile(np.abs(np.asarray(logits, dtype=float)), 95))
            if np.isfinite(p95_abs) and (p95_abs >= float(saturation_abslogit_p95_thr)):
                t = float(saturation_fallback_thr)

        sens, spec, fpr = _rates_from_thr(y, p, float(t))
        info = {
            "t": float(t),
            "mode": "collapse_fallback",
            "viable_count": 0,
            "collapse_pen": float(np.clip(1.0 - prange / 1e-3, 0.0, 1.0)),
            "sens": float(sens),
            "spec": float(spec),
            "fpr": float(fpr),
            "pmin": pmin,
            "pmax": pmax,
            "prange": prange,
        }
        if logger is not None:
            ps = safe_stats(p)
            logger.log_to_file(
                "thr",
                f"[thr] COLLAPSE probs(min/mean/max)={ps['min']:.4f}/{ps['mean']:.4f}/{ps['max']:.4f} "
                f"pctl(p5/p50/p95)={ps['p5']:.4f}/{ps['p50']:.4f}/{ps['p95']:.4f} range={ps['range']:.6f}",
            )
            if logits is not None:
                ls = safe_stats(np.asarray(logits, dtype=float))
                logger.log_to_file(
                    "thr",
                    f"[thr] COLLAPSE logits(min/mean/max)={ls['min']:.4f}/{ls['mean']:.4f}/{ls['max']:.4f} "
                    f"pctl(p5/p50/p95)={ls['p5']:.4f}/{ls['p50']:.4f}/{ls['p95']:.4f} range={ls['range']:.6f}",
                )
            logger.log_to_file(
                "thr",
                f"[thr] collapse_fallback t=0.5000 sens={float(sens):.4f} spec={float(spec):.4f} fpr={float(fpr):.4f}",
            )
        return (float(t), info) if bool(return_info) else float(t)

    # thresholds only where it matters (inside score support)
    lo = max(0.0, pmin - 1e-6)
    hi = min(1.0, pmax + 1e-6)
    thrs = np.linspace(lo, hi, int(grid_size))

    best = None
    best_viable = None
    best_ba = None
    viable_count = 0
    for t in thrs:
        t = float(t)
        sens, spec, fpr = _rates_from_thr(y, p, t)

        yp = (p >= t).astype(int)
        tp = int(((y == 1) & (yp == 1)).sum())
        fp = int(((y == 0) & (yp == 1)).sum())
        prec = float(tp) / float(max(tp + fp, 1))

        beta2 = 4.0
        f2 = (1.0 + beta2) * prec * sens / max(1e-12, (beta2 * prec + sens))
        ba = 0.5 * (sens + spec)

        viable = (
            (sens >= float(sens_target)) and (spec >= float(spec_min)) and (fpr <= float(fpr_max))
        )
        if viable:
            viable_count += 1

        if str(mode).lower().startswith("hard"):
            if viable and ((best_viable is None) or (f2 > best_viable["f2"])):
                best_viable = dict(
                    t=t,
                    f2=float(f2),
                    ba=float(ba),
                    sens=float(sens),
                    spec=float(spec),
                    fpr=float(fpr),
                    obj=float(f2),
                )
            if (best_ba is None) or (ba > best_ba["ba"]):
                best_ba = dict(
                    t=t,
                    f2=float(f2),
                    ba=float(ba),
                    sens=float(sens),
                    spec=float(spec),
                    fpr=float(fpr),
                    obj=float(ba),
                )
            continue

        # soft objective
        pen_spec = max(0.0, float(spec_min) - float(spec))
        pen_fpr = max(0.0, float(fpr) - float(fpr_max))
        pen_sens = max(0.0, float(sens_target) - float(sens))
        obj = (
            float(f2)
            - float(lam_spec) * pen_spec
            - float(lam_fpr) * pen_fpr
            - float(lam_sens) * pen_sens
        )

        cand = dict(
            t=t,
            f2=float(f2),
            ba=float(ba),
            sens=float(sens),
            spec=float(spec),
            fpr=float(fpr),
            obj=float(obj),
            viable=bool(viable),
        )

        # tie-break: prefer higher spec then sens (more stable)
        if (
            (best is None)
            or (cand["obj"] > best["obj"])
            or (
                abs(cand["obj"] - best["obj"]) < 1e-12
                and (cand["spec"], cand["sens"]) > (best["spec"], best["sens"])
            )
        ):
            best = cand

    if str(mode).lower().startswith("hard"):
        chosen = best_viable if (best_viable is not None) else best_ba
    else:
        chosen = best

    info = dict(chosen) if isinstance(chosen, dict) else {"t": float(chosen)}
    info.update(
        {
            "mode": str(mode),
            "viable_count": int(viable_count),
            "pmin": float(pmin),
            "pmax": float(pmax),
            "prange": float(prange),
            "collapse_pen": 0.0,
        }
    )

    if logger is not None:
        ps = safe_stats(p)
        p_rng = float(ps.get("max", pmax)) - float(ps.get("min", pmin))

        if str(mode).lower().startswith("soft"):
            pen_spec = max(0.0, float(spec_min) - float(info.get("spec", 0.0)))
            pen_fpr = max(0.0, float(info.get("fpr", 0.0)) - float(fpr_max))
            pen_sens = max(0.0, float(sens_target) - float(info.get("sens", 0.0)))

            logger.log_to_file(
                "thr",
                f"[thr] mode=soft viable={int(viable_count)}/{len(thrs)} "
                f"best t={float(info.get('t', 0.5)):.4f} obj={float(info.get('obj', 0.0)):.4f} "
                f"f2={float(info.get('f2', 0.0)):.4f} sens={float(info.get('sens', 0.0)):.4f} "
                f"spec={float(info.get('spec', 0.0)):.4f} fpr={float(info.get('fpr', 0.0)):.4f} | "
                f"pen(spec/fpr/sens)={pen_spec:.3f}/{pen_fpr:.3f}/{pen_sens:.3f} "
                f"lam(spec/fpr/sens)={float(lam_spec):.2f}/{float(lam_fpr):.2f}/{float(lam_sens):.2f} | "
                f"probs(min/mean/max)={ps.get('min', pmin):.4f}/{ps.get('mean', float(np.mean(p))):.4f}/{ps.get('max', pmax):.4f} "
                f"range={p_rng:.6f}",
            )
        else:
            logger.log_to_file(
                "thr",
                f"[thr] mode={str(mode)} viable={int(viable_count)}/{len(thrs)} "
                f"best t={float(info.get('t', 0.5)):.4f} obj={float(info.get('obj', 0.0)):.4f} "
                f"f2={float(info.get('f2', 0.0)):.4f} sens={float(info.get('sens', 0.0)):.4f} "
                f"spec={float(info.get('spec', 0.0)):.4f} fpr={float(info.get('fpr', 0.0)):.4f} | "
                f"probs(min/mean/max)={ps.get('min', pmin):.4f}/{ps.get('mean', float(np.mean(p))):.4f}/{ps.get('max', pmax):.4f} "
                f"range={p_rng:.6f}",
            )

        if logits is not None:
            ls = safe_stats(np.asarray(logits, dtype=float))
            logger.log_to_file(
                "thr",
                f"[thr] logits(min/mean/max)={ls['min']:.4f}/{ls['mean']:.4f}/{ls['max']:.4f} "
                f"range={ls['range']:.4f} pctl(p5/p50/p95)={ls['p5']:.4f}/{ls['p50']:.4f}/{ls['p95']:.4f}",
            )
            # keep same odd try/except behavior (defensive)
            try:
                _ = ls["range"]
            except Exception:
                logits_np = np.asarray(logits, dtype=float).reshape(-1)
                lmin = float(np.min(logits_np)) if logits_np.size else 0.0
                lmax = float(np.max(logits_np)) if logits_np.size else 0.0
                l_rng = float(lmax - lmin)
                logger.log_to_file(
                    "thr",
                    f"[thr] logits(min/mean/max)={ls.get('min', lmin):.4f}/{ls.get('mean', float(np.mean(logits_np))):.4f}/{ls.get('max', lmax):.4f} "
                    f"range={l_rng:.6f} pctl(p5/p50/p95)={ls.get('p5', float('nan')):.4f}/{ls.get('p50', float('nan')):.4f}/{ls.get('p95', float('nan')):.4f}",
                )

    t_star = float(info.get("t", 0.5))
    return (t_star, info) if bool(return_info) else t_star
