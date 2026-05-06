import numpy as np
import torch

from refactor_project.config.config import Config
from refactor_project.environments.qml.env import QMLEnvEnd2End
from refactor_project.environments.states_encode.state_encoder import state_to_vec
from refactor_project.model.builder.agent import DDQNAgent
from refactor_project.model.util.metrics import _bacc, _proxy_score_search
from refactor_project.util.util import Logger, RunningStd, set_seeds


def run_arch_search_end2end(
    X_tr,
    Y_tr,
    X_val,
    Y_val,
    cfg: Config,
    logger: Logger,
    seed: int = 0,
    device: torch.device | str = "cpu",
):
    print(device)
    # SEARCH phase: relax threshold constraints
    try:
        cfg.phase = "search"
    except Exception:
        pass
    set_seeds(seed)
    DEVICE = torch.device(device)

    env = QMLEnvEnd2End(X_tr, Y_tr, X_val, Y_val, cfg, logger=logger, seed=seed, device=DEVICE)
    env._best_adjusted_score = -1.0

    agent = DDQNAgent(cfg, n_actions=env.N_ACTIONS, device=DEVICE)

    run_score_std = RunningStd()
    best_arch = None
    best_score = -1.0
    best_nq = cfg.start_qubits
    best_proxy = None

    # PRE-WARMUP
    _vqc_init_ok = False  # ← inicializa ANTES do check
    _vqc_init_attempts = 0
    _vqc_max_attempts = int(cfg.vqc_init_max_attempts)

    _n_ops_initial = int(np.sum(env.state[2, :] != 0))

    if _n_ops_initial == 0:
        logger.log_to_file("rl", "[vqc_init] empty init state (all-NOP) — skip pre-warmup")
        _vqc_init_ok = True

    while not _vqc_init_ok and _vqc_init_attempts < _vqc_max_attempts:
        _vqc_init_attempts += 1
        _auc0, _sens0, _spec0, _thr0, _, _, _tinfo0 = env.terminal_evaluate()
        _cdbg = _tinfo0.get("collapse_dbg", {}) if isinstance(_tinfo0, dict) else {}
        _logit_margin = float(_cdbg.get("logit_p95_p5", 0.0))

        if _auc0 > 0.55 or _logit_margin > 0.10:
            # Inicialização produz sinal — continuar normalmente
            _vqc_init_ok = True
            logger.log_to_file(
                "rl",
                f"[vqc_init] attempt={_vqc_init_attempts} OK "
                f"auc0={_auc0:.4f} logit_margin={_logit_margin:.4f}",
            )
        else:
            # Colapso detectado — reinicializar theta com nova seed
            logger.log_to_file(
                "rl",
                f"[vqc_init] attempt={_vqc_init_attempts} COLLAPSE "
                f"auc0={_auc0:.4f} logit_margin={_logit_margin:.4f} → reinit theta",
            )
            _new_seed = seed + _vqc_init_attempts * 1000
            set_seeds(_new_seed)
            # Reinicializa só o modelo interno do env (não recria o env inteiro)
            env._reinit_vqc_theta()  # ← método a adicionar no env (ver abaixo)

    collapse_count = 0
    collapse_streak = 0
    collapse_streak_max = cfg.collapse_streak_max

    # Early stopping state
    _no_improve_count = 0
    _last_best_score_for_es = -1.0

    ep_rewards = []
    for ep in range(cfg.episodes):
        T1 = cfg.curriculum_T1
        T2 = cfg.curriculum_T2
        if ep < T1:
            env.set_metric_weight(cfg.curriculum_w_metric_early)
        elif ep < T2:
            env.set_metric_weight(cfg.curriculum_w_metric_mid)
        else:
            env.set_metric_weight(cfg.curriculum_w_metric_late)

        if cfg.feature_bank_decay_enabled and len(cfg.feature_bank_schedule) > 0:
            stage = ep // max(int(cfg.feature_bank_decay_every), 1)
            stage = int(np.clip(stage, 0, len(cfg.feature_bank_schedule) - 1))

            sched = cfg.feature_bank_schedule
            k_now = int(sched[stage])
            k_now = int(
                np.clip(k_now, int(env.feature_bank_min_eff), int(env.feature_bank_size_eff))
            )
            env.set_current_bank_k(k_now)

        if (
            cfg.feature_bank_update.lower() == "saliency"
            and ep > 0
            and (ep % max(int(cfg.feature_bank_rescore_every), 1) == 0)
        ):
            arch_use = best_arch if best_arch is not None else env.state
            env.maybe_rescore_feature_bank_saliency(arch_use)

        s = env.reset()
        done = False
        last_metric = 0.0
        ep_ret = 0.0

        # Garantir que o atributo "last_proxy_score" exista no env (pode ser usado na função de recompensa)
        if not hasattr(env, "last_proxy_score"):
            env.last_proxy_score = (
                0.5  # ← inicializa o atributo no env (pode ser usado na função de recompensa)
            )

        while not done:
            s_vec = state_to_vec(s, last_metric, env.current_n_qubits, cfg)
            s_in = torch.tensor(s_vec.reshape(1, -1), dtype=torch.float32, device=DEVICE)

            valid_mask = env.valid_action_mask()
            a = agent.select_action(s_in, valid_mask)

            s1, r, done, info = env.step(a)

            # Terminal reward injection
            terminal_bonus = 0.0
            terminal_info = None

            if bool(done):
                auc_t, sens_t, spec_t, thr_star, depth_t, cnot_t, tinfo = env.terminal_evaluate()
                logger.log_to_file(
                    "rl",
                    f">>> [ep {ep + 1:03d}] TERMINAL EVAL: AUC={float(auc_t):.4f} SENS={float(sens_t):.4f} SPEC={float(spec_t):.4f} THR*={float(thr_star):.3f}",
                )
                score = 0.0

                # Balanced accuracy exists and is used later (do NOT remove)
                bacc = _bacc(float(sens_t), float(spec_t))
                phase_l = str(cfg.phase).lower()
                logit_margin = None

                try:
                    cdbg = tinfo.get("collapse_dbg", {}) if isinstance(tinfo, dict) else {}
                    if isinstance(cdbg, dict):
                        logit_margin = cdbg.get("logit_p95_p5", None)
                except Exception:
                    logit_margin = None

                proxy_score, proxy_dbg = _proxy_score_search(
                    auc=float(auc_t),
                    sens=float(sens_t),
                    spec=float(spec_t),
                    logit_margin=(float(logit_margin) if logit_margin is not None else None),
                    phase=str(phase_l),
                    cfg=cfg,
                )

                # keep debug packed (so you can inspect in logs)
                if isinstance(tinfo, dict):
                    tinfo["proxy_dbg"] = dict(proxy_dbg)

                w_bacc = cfg.terminal_w_bacc
                proxy_score = float(
                    np.clip(float(proxy_score) + w_bacc * (float(bacc) - 0.5), 0.0, 1.0)
                )
                score = float(proxy_score)

                # degenerate threshold penalty
                degenerate_penalty = 0.0
                lo = cfg.thr_degenerate_lo
                hi = cfg.thr_degenerate_hi
                if thr_star < lo or thr_star > hi:
                    degenerate_penalty = -float(cfg.lambda_thr)

                # Delta-proxy terminal reward (reduces plateus, encourages exploration)
                use_delta = cfg.terminal_use_delta_proxy
                ema = float(cfg.terminal_proxy_ema)
                prev_proxy = float(env.last_proxy_score)
                if ema > 0.0:
                    baseline = prev_proxy
                    new_base = float((1.0 - ema) * prev_proxy + ema * float(proxy_score))
                    env.last_proxy_score = float(new_base)
                else:
                    baseline = prev_proxy  # no baseline → full reward is the (clipped) proxy score
                    env.last_proxy_score = float(proxy_score)
                delta_proxy = (
                    float(proxy_score - baseline) if use_delta else float(proxy_score - 0.5)
                )
                try:
                    collapse_pen = (
                        float(tinfo.get("collapse_pen", 0.0)) if isinstance(tinfo, dict) else 0.0
                    )
                except Exception:
                    collapse_pen = 0.0

                if collapse_pen <= 0.0:
                    spec_floor = float(cfg.terminal_spec_floor)
                    if spec_t < spec_floor:
                        collapse_pen = (
                            float(cfg.terminal_collapse_penalty)
                            * float(spec_floor - spec_t)
                            / max(spec_floor, 1e-6)
                        )

                if float(spec_t) < cfg.terminal_spec_floor:
                    collapse_count += 1
                    collapse_streak = min(collapse_streak + 1, collapse_streak_max)
                else:
                    collapse_streak = max(collapse_streak - 1, 0)

                # With proxy_aggregation="min", proxy_score is systematically conservative.
                # Small K early avoids large negative terminals that swamp the replay buffer.
                # After the ramp, std-adaptive scaling continues to keep signal well-calibrated.
                K_start = cfg.terminal_K_start
                K_end = cfg.terminal_K_end
                curriculum_K_T = cfg.terminal_curriculum_K_T
                adaptative = cfg.terminal_adaptive_K

                ramp_frac = float(np.clip(ep / max(curriculum_K_T, 1), 0.0, 1.0))
                K_ramp = float(K_start + ramp_frac * (K_end - K_start))

                if adaptative:
                    d_center = float(proxy_score - 0.5)
                    run_score_std.update(d_center)
                    warm = int(cfg.terminal_K_warmup)

                    if (run_score_std.n > warm) and run_score_std.std > 0:
                        K = float(K_ramp)
                    else:
                        target_std = float(cfg.terminal_target_std)
                        scale = float(target_std) / max(run_score_std.std, 1e-6)
                        K = float(K_ramp) * scale
                else:
                    K = float(K_ramp)

                # delta_proxy alone stays ≤ 0 for many early episodes when min-agg is used.
                # abs_signal adds a small positive reward whenever proxy_score > agg_floor
                # (the expected score at near-random AUC ≈ 0.5). This prevents the reward
                # from being all-negative before the curriculum K has ramped up.
                agg_floor = float(cfg.terminal_agg_floor)
                K_abs = float(cfg.terminal_K_abs)
                abs_signal = float(K_abs * max(0.0, float(proxy_score) - agg_floor))
                terminal_bonus = float(
                    K * float(delta_proxy)
                    + abs_signal
                    + float(degenerate_penalty)
                    - float(collapse_pen)
                )

                # Level 2: penalty instabily proxy
                try:
                    std_proxy = (
                        float(tinfo.get("std_proxy", 0.0)) if isinstance(tinfo, dict) else 0.0
                    )
                except Exception:
                    std_proxy = 0.0

                # terminal bonus
                lambda_var = float(cfg.lambda_var)
                std_proxy_threshold = float(cfg.std_proxy_threshold)
                if std_proxy > std_proxy_threshold:
                    terminal_bonus *= std_proxy_threshold / max(float(std_proxy), 1e-6)
                else:
                    terminal_bonus -= lambda_var * float(std_proxy)

                # thr_std is populated by env.terminal_evaluate() and placed in tinfo["thr_std"].
                # Penalises circuits whose optimal threshold varies across seeds — a signal of
                # fragile separation that would be unreliable in complex  deployment.
                # 43.9% of proxy evals in Cross/Circle logs had prob_std < 0.02 (prob collapse);
                # unstable thr* is the downstream symptom of that fragility.
                try:
                    thr_std = float(tinfo.get("thr_std", 0.0)) if isinstance(tinfo, dict) else 0.0
                except Exception:
                    thr_std = 0.0
                thr_std_threshold = cfg.thr_std_threshold
                lambda_thr_std = cfg.lambda_thr_std
                if thr_std > thr_std_threshold:
                    terminal_bonus -= lambda_thr_std * (thr_std - thr_std_threshold)

                # Applied once per terminal; reads the arch hash set maintained by env.
                # env.terminal_evaluate() adds the current hash before returning, so
                # len(_arch_hash_set) > 1 means the arch was seen in a previous episode.
                try:
                    _repeat_arch_pen = cfg.repeat_arch_penalty
                    _arch_hash_set = getattr(env, "_terminal_arch_hashes", set())
                    _arch_hash_now = hash(env.state.tobytes())
                    if len(_arch_hash_set) > 1 and _arch_hash_now in _arch_hash_set:
                        terminal_bonus -= _repeat_arch_pen
                        env._ep_reward_sums["repeat_arch"] = (
                            env._ep_reward_sums.get("repeat_arch", 0.0) - _repeat_arch_pen
                        )
                    # Bound memory: clear hash set when it exceeds repeat_arch_window.
                    _win = cfg.repeat_arch_window
                    if len(_arch_hash_set) > _win:
                        _arch_hash_set.clear()
                except Exception:
                    pass

                try:
                    floor = cfg.terminal_spec_floor
                    if float(spec_t) < floor:
                        bump = cfg.eps_boost_on_collapse
                        agent.eps = float(min(1.0, float(agent.eps) + abs(bump)))
                    # If collapse keeps happening, slow down eps decay by effectively raising eps_end for a while
                    if collapse_streak >= cfg.collapse_streak_trigger:
                        agent.eps = float(
                            min(1.0, max(agent.eps, float(cfg.eps_min_when_collapsing)))
                        )
                except Exception:
                    pass

                clip_val = cfg.terminal_clip
                terminal_bonus = float(np.clip(terminal_bonus, -clip_val, clip_val))

                # save for logging best arch
                effective_score = float(np.clip(proxy_score - collapse_pen, 0.0, 1.0))
                env.last_ep_score = effective_score
                terminal_info = {
                    "terminal_auc": float(auc_t),
                    "terminal_sens": float(sens_t),
                    "terminal_score": float(score),
                    "terminal_spec": float(spec_t),
                    "thr_star": float(thr_star),
                    "proxy_score": float(proxy_score),
                    "delta_proxy": float(delta_proxy),
                    "degenerate_penalty": float(degenerate_penalty),
                    "effective_score": float(effective_score),
                    "balanced_acc": float(bacc),
                    "collapse_pen": float(collapse_pen),
                    "std_proxy": float(std_proxy),
                    "terminal_bonus": float(terminal_bonus),
                    "terminal_K": float(K),
                    "terminal_K_ramp": float(K_ramp),
                    "terminal_abs_signal": float(abs_signal),
                    "terminal_agg_floor": float(agg_floor),
                    "terminal_depth": depth_t,
                    "terminal_cnot": cnot_t,
                }
                std_proxy_gate = cfg.std_proxy_threshold
                reliable_score_min = cfg.reliable_score_min
                use_sw_gate = cfg.use_stability_weighted_gate

                if use_sw_gate:
                    stability_weight = float(
                        max(0.0, 1.0 - float(std_proxy) / max(std_proxy_gate, 1e-9))
                    )
                    adjusted_score = float(effective_score) * stability_weight
                    _gate_pass = (
                        float(std_proxy) < std_proxy_gate
                    )  # ainda necessário para reliable_archs
                else:
                    stability_weight = 1.0 if float(std_proxy) < std_proxy_gate else 0.0
                    adjusted_score = float(effective_score) * stability_weight
                    _gate_pass = float(std_proxy) < std_proxy_gate

                if not hasattr(env, "_best_adjusted_score"):
                    env._best_adjusted_score = -1.0

                if adjusted_score > float(env._best_adjusted_score):
                    env._best_adjusted_score = float(adjusted_score)
                    best_score = float(effective_score)  # continua logando o score real
                    best_arch = env.state.copy()
                    best_nq = env.current_n_qubits
                    best_proxy = float(effective_score)

                # log comparativo para auditoria (permite comparar modos post-hoc)
                logger.log_to_file(
                    "rl",
                    f"[ep {ep + 1:03d}] score={effective_score:.4f} best={best_score:.4f} "
                    f"nq={env.current_n_qubits} "
                    f"std_proxy={std_proxy:.4f} stability_w={stability_weight:.3f} "
                    f"adjusted={adjusted_score:.4f} best_adjusted={env._best_adjusted_score:.4f}",
                )
                if _gate_pass and float(effective_score) > reliable_score_min:
                    logger.log_to_file(
                        "reliable_archs",
                        f"[ep {ep + 1:03d}] score={effective_score:.4f} std_proxy={std_proxy:.4f} "
                        f"stability_w={stability_weight:.3f} adjusted={adjusted_score:.4f} "
                        f"nq={env.current_n_qubits} thr_std={thr_std:.4f} "
                        f"arch_hash={hash(env.state.tobytes())}",
                    )
                # injeta o bônus no último reward do episódio
                r = float(r) + float(terminal_bonus)
                info = dict(info)
                info["terminal"] = terminal_info

                # Important: account terminal bonus in per-episode breakdown
                try:
                    env._ep_reward_sums["terminal"] += float(terminal_bonus)
                except Exception:
                    pass

            ep_ret += float(r)

            try:
                auc_v = float(info.get("auc_val", 0.5))
                sens_v = float(info.get("sens_val", 0.0))
                # prefer env.last_spec if you track it there; else allow info["spec_val"]
                spec_v = float(getattr(env, "last_spec", info.get("spec_val", 0.0)))
                last_metric = 0.5 * float(auc_v) + 0.5 * float(
                    np.clip(sens_v + spec_v - 1.0, -1.0, 1.0)
                )
            except Exception:
                last_metric = 0.0

            s1_vec = state_to_vec(s1, last_metric, env.current_n_qubits, cfg)

            agent.nbuf.push(s_vec, a, r, s1_vec, done)
            if agent.nbuf.is_ready():
                ns = agent.nbuf.pop_nstep(cfg.gamma())
                agent.replay.push(*ns)
                agent.update(cfg.gamma())

            s = s1
            ep_rewards.append(r)

        sums = env.flush_episode_reward_sums()
        score_print = float(env.last_ep_score) if (env.last_ep_score is not None) else float("nan")
        logger.log_to_file(
            "rl",
            "[ep %03d] score=%.4f best=%.4f nq=%d ep_ret=%.3f | "
            "metric=%.3f depth=%.3f cnot=%.3f qubit=%.3f rot=%.3f dead=%.3f budget=%.3f repeat=%.3f terminal=%.3f"
            % (
                ep + 1,
                score_print,
                best_score,
                int(best_nq),
                float(ep_ret),
                float(sums.get("metric", 0.0)),
                float(sums.get("depth", 0.0)),
                float(sums.get("cnot", 0.0)),
                float(sums.get("qubit", 0.0)),
                float(sums.get("rot", 0.0)),
                float(sums.get("dead", 0.0)),
                float(sums.get("budget", 0.0)),
                float(sums.get("repeat", 0.0)),
                float(sums.get("terminal", 0.0)),
            ),
        )
        _es_patience = cfg.early_stop_patience
        _es_min_eps = cfg.early_stop_min_eps

        if best_score > _last_best_score_for_es + 1e-6:
            _no_improve_count = 0
            _last_best_score_for_es = best_score
        else:
            _no_improve_count += 1
        if ep >= _es_min_eps and _no_improve_count >= _es_patience:
            logger.log_to_file(
                "rl",
                f"[early_stop] ep={ep + 1} sem melhora em {_no_improve_count} eps "
                f"(patience={_es_patience}, best={best_score:.4f}). Encerrando busca.",
            )
            break

        if (ep + 1) % int(max(1, cfg.early_stop_log)) == 0:
            logger.log_to_file(
                "rl",
                f"[health] collapse_count={int(collapse_count)} / {int(ep + 1)} (spec < terminal_spec_floor)",
            )

        if env.last_ep_score is not None:
            logger.log_to_file(
                "rl_terminal", f"[ep {ep + 1:03d}] terminal_score={env.last_ep_score:.4f}"
            )

    if best_arch is None:
        best_arch = env.state.copy()
        best_nq = env.current_n_qubits
    arch_mat = torch.tensor(best_arch, dtype=torch.int64)

    return arch_mat, int(best_nq), (None if best_proxy is None else float(best_proxy))
