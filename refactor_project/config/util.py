from refactor_project.config.config import Config
from refactor_project.features.features import num_patches


def apply_overrides(cfg: Config, overrides: dict) -> Config:
    """
    Returns a NEW Config with overrides applied.
    Also enforces minimal consistency between patch-bank and feature bank sizing.
    """
    cfg2 = Config(**cfg.__dict__)
    for k, v in (overrides or {}).items():
        if not hasattr(cfg2, k):
            raise AttributeError(f"Config has no field '{k}' (override error)")
        setattr(cfg2, k, v)

    # Hard consistency: if patch-bank is enabled and compact_features True,
    # then feature bank domain should be P patches.
    if bool(cfg2.use_patch_bank) and bool(cfg2.patch_bank_compact_features):
        P = num_patches(28, 28, int(cfg2.patch_size), int(cfg2.patch_stride))
        cfg2.feature_bank_size = int(P)
        cfg2.feature_bank_min_size = int(min(int(cfg2.feature_bank_min_size), int(P)))
        if len(cfg2.feature_bank_schedule):
            cfg2.feature_bank_schedule = tuple(
                int(min(int(k), int(P))) for k in cfg2.feature_bank_schedule
            )
        else:
            cfg2.feature_bank_schedule = (int(P),)

    # If patch-bank is OFF, keep your qubit-based pixel-domain heuristic inside make_cfg_for_qubits()
    # (done per nq later). Here we only ensure no obviously invalid schedule:
    if len(cfg2.feature_bank_schedule) == 0:
        cfg2.feature_bank_schedule = (int(cfg2.feature_bank_size),)

    # Clamp min_size to size
    cfg2.feature_bank_min_size = int(
        min(int(cfg2.feature_bank_min_size), int(cfg2.feature_bank_size))
    )
    return cfg2


def scenario_tag(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(name))
    return safe[:120] if len(safe) > 120 else safe


def get_thr_targets(cfg: "Config") -> tuple[float, float, float]:
    """
    Returns (sens_target, spec_min, fpr_max) depending on cfg.phase.
    - search: relaxed targets (avoid viable=0)
    - final: strict/clinical targets
    Falls back to legacy cfg.sens_target / cfg.thr_spec_min / cfg.thr_fpr_max if missing.
    """
    phase = str(cfg.phase).lower()
    if phase == "search":
        sens = float(cfg.search_sens_target)
        spec = float(cfg.search_thr_spec_min)
        fpr = float(cfg.search_thr_fpr_max)
    else:
        sens = float(cfg.final_sens_target)
        spec = float(cfg.final_thr_spec_min)
        fpr = float(cfg.final_thr_fpr_max)
    return sens, spec, fpr
