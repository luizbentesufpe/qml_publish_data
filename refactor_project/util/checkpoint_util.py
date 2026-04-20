from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _checkpoint_path(sc_dir: Path, nq: int, seed: int) -> Path:
    return sc_dir / f"checkpoint_nq{nq}_seed{seed}.json"


def save_checkpoint(sc_dir: Path, nq: int, seed: int, result: dict[str, Any]) -> None:
    """Salva resultado como JSON. Silencioso em caso de erro."""
    try:
        path = _checkpoint_path(sc_dir, nq, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        # arch_mat pode ser tensor — converte para lista
        safe = _make_json_safe(result)
        path.write_text(json.dumps(safe, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[WARN] checkpoint save failed nq={nq} seed={seed}: {e}")


def load_checkpoint(sc_dir: Path, nq: int, seed: int) -> dict[str, Any] | None:
    """
    Retorna o resultado salvo se existir, None caso contrário.
    Loga skip para visibilidade.
    """
    path = _checkpoint_path(sc_dir, nq, seed)
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        print(f"[CHECKPOINT] skip nq={nq} seed={seed} — já computado ({path})")
        return result
    except Exception as e:
        print(f"[WARN] checkpoint load failed nq={nq} seed={seed}: {e} — recomputando")
        return None


def _make_json_safe(obj: Any) -> Any:
    """Converte recursivamente tipos não-serializáveis para tipos JSON."""
    import numpy as np
    import torch

    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj