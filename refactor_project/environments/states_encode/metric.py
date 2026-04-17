import torch


def bce_logits_loss_torch(
    logits: torch.Tensor, y: torch.Tensor, pos_weight: torch.Tensor | None = None
) -> float:
    """
    BCEWithLogitsLoss em float, robusto para shapes.
    Retorna float (python) para logging.
    """
    if logits.dim() > 1:
        logits = logits.view(-1)
    if y.dim() > 1:
        y = y.view(-1)
    y = y.to(dtype=logits.dtype)
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss = crit(logits, y)
    return float(loss.detach().cpu().item())
