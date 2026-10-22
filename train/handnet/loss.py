"""CenterNet loss: penalty-reduced focal loss on the heatmap, L1 on size and offset."""

import torch
import torch.nn.functional as F


def hand_loss(
    out: torch.Tensor, heat: torch.Tensor, reg: torch.Tensor, mask: torch.Tensor
) -> dict[str, torch.Tensor]:
    """out (B, 5, Hs, Ws); heat (B, Hs, Ws); reg (B, 4, Hs, Ws); mask (B, Hs, Ws) weights."""
    p = torch.sigmoid(out[:, 0]).clamp(1e-4, 1 - 1e-4)
    pos = heat.eq(1).float()
    n = pos.sum().clamp(min=1)
    # Positives: push p to 1. Negatives near a centre are penalised less ((1 - heat)^4).
    pos_loss = -(torch.log(p) * (1 - p) ** 2 * pos).sum()
    neg_loss = -(torch.log(1 - p) * p**2 * (1 - heat) ** 4 * (1 - pos)).sum()
    focal = (pos_loss + neg_loss) / n

    # Size and offset at the centre cells only (see targets.py), averaged over
    # hands and the 2 channels of each.
    m = mask.unsqueeze(1)
    total = (m.sum() * 2).clamp(min=1e-6)
    size = (F.l1_loss(out[:, 1:3], reg[:, 0:2], reduction="none") * m).sum() / total
    offset = (F.l1_loss(out[:, 3:5], reg[:, 2:4], reduction="none") * m).sum() / total
    return {"total": focal + size + offset, "heat": focal, "size": size, "offset": offset}
