"""FlashSplat closed-form label assignment (FlashSplat colormask.py, multi_instance_opt)."""
import torch
import torch.nn.functional as F


def flashsplat_labels(votes, gamma=0.0, label=1):
    """(P, L) votes -> (P,) int32, 1 where Gaussian belongs to `label`, else 0.

    Per Gaussian, [votes of all other labels, votes of `label`] is L2-normalised, gamma is added to the
    first entry (background bias, gamma in [-1, 1]) and the argmax taken. Unvoted Gaussians are 0."""
    obj = votes[:, label]
    contrib = F.normalize(torch.stack([votes.sum(dim=1) - obj, obj], dim=1), dim=1)
    contrib[:, 0] += gamma
    return ((contrib.argmax(dim=1) == 1) & (votes.sum(dim=1) > 0)).int()
