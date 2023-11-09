import torch
import numpy as np

def difference_of_logits(logits, labels, labels_infhot):
    if labels_infhot is None:
        labels_infhot = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), float('inf'))

    class_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    other_logits = (logits - labels_infhot).amax(dim=1)
    return class_logits - other_logits


def dlr_loss(x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
    u = torch.arange(x.shape[0])

    return -(x[u, y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (
            1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)
