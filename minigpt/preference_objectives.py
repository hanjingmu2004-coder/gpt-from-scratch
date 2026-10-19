"""Single-step categorical control objectives, not long-sequence RLHF trainers."""
import torch
from torch.nn import functional as F


def exact_kl(logp, reference):
    return (logp.exp() * (logp - reference)).sum(-1)


def dpo(logp, reference, labels, beta=.1):
    chosen = logp.gather(-1, labels[:, None]).squeeze(-1)
    rejected = logp.gather(-1, (1 - labels)[:, None]).squeeze(-1)
    ref_chosen = reference.gather(-1, labels[:, None]).squeeze(-1)
    ref_rejected = reference.gather(-1, (1 - labels)[:, None]).squeeze(-1)
    return -F.logsigmoid(beta * ((chosen - rejected) - (ref_chosen - ref_rejected))).mean()


def group_advantages(rewards):
    # Groups of identical rewards have zero advantage, not NaN.
    return (rewards - rewards.mean(-1, keepdim=True)) / rewards.std(-1, keepdim=True, correction=0).clamp_min(1e-6)


def clipped_policy_loss(logp, old_logp, actions, advantages, clip=.2):
    selected = logp.gather(-1, actions)
    ratio = (selected - old_logp).exp()
    return -torch.minimum(ratio * advantages, ratio.clamp(1 - clip, 1 + clip) * advantages).mean()
