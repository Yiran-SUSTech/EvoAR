import torch
import torch.nn.functional as F


def compute_samplewise_loss(logits, targets, valid=None):
    batch_size, seq_len, vocab_size = logits.shape
    loss_all = F.cross_entropy(
        logits.reshape(batch_size * seq_len, vocab_size),
        targets.reshape(batch_size * seq_len),
        reduction="none",
    ).reshape(batch_size, seq_len)

    if valid is None:
        sample_loss = loss_all.mean(dim=1)
        return sample_loss, loss_all

    valid = valid.to(loss_all.device).float().reshape(batch_size)
    token_weights = valid[:, None].expand_as(loss_all)
    denom = token_weights.sum(dim=1).clamp_min(1.0)
    sample_loss = (loss_all * token_weights).sum(dim=1) / denom
    return sample_loss, loss_all


def compute_latency_proxy(schedule_steps, mode="num_groups"):
    schedule_steps = schedule_steps.long()
    num_groups = schedule_steps.max(dim=1).values + 1
    if mode == "num_groups":
        return num_groups.float()
    if mode == "num_groups_plus_max_group":
        proxies = []
        for sample_steps in schedule_steps:
            counts = torch.bincount(sample_steps, minlength=int(sample_steps.max().item()) + 1)
            proxies.append(float(len(counts) + counts.max().item()))
        return torch.tensor(proxies, device=schedule_steps.device, dtype=torch.float32)
    raise ValueError(f"unsupported latency proxy mode: {mode}")


def compute_fitness(logits, targets, schedule_steps, valid=None, latency_mode="num_groups"):
    sample_loss, token_loss = compute_samplewise_loss(logits, targets, valid=valid)
    latency = compute_latency_proxy(schedule_steps, mode=latency_mode)
    return {
        "sample_loss": sample_loss.detach(),
        "latency_proxy": latency.detach(),
        "token_loss": token_loss.detach(),
    }
