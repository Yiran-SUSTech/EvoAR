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


def compute_latency_proxy(schedule_steps, mode="stepwise_surrogate"):
    schedule_steps = schedule_steps.long()
    proxies = []
    for sample_steps in schedule_steps:
        counts = torch.bincount(sample_steps, minlength=int(sample_steps.max().item()) + 1).float()
        num_groups = float(len(counts))
        max_group = float(counts.max().item())
        mean_group = float(counts.mean().item())
        variance_group = float(((counts - mean_group) ** 2).mean().item())
        if mode == "num_groups":
            proxy = num_groups
        elif mode == "num_groups_plus_max_group":
            proxy = num_groups + max_group
        elif mode == "stepwise_surrogate":
            proxy = num_groups + 0.25 * max_group + 0.05 * variance_group
        else:
            raise ValueError(f"unsupported latency proxy mode: {mode}")
        proxies.append(proxy)
    return torch.tensor(proxies, device=schedule_steps.device, dtype=torch.float32)


def compute_grouped_step_loss(logits, targets, positions, valid=None):
    positions = torch.as_tensor(positions, dtype=torch.long, device=logits.device)
    step_logits = logits[:, positions, :]
    step_targets = targets[:, positions]
    batch_size, step_len, vocab_size = step_logits.shape
    loss_all = F.cross_entropy(
        step_logits.reshape(batch_size * step_len, vocab_size),
        step_targets.reshape(batch_size * step_len),
        reduction="none",
    ).reshape(batch_size, step_len)

    if valid is None:
        sample_loss = loss_all.mean(dim=1)
        return sample_loss.mean(), sample_loss.detach(), loss_all.detach()

    valid = valid.to(loss_all.device).float().reshape(batch_size, 1)
    weighted_loss = loss_all * valid
    sample_denom = (valid * step_len).clamp_min(1.0)
    sample_loss = weighted_loss.sum(dim=1, keepdim=True) / sample_denom
    total_denom = sample_denom.sum().clamp_min(1.0)
    return weighted_loss.sum() / total_denom, sample_loss.squeeze(1).detach(), loss_all.detach()


def compute_fitness(logits, targets, schedule_steps, valid=None, latency_mode="stepwise_surrogate"):
    sample_loss, token_loss = compute_samplewise_loss(logits, targets, valid=valid)
    latency = compute_latency_proxy(schedule_steps, mode=latency_mode)
    return {
        "sample_loss": sample_loss.detach(),
        "latency_proxy": latency.detach(),
        "token_loss": token_loss.detach(),
    }
