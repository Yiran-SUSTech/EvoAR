import torch


def make_fixed_schedule(code_len, num_groups, device=None):
    if code_len <= 0:
        raise ValueError("code_len must be positive")
    if num_groups <= 0:
        raise ValueError("num_groups must be positive")
    positions = torch.arange(code_len, device=device)
    schedule = torch.div(positions * num_groups, code_len, rounding_mode="floor")
    return torch.clamp(schedule, max=num_groups - 1).long()


def canonicalize_schedule(schedule_steps):
    if schedule_steps.ndim != 1:
        raise ValueError("schedule_steps must be a 1D tensor")
    unique_steps = torch.unique(schedule_steps, sorted=True)
    canonical = torch.empty_like(schedule_steps)
    for new_step, old_step in enumerate(unique_steps.tolist()):
        canonical[schedule_steps == old_step] = new_step
    return canonical.long()


def build_grouped_attention_mask(prefix_valid_mask, schedule_steps):
    prefix_valid_mask = prefix_valid_mask.bool()
    schedule_steps = schedule_steps.long()
    if prefix_valid_mask.ndim != 2:
        raise ValueError("prefix_valid_mask must have shape [batch, prefix_len]")
    if schedule_steps.ndim != 2:
        raise ValueError("schedule_steps must have shape [batch, code_len]")
    if prefix_valid_mask.shape[0] != schedule_steps.shape[0]:
        raise ValueError("prefix_valid_mask and schedule_steps must share batch size")

    batch_size, prefix_len = prefix_valid_mask.shape
    _, code_len = schedule_steps.shape
    total_len = prefix_len + code_len
    device = schedule_steps.device

    full_mask = torch.zeros((batch_size, total_len, total_len), dtype=torch.bool, device=device)

    if prefix_len > 0:
        prefix_causal = torch.tril(torch.ones((prefix_len, prefix_len), dtype=torch.bool, device=device))
        prefix_prefix = prefix_causal.unsqueeze(0) & prefix_valid_mask[:, None, :]
        full_mask[:, :prefix_len, :prefix_len] = prefix_prefix
        full_mask[:, prefix_len:, :prefix_len] = prefix_valid_mask[:, None, :].expand(batch_size, code_len, prefix_len)

    latent_query_steps = schedule_steps[:, :, None]
    latent_key_steps = schedule_steps[:, None, :]
    latent_mask = latent_key_steps < latent_query_steps
    latent_mask = latent_mask | torch.eye(code_len, dtype=torch.bool, device=device).unsqueeze(0)
    full_mask[:, prefix_len:, prefix_len:] = latent_mask

    full_mask = full_mask | torch.eye(total_len, dtype=torch.bool, device=device).unsqueeze(0)
    return full_mask.unsqueeze(1)


def build_training_mask(prefix_valid_mask, schedule_steps):
    return build_grouped_attention_mask(prefix_valid_mask, schedule_steps)


def debug_mask_from_schedule(schedule_steps, prefix_valid_mask=None):
    schedule_steps = torch.as_tensor(schedule_steps, dtype=torch.long)
    if schedule_steps.ndim == 1:
        schedule_steps = schedule_steps.unsqueeze(0)
    if prefix_valid_mask is None:
        prefix_valid_mask = torch.ones((schedule_steps.shape[0], 0), dtype=torch.bool)
    else:
        prefix_valid_mask = torch.as_tensor(prefix_valid_mask, dtype=torch.bool)
        if prefix_valid_mask.ndim == 1:
            prefix_valid_mask = prefix_valid_mask.unsqueeze(0)
    return build_grouped_attention_mask(prefix_valid_mask, schedule_steps)
