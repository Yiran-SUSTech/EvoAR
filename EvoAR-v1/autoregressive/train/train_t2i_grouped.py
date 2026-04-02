import argparse
import os
import sys
import time
from glob import glob
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

ROOT_DIR = Path(__file__).resolve().parents[2]
PARENT_DIR = ROOT_DIR.parent
LLAMAGEN_DIR = PARENT_DIR / "LlamaGen"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))
if str(LLAMAGEN_DIR) not in sys.path:
    sys.path.insert(0, str(LLAMAGEN_DIR))

from autoregressive.train.fitness import compute_fitness
from autoregressive.train.mask_builder import build_training_mask
from autoregressive.train.pareto_plot import save_pareto_front_plots
from autoregressive.train.schedule_manager import ScheduleManager, broadcast_schedule_manager_state, gather_records_to_rank0
from dataset.build import build_dataset
from LlamaGen.autoregressive.models.gpt import GPT_models
from LlamaGen.autoregressive.train.train_c2i import creat_optimizer
from LlamaGen.tokenizer.tokenizer_image.vq_model import VQ_models
from LlamaGen.dataset.augmentation import center_crop_arr
from utils.distributed import init_distributed_mode
from utils.logger import create_logger


def create_experiment_logger(args, rank):
    if rank != 0:
        return create_logger(None), None, None, None, None
    os.makedirs(args.results_dir, exist_ok=True)
    experiment_index = len(glob(f"{args.results_dir}/*"))
    model_string_name = args.gpt_model.replace("/", "-")
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")
    pareto_dir = f"{experiment_dir}/pareto_fronts"
    os.makedirs(pareto_dir, exist_ok=True)

    time_record = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    cloud_results_dir = f"{args.cloud_save_path}/{time_record}"
    cloud_checkpoint_dir = f"{cloud_results_dir}/{experiment_index:03d}-{model_string_name}/checkpoints"
    os.makedirs(cloud_checkpoint_dir, exist_ok=True)
    cloud_pareto_dir = f"{cloud_results_dir}/{experiment_index:03d}-{model_string_name}/pareto_fronts"
    os.makedirs(cloud_pareto_dir, exist_ok=True)
    logger.info(f"Experiment directory created in cloud at {cloud_checkpoint_dir}")
    return logger, checkpoint_dir, cloud_checkpoint_dir, pareto_dir, cloud_pareto_dir


def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    init_distributed_mode(args)
    distributed = bool(getattr(args, "distributed", False))
    world_size = int(getattr(args, "world_size", 1))
    rank = int(getattr(args, "rank", 0))
    device = int(getattr(args, "gpu", 0)) if distributed else 0
    assert args.global_batch_size % world_size == 0
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    logger, checkpoint_dir, cloud_checkpoint_dir, pareto_dir, cloud_pareto_dir = create_experiment_logger(args, rank)
    logger.info(f"{args}")
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={world_size}, distributed={distributed}")

    latent_size = args.image_size // args.downsample_size
    code_len = latent_size ** 2
    model = GPT_models[args.gpt_model](
        vocab_size=args.vocab_size,
        block_size=code_len,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        resid_dropout_p=args.dropout_p,
        ffn_dropout_p=args.dropout_p,
        token_dropout_p=args.token_dropout_p,
    ).to(device)
    logger.info(f"GPT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = creat_optimizer(model, args.weight_decay, args.lr, (args.beta1, args.beta2), logger)
    schedule_manager = ScheduleManager(
        code_len=code_len,
        evolve_every=args.evolve_every,
        population_size=args.schedule_population,
        mutation_prob=args.schedule_mutation_prob,
        max_groups=args.max_schedule_groups,
        device=torch.device(f"cuda:{device}"),
        crossover_prob=args.schedule_crossover_prob,
        shift_radius=args.schedule_shift_radius,
        block_max_size=args.schedule_block_max_size,
        split_prob=args.schedule_split_prob,
        trend_weight=args.schedule_trend_weight,
        final_loss_weight=args.schedule_final_loss_weight,
    )

    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
    )
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    dataset = build_dataset(args, transform=transform)
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.global_seed,
        )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // world_size),
        shuffle=not distributed,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    logger.info(f"Dataset contains {len(dataset):,} images")

    if args.gpt_ckpt:
        checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "schedule_manager" in checkpoint:
            schedule_manager.load_state_dict(checkpoint["schedule_manager"])
        train_steps = checkpoint["steps"] if "steps" in checkpoint else int(Path(args.gpt_ckpt).stem)
        steps_per_epoch = max(int(len(dataset) / max(int(args.global_batch_size / world_size), 1)), 1)
        start_epoch = int(train_steps / steps_per_epoch)
        train_steps = int(start_epoch * steps_per_epoch)
        del checkpoint
        logger.info(f"Resume training from checkpoint: {args.gpt_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    else:
        train_steps = 0
        start_epoch = 0

    if not args.no_compile:
        logger.info("compiling the model...")
        model = torch.compile(model)

    model = model.to(device)
    if distributed:
        model = DDP(model, device_ids=[args.gpu])
    model.train()

    ptdtype = {"none": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.mixed_precision]
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == "fp16"))
    log_steps = 0
    running_loss = 0.0
    running_sample_loss = 0.0
    running_latency = 0.0
    running_train_time = 0.0
    running_evolve_time = 0.0
    start_time = time.time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y, prefix_valid_mask, valid in loader:
            iter_start_time = time.time()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            prefix_valid_mask = prefix_valid_mask.to(device, non_blocking=True).bool()
            valid = valid.to(device, non_blocking=True)
            with torch.no_grad():
                _, _, [_, _, indices] = vq_model.encode(x)
            z_indices = indices.reshape(x.shape[0], -1).long()
            c_indices = y.reshape(y.shape[0], y.shape[-2], y.shape[-1]).to(device)
            schedule_steps = schedule_manager.sample(z_indices.shape[0], train_steps, device=device)
            attn_mask = build_training_mask(prefix_valid_mask, schedule_steps)

            with torch.cuda.amp.autocast(dtype=ptdtype):
                logits, loss = model(
                    cond_idx=c_indices,
                    idx=z_indices[:, :-1],
                    targets=z_indices,
                    mask=attn_mask[:, :, :-1, :-1],
                    valid=valid,
                )

            fitness = compute_fitness(
                logits=logits,
                targets=z_indices,
                schedule_steps=schedule_steps,
                valid=valid,
                latency_mode=args.latency_proxy_mode,
            )
            schedule_manager.record(schedule_steps, fitness["sample_loss"], fitness["latency_proxy"])

            scaler.scale(loss).backward()
            if args.max_grad_norm != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            running_train_time += time.time() - iter_start_time

            running_loss += loss.item()
            running_sample_loss += fitness["sample_loss"].mean().item()
            running_latency += fitness["latency_proxy"].mean().item()
            log_steps += 1
            train_steps += 1

            evolved = False
            if schedule_manager.should_evolve(train_steps):
                evolve_start_time = time.time()
                gathered_records = gather_records_to_rank0(schedule_manager.flush_pending_records(), dst=0)
                if rank == 0:
                    schedule_manager.ingest_records(gathered_records or [])
                    evolved = schedule_manager.evolve_if_needed(train_steps)
                    if evolved:
                        save_pareto_front_plots(schedule_manager.archive, pareto_dir, train_steps)
                        save_pareto_front_plots(schedule_manager.archive, cloud_pareto_dir, train_steps)
                broadcast_schedule_manager_state(schedule_manager, src=0)
                running_evolve_time += time.time() - evolve_start_time

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_sample_loss = torch.tensor(running_sample_loss / log_steps, device=device)
                avg_latency = torch.tensor(running_latency / log_steps, device=device)
                if distributed:
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    dist.all_reduce(avg_sample_loss, op=dist.ReduceOp.SUM)
                    dist.all_reduce(avg_latency, op=dist.ReduceOp.SUM)
                    avg_loss = avg_loss.item() / world_size
                    avg_sample_loss = avg_sample_loss.item() / world_size
                    avg_latency = avg_latency.item() / world_size
                else:
                    avg_loss = avg_loss.item()
                    avg_sample_loss = avg_sample_loss.item()
                    avg_latency = avg_latency.item()
                archive_summary = schedule_manager.archive_summary()
                avg_train_time = running_train_time / log_steps
                avg_evolve_time = running_evolve_time / log_steps
                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Sample Loss: {avg_sample_loss:.4f}, Latency Proxy: {avg_latency:.4f}, Archive: {archive_summary['size']}, Avg Groups: {archive_summary['avg_groups']}, Evolved: {evolved}, Train Steps/Sec: {steps_per_sec:.2f}, Avg Train Time: {avg_train_time:.4f}s, Avg Evolve Time: {avg_evolve_time:.4f}s"
                )
                running_loss = 0.0
                running_sample_loss = 0.0
                running_latency = 0.0
                running_train_time = 0.0
                running_evolve_time = 0.0
                log_steps = 0
                start_time = time.time()

            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    model_source = model.module if distributed else model
                    model_weight = model_source._orig_mod.state_dict() if not args.no_compile else model_source.state_dict()
                    checkpoint = {
                        "model": model_weight,
                        "optimizer": optimizer.state_dict(),
                        "steps": train_steps,
                        "args": args,
                        "schedule_manager": schedule_manager.state_dict(),
                        "pareto_archive": schedule_manager.state_dict().get("archive", []),
                    }
                    if not args.no_local_save:
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                    cloud_checkpoint_path = f"{cloud_checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, cloud_checkpoint_path)
                    logger.info(f"Saved checkpoint in cloud to {cloud_checkpoint_path}")
                if distributed:
                    dist.barrier()

    model.eval()
    logger.info("Done!")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--t5-feat-path", type=str, required=True)
    parser.add_argument("--short-t5-feat-path", type=str, default=None)
    parser.add_argument("--cloud-save-path", type=str, required=True)
    parser.add_argument("--no-local-save", action="store_true")
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None)
    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--codebook-embed-dim", type=int, default=8)
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument("--gpt-type", type=str, choices=["c2i", "t2i"], default="t2i")
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--cls-token-num", type=int, default=120)
    parser.add_argument("--dropout-p", type=float, default=0.1)
    parser.add_argument("--token-dropout-p", type=float, default=0.1)
    parser.add_argument("--drop-path", type=float, default=0.0)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--dataset", type=str, default="t2i")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=384)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--max-grad-norm", default=1.0, type=float)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default="bf16", choices=["none", "fp16", "bf16"])
    parser.add_argument("--dist-backend", type=str, default="nccl")
    parser.add_argument("--max-schedule-groups", type=int, default=256)
    parser.add_argument("--schedule-population", type=int, default=32)
    parser.add_argument("--schedule-mutation-prob", type=float, default=0.8)
    parser.add_argument("--schedule-crossover-prob", type=float, default=0.5)
    parser.add_argument("--schedule-shift-radius", type=int, default=4)
    parser.add_argument("--schedule-block-max-size", type=int, default=8)
    parser.add_argument("--schedule-split-prob", type=float, default=0.15)
    parser.add_argument("--schedule-trend-weight", type=float, default=0.25)
    parser.add_argument("--schedule-final-loss-weight", type=float, default=0.75)
    parser.add_argument("--evolve-every", type=int, default=0)
    parser.add_argument("--latency-proxy-mode", type=str, default="stepwise_surrogate", choices=["num_groups", "num_groups_plus_max_group", "stepwise_surrogate"])
    args = parser.parse_args()
    main(args)
