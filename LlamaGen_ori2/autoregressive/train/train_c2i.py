# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/train.py
#   nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import LambdaLR
from glob import glob
from copy import deepcopy
import os
import time
import inspect
import argparse
import math

import sys

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.join(current_script_dir, '../../') # adjust the path based on your project structure
project_root_dir = os.path.abspath(project_root_dir)

if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

from utils.logger import create_logger
from utils.distributed import init_distributed_mode
from utils.ema import update_ema, requires_grad
from dataset.build import build_dataset
from autoregressive.models.gpt import GPT_models


#################################################################################
#                             Training Helper Functions                         #
#################################################################################
def creat_optimizer(model, weight_decay, learning_rate, betas, logger):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    logger.info(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if fused_available else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    logger.info(f"using fused AdamW: {fused_available}")
    return optimizer

def print_model_summary(model, logger):
    """
    打印模型每一层的名称和参数量。
    """
    total_params = 0
    trainable_params = 0
    
    # 打印标题
    logger.info("=" * 60)
    logger.info("Model Parameter Summary:")
    logger.info("{:<50} {:>10} {:>10}".format("Layer Name", "Params (M)", "Trainable"))
    logger.info("-" * 60)

    # 使用 named_parameters() 遍历所有参数
    for name, parameter in model.named_parameters():
        # 获取当前参数的元素数量
        param_count = parameter.numel()
        total_params += param_count
        
        # 检查参数是否可训练
        is_trainable = parameter.requires_grad
        if is_trainable:
            trainable_params += param_count
        
        # 打印当前层的信息
        logger.info("{:<50} {:>10.2f} {:>10}".format(
            name,
            param_count / 1e6, # 转换为百万 (M)
            "Yes" if is_trainable else "No"
        ))

    # 打印总结
    logger.info("-" * 60)
    logger.info("Total Model Parameters: {:.2f} M".format(total_params / 1e6))
    logger.info("Total Trainable Parameters: {:.2f} M".format(trainable_params / 1e6))
    logger.info("=" * 60)

#################################################################################
#                                  Training Loop                                #
#################################################################################
def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    
    # Setup DDP:
    init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.gpt_model.replace("/", "-")  # e.g., GPT-XL/2 --> GPT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        time_record = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        cloud_results_dir = f"{args.cloud_save_path}/{time_record}"
        # cloud_checkpoint_dir = f"{cloud_results_dir}/{experiment_index:03d}-{model_string_name}/checkpoints"
        # os.makedirs(cloud_checkpoint_dir, exist_ok=True)
        # logger.info(f"Experiment directory created in cloud at {cloud_checkpoint_dir}")
    
    else:
        logger = create_logger(None)

    # training args
    logger.info(f"{args}")

    # training env
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")


    # Setup model
    if args.drop_path_rate > 0.0:
        dropout_p = 0.0
    else:
        dropout_p = args.dropout_p
    latent_size = args.image_size // args.downsample_size
    model = GPT_models[args.gpt_model](
        vocab_size=args.vocab_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        resid_dropout_p=dropout_p,
        ffn_dropout_p=dropout_p,
        drop_path_rate=args.drop_path_rate,
        token_dropout_p=args.token_dropout_p,
    ).to(device)
    logger.info(f"GPT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.ema:
        ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
        requires_grad(ema, False)
        logger.info(f"EMA Parameters: {sum(p.numel() for p in ema.parameters()):,}")

    # Setup optimizer
    optimizer = creat_optimizer(model, args.weight_decay, args.lr, (args.beta1, args.beta2), logger)

    # Setup data:
    dataset = build_dataset(args)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        sampler=sampler,
        drop_last=True,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
        prefetch_factor=args.prefetch_factor,
        pin_memory=False
    )
    flip_info = 'with' if dataset.flip else 'without'
    aug_info = 10 if 'ten_crop' in dataset.feature_dir else 1
    aug_info = 2 * aug_info if dataset.aug_feature_dir is not None else aug_info

    steps_per_epoch = len(loader) # the number of batches loaded into the rank, also the number of steps per epoch
    total_training_steps = steps_per_epoch * args.epochs
    # Set default values for learning rate parameters
    if args.max_lr is None:
        args.max_lr = args.lr
    if args.min_lr is None:
        args.min_lr = args.lr * 0.1
    if args.cosine_percent is None:
        args.cosine_percent = 1.0 - args.warmup_percent - args.const_percent

    # Validate that percentages sum to 1.0
    total_percent = args.warmup_percent + args.const_percent + args.cosine_percent
    if abs(total_percent - 1.0) > 1e-6:
        logger.warning(f"Warning: warmup_percent ({args.warmup_percent}) + const_percent ({args.const_percent}) + cosine_percent ({args.cosine_percent}) = {total_percent}, which does not equal 1.0")

    warmup_steps = int(total_training_steps * args.warmup_percent)
    constant_steps = int(total_training_steps * args.const_percent)
    cosine_steps = int(total_training_steps * args.cosine_percent)
    cosine_start_step = warmup_steps + constant_steps

    logger.info(f"Dataset contains {len(dataset):,} images ({args.code_path}) "
                f"{flip_info} flip augmentation and {aug_info} crop augmentation")
    logger.info(f"Learning rate schedule: max_lr={args.max_lr}, min_lr={args.min_lr}")
    logger.info(f"warmup steps: {warmup_steps} ({args.warmup_percent*100:.1f}%), constant steps: {constant_steps} ({args.const_percent*100:.1f}%), cosine steps: {cosine_steps} ({args.cosine_percent*100:.1f}%), total training steps: {total_training_steps}")
    
    def lr_lambda(current_step: int):
        """
        Warmup -> Constant -> Cosine Annealing 调度策略。
        返回学习率相对于 optimizer 初始 lr 的乘数。

        - Warmup 阶段: 从 0 线性增长到 max_lr
        - Constant 阶段: 保持 max_lr
        - Cosine Annealing 阶段: 从 max_lr 按余弦曲线下降到 min_lr
        """
        if current_step < warmup_steps:
            # Warmup: 从 0 线性增长到 max_lr
            # 返回值范围: [0, max_lr/lr]
            return (args.max_lr / args.lr) * (float(current_step) / float(max(1, warmup_steps)))

        elif current_step < cosine_start_step:
            # Constant: 保持 max_lr
            return args.max_lr / args.lr

        elif current_step < total_training_steps:
            # Cosine Annealing: 从 max_lr 下降到 min_lr
            if cosine_steps <= 0:
                return args.min_lr / args.lr

            progress = float(current_step - cosine_start_step) / float(cosine_steps)
            progress = min(1.0, progress)

            # Cosine annealing formula: 从 1.0 下降到 0.0
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))

            # 将 cosine_factor 映射到 [min_lr, max_lr] 范围
            current_lr = args.min_lr + (args.max_lr - args.min_lr) * cosine_factor

            return current_lr / args.lr
        else:
            # 超出训练步数，返回最小学习率
            return args.min_lr / args.lr

    # Prepare models for training:
    if args.gpt_ckpt:
        # checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
        # model.load_state_dict(checkpoint["model"])
        checkpoint = torch.load(args.gpt_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=False)
        if args.ema:
            ema.load_state_dict(checkpoint["ema"] if "ema" in checkpoint else checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        train_steps = checkpoint["steps"] if "steps" in checkpoint else int(args.gpt_ckpt.split('/')[-1].split('.')[0])
        start_epoch = int(train_steps / int(len(dataset) / args.global_batch_size))
        train_steps = int(start_epoch * int(len(dataset) / args.global_batch_size))
        del checkpoint
        logger.info(f"Resume training from checkpoint: {args.gpt_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    else:
        train_steps = 0
        start_epoch = 0
        if args.ema:
            update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights

    # setup lr scheduler for optimizer
    scheduler = None
    if args.is_lr_scheduler:
        # Create a learning rate scheduler
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        if args.gpt_ckpt is not None:
            scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=train_steps - 1)

    if not args.no_compile:
        logger.info("compiling the model... (may take several minutes)")
        model = torch.compile(model) # requires PyTorch 2.0        
    
    model = DDP(model.to(device), device_ids=[args.gpu])

    # --- 获取原始模型实例 ---
    if hasattr(model, 'module'):
        # model is DDP wrapped
        raw_model = model.module
    else:
        raw_model = model

    if hasattr(raw_model, '_orig_mod'):
        # model is torch.compile wrapped
        raw_model = raw_model._orig_mod

    # --- 打印模型参数总结 ---
    print_model_summary(raw_model, logger)

    
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    if args.ema:
        ema.eval()  # EMA model should always be in eval mode

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16')) ####################################
    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time.time()
    # monitoring gradients
    max_grad_norm = -1000000000.0
    max_grad_norm_step = -1

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            z_indices = x.reshape(x.shape[0], -1)
            c_indices = y.reshape(-1)
            assert z_indices.shape[0] == c_indices.shape[0]

            # logger.info(f'z_indices.shape: {z_indices.shape}')
            # logger.info(f'c_indices.shape: {c_indices.shape}')

            # -------------------------- [诊断点 1: 检查输入数据] --------------------------
            # 检查输入 z_indices 和 c_indices 中是否有 NaN 或 Inf
            if torch.any(torch.isnan(z_indices)) or torch.any(torch.isinf(z_indices)) or \
               torch.any(torch.isnan(c_indices)) or torch.any(torch.isinf(c_indices)):
                logger.info(f"FATAL ERROR: Input data contains NaN or Inf at step {train_steps}.")
                # 打印出有问题的 step，然后退出
                raise RuntimeError("Input data error (NaN/Inf) detected.")
            # --------------------------------------------------------------------------

            with torch.cuda.amp.autocast(dtype=ptdtype):  ####################################################
                logits, loss = model(cond_idx=c_indices, idx=z_indices[:,:-1], targets=z_indices)
            # backward pass, with gradient scaling if training in fp16  
            
            # -------------------------- [诊断点 2: 检查 Loss] --------------------------
            # 检查 loss 是否为 NaN 或 Inf
            if not torch.isfinite(loss):
                logger.info(f"FATAL ERROR: Loss is not finite (NaN/Inf) at step {train_steps}.")
                # 打印 logits 的数值范围来诊断交叉熵问题
                logger.info(f"Logits min: {logits.min().item():.2f}, max: {logits.max().item():.2f}, mean: {logits.mean().item():.2f}")
                logger.info(f"Logits contain NaN: {torch.any(torch.isnan(logits))}, Inf: {torch.any(torch.isinf(logits))}")
                
                # 打印 GradScaler 的状态 (如果使用了 fp16)
                if args.mixed_precision == 'fp16':
                    logger.info(f"GradScaler Scale: {scaler.get_scale()}")

                raise RuntimeError("Non-finite loss detected.")
            # --------------------------------------------------------------------------


            scaler.scale(loss).backward()

            if args.max_grad_norm != 0.0:
                # -------------------------- [诊断点 3: 检查梯度范数] --------------------------
                # 必须先 unscale 才能计算真实的梯度范数
                scaler.unscale_(optimizer)

                # 计算所有参数的梯度范数
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm) 
                
                if not torch.isfinite(grad_norm):
                    logger.info(f"FATAL ERROR: Gradient norm is not finite (NaN/Inf) at step {train_steps}.")
                    
                    # 打印单个参数的梯度状态以进一步诊断
                    for name, param in model.named_parameters():
                        if param.grad is not None and not torch.isfinite(param.grad).all():
                            logger.info(f"Non-finite gradient found in parameter: {name}")
                    
                    raise RuntimeError("Non-finite gradient detected.")
                    
                if grad_norm.item() > max_grad_norm:
                    max_grad_norm = grad_norm.item()
                    max_grad_norm_step = train_steps
            # --------------------------------------------------------------------------

            # if args.max_grad_norm != 0.0:
            #     scaler.unscale_(optimizer)
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)
            if args.ema:
                update_ema(ema, model.module._orig_mod if not args.no_compile else model.module)

            current_lr = optimizer.param_groups[0]['lr'] # get current learning rate that is used to update parameters

            if args.is_lr_scheduler:
                scheduler.step() # update learning rate

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                sec_per_step = (end_time - start_time) / log_steps
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Sec/Step: {sec_per_step:.2f}, Current learning rate: {current_lr:.6f}, " + 
                            f"Max Grad Norm so far: {max_grad_norm:.2f} at step {max_grad_norm_step} in this {args.log_every}-step interval.")
                max_grad_norm = -1000000000.0
                max_grad_norm_step = -1
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time.time()

            # Save checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    if not args.no_compile:
                        model_weight = model.module._orig_mod.state_dict()
                    else:
                        model_weight = model.module.state_dict()  
                    checkpoint = {
                        "model": model_weight,
                        "optimizer": optimizer.state_dict(),
                        "steps": train_steps,
                        "args": args
                    }
                    if args.ema:
                        checkpoint["ema"] = ema.state_dict()
                    if not args.no_local_save:
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                    
                    # cloud_checkpoint_path = f"{cloud_checkpoint_dir}/{train_steps:07d}.pt"
                    # torch.save(checkpoint, cloud_checkpoint_path)
                    # logger.info(f"Saved checkpoint in cloud to {cloud_checkpoint_path}")
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-path", type=str, required=True)
    parser.add_argument("--cloud-save-path", type=str, required=True, help='please specify a cloud disk path, if not, local path')
    parser.add_argument("--no-local-save", action='store_true', help='no save checkpoints to local path for limited disk volume')
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i", help="class-conditional or text-conditional")
    parser.add_argument("--vocab-size", type=int, default=16384, help="vocabulary size of visual tokenizer")
    parser.add_argument("--ema", action='store_true', help="whether using ema training")
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--dropout-p", type=float, default=0.1, help="dropout_p of resid_dropout_p and ffn_dropout_p")
    parser.add_argument("--token-dropout-p", type=float, default=0.1, help="dropout_p of token_dropout_p")
    parser.add_argument("--drop-path-rate", type=float, default=0.0, help="using stochastic depth decay")
    parser.add_argument("--no-compile", action='store_true')
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--dataset", type=str, default='imagenet_code')
    parser.add_argument("--image-size", type=int, choices=[256, 384, 448, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use")
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1 parameter for the Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.95, help="beta2 parameter for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--warmup_percent", type=float, default=0.01, help="the ratio of warm-up steps in total number of steps")
    parser.add_argument("--const_percent", type=float, default=0.7, help="the ratio of steps with constant lr in total number of steps")
    parser.add_argument("--cosine_percent", type=float, default=None, help="the ratio of cosine annealing steps in total number of steps, if None, it will be calculated as 1.0 - warmup_percent - const_percent")
    parser.add_argument("--max-lr", type=float, default=None, help="maximum learning rate to reach during warmup, if None, it will be set to lr")
    parser.add_argument("--min-lr", type=float, default=None, help="minimum learning rate at the end of cosine annealing, if None, it will be set to 0.1 * lr")
    parser.add_argument("--is-lr-scheduler", action='store_true')

    args = parser.parse_args()
    main(args)
