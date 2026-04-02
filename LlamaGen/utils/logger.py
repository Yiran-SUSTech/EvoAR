import logging
import torch.distributed as dist


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    is_rank0 = (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
    if is_rank0:  # real logger
        handlers = [logging.StreamHandler()]
        if logging_dir is not None:
            handlers.append(logging.FileHandler(f"{logging_dir}/log.txt"))
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=handlers,
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger
