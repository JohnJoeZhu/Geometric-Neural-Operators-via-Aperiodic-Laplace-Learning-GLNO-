import os
import torch
import torch.distributed as dist
import numpy as np
import logging
from datetime import datetime
import yaml

'''
Including:
system setting
- set_seed
- format_time

paths
- ensure_dir_exists
- get_dir_path

config and logger
- load_config
- save_config
- set_up_logger

pytorch thing
- setup_distributed
'''

# == system setting
def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def format_time(seconds):
    if seconds < 60:
        return "{:.1f}s".format(seconds)
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return "{}m {:.1f}s".format(int(minutes), seconds)
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return "{}h {}m {:.1f}s".format(int(hours), int(minutes), seconds)

# == paths
def ensure_dir_exists(d):
    if not os.path.exists(d):
        os.makedirs(d)

def get_dir_path(model, dataset, path):
    date = datetime.now().strftime("%m_%d")
    time = datetime.now().strftime("_%H_%M_%S")
    dir_path = os.path.join(path, dataset, date, model + time)
    dir_name = date + "_" + model + time
    return dir_path, dir_name

# == Config and logger
def load_config(args):
    with open(args['config'], 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    for key in config.keys():
        args[key] = config[key]
    return args

def save_config(args, saving_path):
    with open(os.path.join(saving_path, 'config.yaml'), 'w') as f:
        yaml.dump(args, f)

def set_up_logger(model, dataset, log_dir):
    log_dir, dir_name = get_dir_path(model, dataset, log_dir)
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=os.path.join(log_dir, "train.log"),
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    print("Saving logs in: {}".format(log_dir))

    return log_dir, dir_name

# == Pytorch things
def setup_distributed(args): ## TODO check it
    # 检查是否在分布式环境中运行
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.distributed = True
    else:
        print('Not using distributed mode')
        args.distributed = False
        args.world_size = 1
        args.rank = 0
        args.local_rank = 0
        return

    # 设置分布式训练
    if args.distributed:
        # 设置设备
        torch.cuda.set_device(args.local_rank)
        
        # 初始化进程组
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=args.world_size,
            rank=args.rank
        )
        
        print(f"Initialized distributed training with {args.world_size} GPUs, rank {args.rank}, local rank {args.local_rank}")