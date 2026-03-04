import os, sys, shutil
import os.path as osp
import time
import json
import random

import numpy as np
import torch
from torch import nn


def fix_random_seeds(seed=0):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_configs(args):
    if args.is_train:
        with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=4)
            f.close()

def create_folder(args):
    if not args.is_train:
        return

    output_dir = osp.abspath(args.output_dir)
    cwd = osp.abspath(os.getcwd())
    root_dir = osp.abspath(os.sep)

    # Never remove the current working directory or filesystem root.
    if output_dir in (cwd, root_dir):
        os.makedirs(output_dir, exist_ok=True)
        return

    # When resuming, keep existing artifacts in output_dir.
    if getattr(args, 'resume_ckpt', '') and osp.exists(output_dir):
        return

    if osp.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
