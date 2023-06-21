import datetime
import logging
import sys
import time

import torch
from torch_geometric.graphgym import compute_loss
from tqdm import tqdm

sys.path.append('submodules/ChIRo')
sys.path.append('submodules/tetra_dmpnn')
from main import custom_load_cfg, custom_set_out_dir, run_loop_settings, custom_set_run_dir

from graphgps.dataset.utils import create_custom_loader

import graphgps  # noqa, register custom modules

from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             set_cfg)
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric import seed_everything

from graphgps.finetuning import load_pretrained_model_cfg, \
    init_model_from_pretrained
from graphgps.logger import create_logger

if __name__ == '__main__':
    # Load cmd line args
    args = parse_args()
    # Load config file
    set_cfg(cfg)
    custom_load_cfg(cfg, args)
    custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag)
    dump_cfg(cfg)
    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    # Repeat for multiple experiment runs
    for run_id, seed, split_index in zip(*run_loop_settings(args)):
        # Set configurations for each run
        custom_set_run_dir(cfg, run_id)
        set_printing()
        cfg.dataset.split_index = split_index
        cfg.seed = seed
        cfg.run_id = run_id
        cfg.cfg_file = args.cfg_file
        seed_everything(cfg.seed)
        auto_select_device()
        if cfg.pretrained.dir:
            cfg = load_pretrained_model_cfg(cfg)
        logging.info(f"[*] Run ID {run_id}: seed={cfg.seed}, "
                     f"split_index={cfg.dataset.split_index}")
        logging.info(f"    Starting now: {datetime.datetime.now()}")
        # Set machine learning pipeline
        loaders, dataframes = create_custom_loader()
        loggers = create_logger(dataframes)
        model = create_model()
        if cfg.pretrained.dir:
            model = init_model_from_pretrained(
                model, cfg.pretrained.dir, cfg.pretrained.freeze_main,
                cfg.pretrained.reset_prediction_head
            )

        train_loader = loaders[0]
        model.train()
        forward = []
        backward = []
        for batch in tqdm(train_loader):
            batch.to(torch.device(cfg.device))
            start = time.time()
            pred, true = model(batch)
            forward.append(time.time() - start)
            loss, pred_score = compute_loss(pred, true)
            start = time.time()
            loss.backward()
            backward.append(time.time() - start)

        print('total', sum(forward) + sum(backward))
        print('total forward', sum(forward))
        print('total backward', sum(backward))
