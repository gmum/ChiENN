import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import yaml

sys.path.append('submodules/ChIRo')
sys.path.append('submodules/GeometricTransformerMolecule')

import graphgps  # noqa, register custom modules
from graphgps.utils import update_cfg
from main import run_main, custom_load_cfg
from torch_geometric.graphgym.config import (cfg, set_cfg, dump_cfg)


def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')

    parser.add_argument('--cfg', dest='cfg_file', type=str, required=True,
                        help='The configuration file path.')
    parser.add_argument('--repeat', type=int, default=1,
                        help='The number of repeated jobs.')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='See graphgym/config.py for remaining options.')
    parser.add_argument('--reset', default=False, action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    # Load cmd line args
    args = parse_args()
    opts = args.opts
    repeat = args.repeat
    reset = args.reset
    grid_config_path = Path(args.cfg_file)

    with open(grid_config_path, 'r') as f:
        grid_config = json.load(f)

    cfg_file = grid_config['base_config_path']
    max_subset_size = grid_config['max_subset_size']
    set_cfg(cfg)
    custom_load_cfg(cfg=cfg, cfg_file=cfg_file, opts=opts)
    experiment_name = grid_config_path.name.split('.')[0]
    grid_dir = Path(cfg.out_dir) / experiment_name
    grid_dir.mkdir(exist_ok=True)
    cfg.out_dir = str(grid_dir)

    params_grid = grid_config['params_grid']
    params_grid = params_grid if isinstance(params_grid, list) else [params_grid]
    all_params = []
    for grid in params_grid:
        keys, values = zip(*grid.items())
        permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
        all_params.extend(permutations_dicts)

    for params in all_params:
        if set(params.keys()) != set(all_params[0].keys()):
            raise ValueError('Every grid combination must contain the same set of parameters names!')

    metric = cfg.metric_best
    metric_direction = cfg.metric_agg
    assert metric_direction in ['argmax', 'argmin']

    best_params = {}
    best_metric_value = -float('inf') if metric_direction == 'argmax' else float('inf')
    start_param = 0

    if reset:
        for i, params in enumerate(all_params):
            results_path = grid_dir / f'params_{i}' / 'agg/val/best.json'
            params_path = grid_dir / f'params_{i}' / 'params.json'
            if results_path.exists():
                with open(results_path, 'r') as f:
                    results = json.load(f)
                with open(params_path, 'r') as f:
                    params = json.load(f)
                metric_value = results[metric]
                if (metric_direction == 'argmax' and metric_value > best_metric_value) or (
                        metric_direction == 'argmin' and metric_value < best_metric_value):
                    best_metric_value = metric_value
                    best_params = params
                start_param = i + 1

    for i, params in enumerate(all_params):
        if start_param > i:
            continue
        for k, v in params.items():
            if v == '@BestParam()':
                params[k] = best_params[k]

        out_dir = grid_dir / f'params_{i}'
        out_dir.mkdir(exist_ok=True)
        cfg.out_dir = str(out_dir)
        with open(out_dir / 'params.json', 'w') as f:
            json.dump(params, f)
        wandb_base_name = f'{experiment_name}-params_{i}'
        update_cfg(cfg, params)
        dump_cfg(cfg)
        run_main(cfg=cfg, wandb_base_name=wandb_base_name, repeat=1, subset=max_subset_size)

        with open(out_dir / 'agg/val/best.json', 'r') as f:
            results = json.load(f)

        metric_value = results[metric]
        if (metric_direction == 'argmax' and metric_value > best_metric_value) or (
                metric_direction == 'argmin' and metric_value < best_metric_value):
            best_metric_value = metric_value
            best_params = params

        for path, dir, files in os.walk(out_dir):
            for file in files:
                if file.endswith('.ckpt'):
                    os.remove(Path(path) / file)

        with open(grid_dir / 'best_params.json', 'w') as f:
            json.dump(best_params, f)

        with open(grid_dir / 'best_params.yaml', 'w') as f:
            yaml.dump(best_params, f, default_flow_style=False)

    out_dir = grid_dir / f'final_run'
    out_dir.mkdir(exist_ok=True)
    cfg.out_dir = str(out_dir)
    with open(out_dir / 'params.json', 'w') as f:
        json.dump(best_params, f)
    update_cfg(cfg, best_params)
    dump_cfg(cfg)
    run_main(cfg=cfg, wandb_base_name=f'{experiment_name}-final_run', repeat=repeat, subset=None)
