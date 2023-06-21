import json
import os
from pathlib import Path

import pandas as pd
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, set_cfg)

from main import custom_set_out_dir, custom_load_cfg

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


if __name__ == '__main__':
    # Load cmd line args
    args = parse_args()
    opts = args.opts
    repeat = args.repeat
    grid_config_path = Path(args.cfg_file)

    with open(grid_config_path, 'r') as f:
        grid_config = json.load(f)

    cfg_file = grid_config['base_config_path']
    set_cfg(cfg)
    custom_load_cfg(cfg=cfg, cfg_file=cfg_file, opts=opts)
    experiment_name = grid_config_path.name.split('.')[0]
    grid_dir = Path(cfg.out_dir) / experiment_name

    metric = cfg.metric_best
    direction = cfg.metric_agg

    results_list = []
    for param_path in os.listdir(grid_dir):
        if param_path.endswith('.yaml'):
            continue
        param_dir_path = grid_dir / param_path
        param_path = param_dir_path / 'params.json'
        test_path = param_dir_path / 'agg' / 'test' / 'best.json'
        valid_path = param_dir_path / 'agg' / 'val' / 'best.json'

        param_dict = load_json(param_path)
        test_dict = load_json(test_path)
        val_dict = load_json(valid_path)

        param_dict.update({
            f'test_{metric}': round(test_dict[metric], 3),
            f'val_{metric}': round(val_dict[metric], 3)
        })

        results_list.append(param_dict)

    df = pd.DataFrame(results_list)
    df = df.sort_values(by=f'val_{metric}', ascending=direction != 'argmax')
    df.to_csv('grid_result.csv', index=None)

