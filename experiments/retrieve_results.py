import json
from pathlib import Path

from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, set_cfg)

from main import custom_set_out_dir, custom_load_cfg

if __name__ == '__main__':
    # Load cmd line args
    args = parse_args()
    # Load config file
    if args.cfg_file.endswith('.json'):
        grid_config_path = Path(args.cfg_file)
        with open(grid_config_path, 'r') as f:
            grid_config = json.load(f)
        cfg_file = grid_config['base_config_path']
        set_cfg(cfg)
        custom_load_cfg(cfg=cfg, cfg_file=cfg_file, opts=args.opts)
        experiment_name = grid_config_path.name.split('.')[0]
        grid_dir = Path(cfg.out_dir) / experiment_name
        results_dir = grid_dir / 'final_run' / 'agg' / 'test'
    else:
        set_cfg(cfg)
        custom_load_cfg(cfg, args.cfg_file, args.opts)
        custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag)
        results_dir = Path(cfg.out_dir) / 'agg' / 'test'

    metric = cfg.metric_best
    metric_std = f'{cfg.metric_best}_std'
    best_path = results_dir / 'best.json'
    last_path = results_dir / 'stats.json'
    with open(best_path, 'r') as fp:
        best = json.load(fp)

    with open(last_path, 'r') as fp:
        last = list(fp.readlines())[-1]
        last = json.loads(last)


    def reformat(m, s):
        return f'{round(m, 3):.3f} \u00B1 {round(s, 3):.3f}'


    best_str = reformat(best[metric], best[metric_std])
    last_str = reformat(last[metric], last[metric_std])
    print(f'{best_str}\t{last_str}')
