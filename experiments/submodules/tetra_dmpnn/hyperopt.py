import os
import math
import torch
import pandas as pd
import optuna
import joblib

from argparse import ArgumentParser
import logging

from features.featurization import construct_loader
from utils import Standardizer, create_logger, get_loss_func

from model.gnn import GNN
from model.training import train, eval, test, build_lr_scheduler
from model.parsing import add_train_args, modify_train_args


def optimize(trial, args):

    setattr(args, 'hidden_size', int(trial.suggest_discrete_uniform('hidden_size', 300, 1200, 300)))
    setattr(args, 'depth', int(trial.suggest_discrete_uniform('depth', 2, 6, 1)))
    setattr(args, 'dropout', int(trial.suggest_discrete_uniform('dropout', 0, 1, 0.2)))
    setattr(args, 'lr', trial.suggest_loguniform('lr', 1e-5, 1e-3))
    setattr(args, 'batch_size', int(trial.suggest_categorical('batch_size', [25, 50, 100])))
    setattr(args, 'graph_pool', trial.suggest_categorical('graph_pool', ['sum', 'mean', 'max', 'attn', 'set2set']))

    setattr(args, 'log_dir', os.path.join(args.hyperopt_dir, str(trial._trial_id)))
    modify_train_args(args)

    torch.manual_seed(args.seed)
    train_logger = create_logger('train', args.log_dir)

    train_loader, val_loader = construct_loader(args)
    mean = train_loader.dataset.mean
    std = train_loader.dataset.std
    stdzer = Standardizer(mean, std, args.task)

    # create model, optimizer, scheduler, and loss fn
    model = GNN(args, train_loader.dataset.num_node_features, train_loader.dataset.num_edge_features).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = build_lr_scheduler(optimizer, args, len(train_loader.dataset))
    loss = get_loss_func(args)
    best_val_loss = math.inf
    best_epoch = 0

    # record args, optimizer, and scheduler info
    train_logger.info('Arguments are...')
    for arg in vars(args):
        train_logger.info(f'{arg}: {getattr(args, arg)}')
    train_logger.info(f'\nOptimizer parameters are:\n{optimizer}\n')
    train_logger.info(f'Scheduler state dict is:')
    for key, value in scheduler.state_dict().items():
        train_logger.info(f'{key}: {value}')
    train_logger.info('')

    # train
    train_logger.info("Starting training...")
    for epoch in range(0, args.n_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, loss, stdzer, args.device, scheduler, args.task)
        train_logger.info(f"Epoch {epoch}: Training Loss {train_loss}")

        val_loss, val_acc = eval(model, val_loader, loss, stdzer, args.device, args.task)
        train_logger.info(f"Epoch {epoch}: Validation Loss {val_loss}")

        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(args.log_dir, 'best_model'))

        # report intermediate results for early stopping
        trial.report(val_loss, epoch)

        # handle pruning based on the intermediate value
        if trial.should_prune():
            train_logger.handlers = []
            raise optuna.TrialPruned()

    train_logger.info(f"Best Validation Loss {best_val_loss} on Epoch {best_epoch}")

    # load best model
    model = GNN(args, train_loader.dataset.num_node_features, train_loader.dataset.num_edge_features).to(args.device)
    state_dict = torch.load(os.path.join(args.log_dir, 'best_model'), map_location=args.device)
    model.load_state_dict(state_dict)

    # predict test data
    test_loader = construct_loader(args, modes='test')
    preds, test_loss, test_acc, test_auc = test(model, test_loader, loss, stdzer, args.device, args.task)
    train_logger.info(f"Test Loss {test_loss}")

    # save predictions
    smiles = test_loader.dataset.smiles
    preds_path = os.path.join(args.log_dir, 'preds.csv')
    pd.DataFrame(list(zip(smiles, preds)), columns=['smiles', 'prediction']).to_csv(preds_path, index=False)

    train_logger.handlers = []
    return best_val_loss


if __name__ == '__main__':

    parser = ArgumentParser()
    add_train_args(parser)
    parser.add_argument('--hyperopt_dir', type=str,
                        help='Directory to save all results')
    parser.add_argument('--n_trials', type=int, default=25,
                        help='Number of hyperparameter choices to try')
    parser.add_argument('--restart', action='store_true', default=False,
                        help='Whether or not to resume study from previous .pkl file')
    args = parser.parse_args()

    if not os.path.exists(args.hyperopt_dir):
        os.makedirs(args.hyperopt_dir)

    logger = logging.getLogger()

    logger.setLevel(logging.INFO)  # Setup the root logger.
    logger.addHandler(logging.FileHandler(os.path.join(args.hyperopt_dir, "hyperopt.log"), mode="w"))

    optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

    if args.restart:
        study = joblib.load(os.path.join(args.hyperopt_dir, "study.pkl"))
    else:
        study = optuna.create_study(
            pruner=optuna.pruners.HyperbandPruner(min_resource=5, max_resource=args.n_epochs, reduction_factor=2),
            sampler=optuna.samplers.CmaEsSampler()
        )
    joblib.dump(study, os.path.join(args.hyperopt_dir, "study.pkl"))

    logger.info("Running optimization...")
    study.optimize(lambda trial: optimize(trial, args), n_trials=args.n_trials)
