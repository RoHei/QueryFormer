from typing import Callable, List

import numpy as np
import pandas as pd
import os
import time
import torch
from scipy.stats import pearsonr
from torch import nn
from torch.nn import MSELoss

from model.database_util import collator, Encoding
from model.dataset import PlanTreeDataset
from model.util import Normalizer


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def print_qerror(preds_unnorm, labels_unnorm):
    qerror = []
    for i in range(len(preds_unnorm)):
        if preds_unnorm[i] > float(labels_unnorm[i]):
            qerror.append(preds_unnorm[i] / float(labels_unnorm[i]))
        else:
            qerror.append(float(labels_unnorm[i]) / float(preds_unnorm[i]))
    print({'q_median': np.median(qerror), 'q_90': np.percentile(qerror, 90), 'q_mean': np.mean(qerror)})


def get_corr(ps, ls):  # unnormalized
    ps = np.array(ps)
    ls = np.array(ls)
    corr, _ = pearsonr(np.log(ps), np.log(ls))
    return corr


def eval_workload(workload, get_sample: Callable, encoding: Encoding, cost_norm: Normalizer,
                  hist_file: List[dict],  model: nn.Module, batch_size: int, device: str):

    workload_file_name = './data/imdb/workloads/' + workload
    table_sample = get_sample(workload_file_name)

    plan_df = pd.read_csv('./data/imdb/{}_plan.csv'.format(workload))

    dataset = PlanTreeDataset(query_plans_df=plan_df,
                              encoding=encoding,
                              hist_file=hist_file,
                              cost_norm=cost_norm,
                              card_norm=cost_norm,
                              table_sample=table_sample)


    model.eval()
    all_cost_predictions = np.empty(0)
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch, batch_labels = collator(list(zip(*[dataset[j] for j in range(i, min(i + batch_size, len(dataset)))])))
            batch = batch.to(device)
            cost_preds, _ = model(batch)
            cost_preds = cost_preds.squeeze()
            all_cost_predictions = np.append(all_cost_predictions, cost_preds.cpu().detach().numpy())
    print_qerror(cost_norm.unnormalize_labels(all_cost_predictions), dataset.costs)


def train_model(model: nn.Module,
                train_dataset: PlanTreeDataset,
                validation_dataset: PlanTreeDataset,
                cost_norm: Normalizer,
                crit: nn.Module = MSELoss(),
                to_predict: str = "cost",
                batch_size: int = 128,
                device: str = "cpu",
                epochs: int = 200,
                clip_size: float = 1.0,
                lr: float = 0.001,
                optimizer=None,
                scheduler=None,
                args=None):

    # Init training
    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if not scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, 0.7)

    start_time = time.time()
    rng = np.random.default_rng()
    best_score = 999999

    # Start training
    model.train()
    print("Start training")
    for epoch in range(epochs):
        print(f'Epoch: {epoch}')
        losses = 0
        cost_predictions = np.empty(0)
        train_indexes = rng.permutation(len(train_dataset))
        cost_labels = np.array(train_dataset.costs)[train_indexes]

        for indexes in chunks(train_indexes, batch_size):
            optimizer.zero_grad()
            batch, batch_labels = collator(list(zip(*[train_dataset[j] for j in indexes])))
            l, r = zip(*batch_labels)
            batch_cost_label = torch.FloatTensor(l).to(device)
            batch = batch.to(device)
            cost_preds, _ = model(batch)
            cost_preds = cost_preds.squeeze()
            loss = crit(cost_preds, batch_cost_label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_size)
            optimizer.step()
            losses += loss.item()
            cost_predictions = np.append(cost_predictions, cost_preds.detach().cpu().numpy())

        #if epoch > 40:
        #    test_scores, correlations = evaluate_model(model, validation_dataset, batch_size, cost_norm, device, False)
        #    if test_scores['q_mean'] < best_score:
        #        best_model_path = logging(args, epoch, test_scores, filename='log.txt', save_model=True, model=model)
        #        best_score = test_scores['q_mean']
        print('Epoch: {}  Avg Loss: {}, Time: {}'.format(epoch, losses / len(train_dataset), time.time() - start_time))
        print_qerror(cost_norm.unnormalize_labels(cost_predictions), cost_labels)
        scheduler.step()
    return model


def logging(args, epoch, qscores, filename=None, save_model=False, model=None):
    arg_keys = [attr for attr in dir(args) if not attr.startswith('__')]
    arg_vals = [getattr(args, attr) for attr in arg_keys]
    res = dict(zip(arg_keys, arg_vals))
    model_checkpoint = str(hash(tuple(arg_vals))) + '.pt'
    res['epoch'] = epoch
    res['model'] = model_checkpoint

    res = {**res, **qscores}
    filename = args.newpath + filename
    model_checkpoint = args.newpath + model_checkpoint

    if filename is not None:
        if os.path.isfile(filename):
            df = pd.read_csv(filename)
            df = df.append(res, ignore_index=True)
            df.to_csv(filename, index=False)
        else:
            df = pd.DataFrame(res, index=[0])
            df.to_csv(filename, index=False)
    if save_model:
        torch.save({'model': model.state_dict(), 'args': args}, model_checkpoint)
    return res['model']
