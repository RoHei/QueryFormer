from model.util import Normalizer
from model.database_util import get_hist_file, get_job_table_sample
from model.model import QueryFormer
from model.dataset import PlanTreeDataset
from model.trainer import eval_workload, train_model
import torch
import torch.nn as nn
import pandas as pd
import os
from model.util import seed_everything

data_path = './data/imdb/'


class Args:
    bs = 128
    lr = 0.001
    epochs = 200
    clip_size: int = 50
    embed_size: int = 64
    pred_hid: int = 128
    ffn_dim = 128
    head_size = 12
    n_layers = 8
    dropout = 0.1
    sch_decay = 0.6
    device: str = "cpu"
    newpath: str = './results/full/cost/'
    target_variable = 'cost'


if __name__ == '__main__':
    imdb_path = "./data/imdb/"
    target_variable = 'cost'

    args = Args()
    if not os.path.exists(args.newpath):
        os.makedirs(args.newpath)

    hist_file = get_hist_file(data_path + 'histogram_string.csv')
    cost_norm = Normalizer(-3.61192, 12.290855)
    card_norm = Normalizer(1, 100)

    encoding = torch.load('checkpoints/encoding.pt')['encoding']
    checkpoint = torch.load('checkpoints/cost_model.pt', map_location='cpu')
    seed_everything()

    model = QueryFormer(embedding_size=args.embed_size,
                        ffn_dim=args.ffn_dim,
                        head_size=args.head_size,
                        dropout=args.dropout,
                        n_layers=args.n_layers,
                        use_sample=True,
                        use_histogram=True,
                        hidden_dim_prediction=args.pred_hid)

    _ = model.to(args.device)

    train_dfs = []
    for i in range(0, 1):  # ToDO: Change later
        file = imdb_path + 'plan_and_cost/train_plan_part{}.csv'.format(i)
        df = pd.read_csv(file)
        train_dfs.append(df)
    full_train_df = pd.concat(train_dfs)

    val_dfs = []
    for i in range(19, 20):
        file = imdb_path + 'plan_and_cost/train_plan_part{}.csv'.format(i)
        df = pd.read_csv(file)
        val_dfs.append(df)
    full_val_df = pd.concat(val_dfs)

    table_sample = get_job_table_sample(imdb_path + 'train')

    train_dataset = PlanTreeDataset(query_plans_df=full_train_df,
                                    encoding=encoding,
                                    hist_file=hist_file,
                                    card_norm=card_norm,
                                    cost_norm=cost_norm,
                                    target_variable=target_variable,
                                    table_sample=table_sample)

    validation_dataset = PlanTreeDataset(query_plans_df=full_val_df,
                                         encoding=encoding,
                                         hist_file=hist_file,
                                         card_norm=card_norm,
                                         cost_norm=cost_norm,
                                         target_variable=target_variable,
                                         table_sample=table_sample)

    model = train_model(model=model,
                        train_dataset=train_dataset,
                        validation_dataset=validation_dataset,
                        cost_norm=cost_norm,
                        args=args)

    for workload in ['job-light', 'synthetic']:
        eval_workload(workload=workload,
                      get_sample=get_job_table_sample,
                      encoding=encoding,
                      cost_norm=cost_norm,
                      hist_file=hist_file,
                      model=model,
                      device="cpu",
                      batch_size=512)
