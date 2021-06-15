# -*- coding:utf-8 -*-
"""
Parameters to optimize for LSTM and GRU:

- Model
    - look_back
    - hidden_dim
    - num_layers
- training
    - epochs
    - batch_size
    - lr
"""
from typing import Literal

from numpy import dtype
import joblib
from tap import Tap
import torch
from torch.optim import Adam
from hyperopt import hp, fmin, tpe, Trials, STATUS_FAIL
from models.load_data import DataLoader
from models.RNN import StockLSTM, StockGRU, train, validate

DATA_PATH = 'data'
AAPL_TRAIN = 'aapl_train.csv'
AAPL_TEST = 'aapl_test.csv'
MSFT_TRAIN = 'msft_train.csv'
MSFT_TEST = 'msft_test.csv'

TRAIN_FILES = {
    'aapl' : AAPL_TRAIN,
    'msft' : MSFT_TRAIN
}

class Parser(Tap):
    dataset: Literal['aapl', 'msft']
    model: Literal['lstm', 'gru']
    mode: Literal['model', 'train']
    device: str = 'cpu'

parser = Parser().parse_args()

# Dataset and Model selection, mode selection
DATASET = parser.dataset
MODEL = StockLSTM if parser.model == 'lstm' else StockGRU
MODE = parser.mode
DEVICE = parser.device

FILE = f"{DATA_PATH}/{TRAIN_FILES[DATASET]}"
if MODE == 'train':
    TRIALS = joblib.load(f'./log/{DATASET}_{MODEL.__name__}_model_optim.joblib')

def _objective(kwargs: dict) -> float:
    """
    the final objective to be optimized

    :params kwargs = {
        'look_back': int,
        'hidden_dim': int,
        'num_layers': int,
        'epochs': int,
        'batch_size': int,
        'lr': float
    }
    """
    trainLoader = DataLoader(FILE,
                             kwargs['look_back'],
                             valid_split=True,
                             random_state=1234,
                             device=DEVICE)
    model = MODEL(kwargs['hidden_dim'], kwargs['num_layers']).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=kwargs['lr'])
    train(model,
          trainLoader,
          kwargs['epochs'],
          kwargs['batch_size'],
          optimizer,
          show_every=100,
          record_every=10000)
    scores = torch.zeros(10)
    for i in range(10):  # 通过平均滤除随机波动，并检验方差
        train(model,
              trainLoader,
              1,
              kwargs['batch_size'],
              optimizer,
              verbose=False)
        scores[i] = torch.scalar_tensor(validate(model, trainLoader), dtype=torch.float32)
    # if scores.var().item() > 1e-3 or scores.mean().item() > 2e-2:  # 如果不收敛就弃去结果
    if scores.var().item() > 1e-2:  # 如果不收敛就弃去结果
        return {'status': STATUS_FAIL}
    else:
        return torch.log(scores.mean()).item()


def objective_model(kwargs: dict) -> float:
    fixed = {'epochs': 2000, 'batch_size': 240, 'lr': 1e-3}
    return _objective(kwargs | fixed)

def objective_train(kwargs: dict) -> float:
    fixed = TRIALS.argmin
    return _objective(kwargs | fixed)


def hyper_model() -> Trials:
    space = {
        # 'look_back': hp.randint('look_back', 10, 30),
        # 'hidden_dim': hp.randint('hidden_dim', 10, 30),
        'look_back': hp.randint('look_back', 20, 30),
        'hidden_dim': hp.randint('hidden_dim', 30, 40),
        'num_layers': hp.randint('num_layers', 1, 3)
    }
    trials = Trials()
    fmin(objective_model, space, tpe.suggest, max_evals=50, timeout=10 * 60, trials=trials)
    print("Best result:", trials.argmin)
    joblib.dump(trials, f'./log/{DATASET}_{MODEL.__name__}_model_optim.joblib')
    return trials

def hyper_train() -> Trials:
    space = {
        'epochs': hp.randint('epochs', 1800, 3000),
        'batch_size': hp.randint('batch_size', 200, 300),
        'lr': hp.loguniform('lr', -8., -5.)
    }
    trials = Trials()
    fmin(objective_train, space, tpe.suggest, max_evals=30, timeout=10 * 60, trials=trials)
    print("Best result:", trials.argmin)
    joblib.dump(trials, f'./log/{DATASET}_{MODEL.__name__}_train_optim.joblib')
    return trials

def hyper() -> Trials:
    if MODE == 'model':
        return hyper_model()
    elif MODE == 'train':
        return hyper_train()
    else:
        raise ValueError("'mode' param must be 'model' or 'train'")

if __name__ == "__main__":
    hyper()
