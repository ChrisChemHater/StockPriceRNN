# -*- coding:utf-8 -*-
from typing import Literal

import joblib
from tap import Tap
import torch
from torch.optim import Adam
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
    device: str = 'cpu'

parser = Parser().parse_args()

# Dataset and Model selection, mode selection
DATASET = parser.dataset
MODEL = StockLSTM if parser.model == 'lstm' else StockGRU
DEVICE = parser.device

FILE = f"{DATA_PATH}/{TRAIN_FILES[DATASET]}"
FIXED = {'epochs': 1000, 'batch_size': 240, 'lr': 1e-3}

if __name__ == '__main__':
  pass