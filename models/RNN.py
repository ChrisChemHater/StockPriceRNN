from typing import Callable, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

import time
from torch.optim import Adam
from .load_data import DataLoader

from sklearn.metrics import r2_score, mean_squared_error


class StockRNN(nn.Module):
    """
    input: (batch, seq_len, 1) 3-D Tensor
    output: (batch,) 1-D Tensor

    :param hidden_dim:
    :param num_layers:

    :method predict:
        :param x: (seq_len, 1) 2-D Tensor
        :param periods: int, num of periods to predict
        :return (seq_len) 1-D Tensor
    """
    def __init__(self, hidden_dim: int, num_layers: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.dense = nn.Linear(hidden_dim, 1)

    def predict(self, x: torch.Tensor, periods: int) -> torch.Tensor:
        predicts = torch.zeros(size=(periods,))
        for i in range(periods):
            predicts[i] = self(x.view(1, -1, 1))[0]
            x = torch.cat([x[1:, :], torch.Tensor([[predicts[i]]])])
        return predicts


class StockLSTM(StockRNN):
    __doc__ = StockRNN.__doc__

    def __init__(self, hidden_dim: int, num_layers: int):
        super().__init__(hidden_dim, num_layers)
        self.rnn = nn.LSTM(1, hidden_dim, num_layers, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        h0 = torch.zeros(size=(self.num_layers, len(x), self.hidden_dim), device=device)
        out, (hn, cn) = self.rnn(x, (h0, h0))
        out = self.dense(out[:, -1, :]).flatten()
        return out


class StockGRU(StockRNN):
    __doc__ = StockRNN.__doc__

    def __init__(self, hidden_dim: int, num_layers: int):
        super().__init__(hidden_dim, num_layers)
        self.rnn = nn.GRU(1, hidden_dim, num_layers, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        h0 = torch.zeros(size=(self.num_layers, len(x), self.hidden_dim), device=device)
        out, hn = self.rnn(x, h0)
        out = self.dense(out[:, -1, :]).flatten()
        return out


def train(model: StockRNN,
          trainData: DataLoader,
          epochs: int,
          batch_size: int,
          optimizer: torch.optim.Optimizer,
          show_every: int = 20,
          record_every: int = 1,
          verbose:bool = True) -> Tuple[StockRNN, Adam, torch.Tensor]:
    criterion = nn.MSELoss()
    record = []

    start_time = time.time()
    for epoch in range(epochs):
        X, y = trainData.getTrainData(batch_size)
        pred = model(X)

        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % show_every == 0:
            print(f"Epoch {epoch}/{epochs}: MSE = {loss.item()}")
        
        if (epoch + 1) % record_every == 0:
            record.append([epoch, loss.item()])

    end_time = time.time()
    if verbose:
        print(f"traing process finished in {end_time - start_time:.3f} seconds.\n"
            f"final MSE score = {loss.item()}")

    return model, optimizer, torch.Tensor(record)

def validate(model:StockRNN, trainData:DataLoader, scorer:Callable=mean_squared_error) -> float:
    Xvalid, yvalid = trainData.getValidData()
    return scorer(yvalid.detach().cpu(), model(Xvalid).detach().cpu())

def evaluate(model:StockRNN, testData:DataLoader, scorer:Callable=mean_squared_error) -> float:
    Xtest, ytest = testData.getTestData()
    return scorer(ytest.detach().cpu(), model.predict(Xtest, periods=len(ytest)).detach().cpu())

def evaluate_roll(model:StockRNN, testData:DataLoader, scorer:Callable=mean_squared_error) -> float:
    Xtest, ytest = testData.getRollTestData()
    return scorer(ytest.detach().cpu(), model(Xtest).detach().cpu())

def test():
    from .load_data import DataLoader
    DATA_PATH = 'data'
    AAPL_TRAIN = 'aapl_train.csv'
    AAPL_TEST = 'aapl_test.csv'
    MSFT_TRAIN = 'msft_train.csv'
    MSFT_TEST = 'msft_test.csv'
    trainLoader = DataLoader(f"{DATA_PATH}/{AAPL_TRAIN}",
                             look_back=24,
                             train=True,
                             valid_split=True,
                             valid_frac=0.2,
                             random_state=1234)
    testLoader = DataLoader(
        f"{DATA_PATH}/{AAPL_TEST}",
        look_back=24,
        train=False,
        train_params=trainLoader.getTrainParams(),
    )
    model = StockLSTM(10, 2)
    optimizer = Adam(model.parameters(), lr=5e-4)
    train(model,
          trainLoader,
          epochs=20,
          batch_size=60,
          optimizer=optimizer,
          show_every=50)
    print(validate(model, trainLoader))
    print(evaluate(model, testLoader))