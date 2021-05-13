from typing import Tuple
import numpy as np
from numpy.random import default_rng
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch


class DataLoader(object):
    def __init__(self,
                 fname: str,
                 look_back: int = 20,
                 train: bool = True,
                 train_params: dict = None,
                 valid_split: bool = False,
                 valid_frac: float = 0.2,
                 random_state: int = None,
                 device: str = 'cpu'):
        # random generator
        self._rng = default_rng(
            random_state if random_state else np.random.randint(0x1000))

        self.train = train
        self.valid_split = valid_split
        self.valid_frac = valid_frac
        self.look_back = look_back
        self.device = device

        # load data from file and preprocessing
        df = pd.read_csv(fname, index_col='date')
        if pd.isna(df).values.any():
            print("There are NA values, trying to fill with 0...")
            df.fillna(0, inplace=True)
        self._data = df.values

        if train:
            # Train mode: fit the scaler, and get unrolled data
            self.scaler = StandardScaler().fit(self._data)
            # shape = (dataSize, look_back + 1, 1)
            self.data = DataLoader._getData(
                self.scaler.transform(self._data).flatten(), look_back)
            self._rng.shuffle(self.data)
        else:
            # Test mode: load the scaler, and transform the data
            self.scaler = train_params['scaler']
            if train_params['look_back'] != look_back:
                raise ValueError(
                    f'look_back parameter in train loader is not the same with test loader'
                )
            self._data = np.r_[train_params['test_head'], self._data]
            # shape = (dataSize, look_back + 1, 1)
            self.data = DataLoader._getData(
                self.scaler.transform(self._data).flatten(), look_back)

        self.data = torch.from_numpy(self.data).float().to(self.device)

        if train and valid_split:
            self.trainData, self.testData = train_test_split(
                self.data, test_size=valid_frac, shuffle=False)

    @classmethod
    def _getData(cls, data: np.ndarray, look_back: int) -> np.ndarray:
        dataLen = len(data) - look_back
        # 索引矩阵，用于生成data矩阵
        idxM = np.repeat(np.arange(look_back + 1).reshape(1, -1),
                         dataLen,
                         axis=0) + np.arange(dataLen).reshape(-1, 1)
        dataM = data[idxM]
        return dataM

    def getTrainData(self,
                     batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        trainData = self.trainData if self.valid_split else self.data
        batch = trainData[(self._rng.random(batch_size) *
                           len(trainData)).astype(int)]
        return batch[:, :-1].view(batch_size, -1, 1), batch[:, -1]

    def getValidData(self) -> Tuple[torch.Tensor, torch.Tensor]:
        Xvalid = self.testData[:, :-1].view(self.testData.shape[0], -1, 1)
        yvalid = self.testData[:, -1]
        return Xvalid, yvalid

    def getTestData(self) -> Tuple[torch.Tensor, torch.Tensor]:
        Xtest = self.data[-1, :-1].view(-1, 1)
        ytest = self.data[:, -1]
        return Xtest, ytest

    def getTrainParams(self) -> dict:
        testHead = self._data[-self.look_back:, :]
        return {
            "scaler": self.scaler,
            "test_head": testHead,
            "look_back": self.look_back
        }


if __name__ == "__main__":
    import os
    os.chdir('data')
    dataloader = DataLoader('aapl_train.csv',
                            look_back=12,
                            train=True,
                            valid_split=True,
                            random_state=1234)
    print(dataloader.getTrainData(56)[0].shape)
    print(dataloader.getValidData()[0].shape)
    testDL = DataLoader('aapl_test.csv',
                        look_back=12,
                        train=False,
                        train_params=dataloader.getTrainParams())
    print(testDL.getTestData()[0].shape)
