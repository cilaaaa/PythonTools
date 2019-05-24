__author__ = 'Cila'
import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataTools(object):
    def __init__(self,
                 path="",
                 prepare_data=False,
                 input_size=180,
                 test_ratio=0.1,
                 normalized=False):
        self.path = path
        self.input_size = input_size
        self.test_ratio = test_ratio
        self.normalized = normalized
        if(prepare_data):
            rawDf = pd.read_excel(path,'Sheet1').values
            self.train_X, self.train_y, self.test_X, self.test_y = self._prepare_data(rawDf)
        else:
            self.train_X, self.train_y, self.test_X, self.test_y = self._load_data()

    def _prepare_data(self, df):
        # split into items of input_size
        X = []
        Y = []
        for i in range(self.input_size,len(df) - 1):
            tempX = []
            for j in range(self.input_size):
                tempX.append(df[i - self.input_size + j][1])
            if df[i + 1][1] - df[i][1] > 1:
                Y.append([1,0,0])
            elif df[i + 1][1] - df[i][1] < -1:
                Y.append([0,0,1])
            else:
                Y.append([0,1,0])
            X.append(tempX)
        pd.DataFrame(X).to_csv("../Data/Data" + str(self.input_size) + ".csv",index=None,header=None)
        pd.DataFrame(Y).to_csv("../Data/Label" + str(self.input_size) + ".csv",index=None,header=None)

        if self.normalized:
            scaler = StandardScaler()
            #Compute the mean and std to be used for later scaling.
            X = scaler.fit_transform(X)

        train_size = int(len(X) * (1.0 - self.test_ratio))

        train_X, test_X = X[:train_size], X[train_size:]
        train_y, test_y = Y[:train_size], Y[train_size:]
        return train_X, train_y, test_X, test_y

    def _load_data(self):
        XDf = pd.read_csv("../Data/Data" + str(self.input_size) + ".csv").values
        YDf = pd.read_csv("../Data/Label" + str(self.input_size) + ".csv").values
        X = XDf.tolist()
        Y = YDf.tolist()
        if self.normalized:
            scaler = StandardScaler()
            #Compute the mean and std to be used for later scaling.
            X = scaler.fit_transform(X)

        train_size = int(len(X) * (1.0 - self.test_ratio))

        train_X, test_X = X[:train_size], X[train_size:]
        train_y, test_y = Y[:train_size], Y[train_size:]
        return train_X, train_y, test_X, test_y

symbol = DataTools(normalized=True)
print(1)