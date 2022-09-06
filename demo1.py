import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# 读取数据到内存
train_data = pd.read_csv('data_ml2022spring-hw1/covid.train.csv').values
test_data = pd.read_csv('data_ml2022spring-hw1/covid.test.csv').values
print(train_data.shape, test_data.shape)

# 数据预处理
train_features = train_data[:2500, 1:-1]
train_lables = train_data[:2500, -1]
test_features = test_data[:, 1:]

valid_features = train_data[2500:, 1:-1]
valid_lables = train_data[2500:, -1]
print(train_features.shape, train_lables.shape, test_features.shape, valid_features.shape, valid_lables.shape)


# 迭代器
class COVID19Dataset(Dataset):

    def __int__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)

        train_dataset = COVID19Dataset(train_features, train_lables)
        valid_dataset = COVID19Dataset(valid_features, valid_lables)
        test_dataset = COVID19Dataset(test_features)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, shuffle=False)


# 模型构建

class My_Model(nn.Module):
    def __init__(self, input_dim):
        super(My_Model, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1)
        return x


model = My_Model(116)
print(model)

# 损失函数和优化器
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)


# 训练
def trainer(model, train_loader, valid_loader):
    for epoches in range(400):
        valid_loss = []
        train_loss = []
        score = 1e5
        for x, y in train_loader:
            optimizer.zero.gard()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss)
        with torch.no_gard():
            for x, y in valid_loader:
                y_hat = model(x)
                loss = criterion(y_hat, y)
                valid_loss.append(loss)

        print("epoch:", epoches, "train_loss:", sum(train_loss) / len(train_loss), "valid_loss:",
              sum(valid_loss) / len(valid_loss), "\n")

    trainer(model, train_loader, valid_loader)


y_pred = model(torch.tensor(test_features, dtype=torch.float))
preds = y_pred.detach().numpy().reshape(-1, 1)
test_data1 = pd.read_csv('data_ml2022spring-hw1/covid.test.csv')
test_data1['tested_positive'] = preds

sub = pd.concat([test_data1['id'], test_data1['tested_positive']], axis=1)

sub.to_csv('data_ml2022spring-hw1/submission1.csv', index=False)
