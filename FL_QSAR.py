import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import pandas as pd
import crypten
import argparse


class Setting:
    """Parameters for training"""

    def __init__(self):
        self.epoch = 180
        self.lr = 0.05
        self.momentum = 0.9
        self.weight_decay = 0.0001
        self.batch_size = 128


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 4000)
        self.fc2 = nn.Linear(4000, 2000)
        self.fc3 = nn.Linear(2000, 1000)
        self.fc4 = nn.Linear(1000, 1000)
        self.fc5 = nn.Linear(1000, 1)
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.25)
        self.dropout3 = nn.Dropout(p=0.25)
        self.dropout4 = nn.Dropout(p=0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(self.dropout1(x)))
        x = F.relu(self.fc3(self.dropout2(x)))
        x = F.relu(self.fc4(self.dropout3(x)))
        x = self.fc5(self.dropout4(x))
        return x


def normalize(target_array, data_max, data_min):
    return (target_array - data_min) / (data_max - data_min)


def logarithmic(target_array):
    return np.log(target_array + 1)


def calculate_r_square(output, target):
    return 1 - torch.div(torch.sum((output - target).pow(2)),
                         torch.sum((target - target.mean()).pow(2)))


def preprocess(train_file, test_file):
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    x_train_origin = np.array(train_data.iloc[:, 2:].values)
    y_train_origin = np.array(train_data.iloc[:, 1].values)
    x_test_origin = np.array(test_data.iloc[:, 2:].values)
    y_test_origin = np.array(test_data.iloc[:, 1].values)
    data_max = y_train_origin.max() if y_train_origin.max(
    ) > y_test_origin.max() else y_test_origin.max()
    data_min = y_train_origin.min() if y_train_origin.min(
    ) < y_test_origin.min() else y_test_origin.min()
    y_train_origin = y_train_origin.reshape((-1, 1))
    y_test_origin = y_test_origin.reshape((-1, 1))
    train_x = logarithmic(x_train_origin)
    train_y = normalize(y_train_origin, data_max, data_min)
    test_x = logarithmic(x_test_origin)
    test_y = normalize(y_test_origin, data_max, data_min)
    train_x = torch.tensor(train_x, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)
    test_x = torch.tensor(test_x, dtype=torch.float32)
    test_y = torch.tensor(test_y, dtype=torch.float32)
    train_set = TensorDataset(train_x, train_y)
    return train_set, test_x, test_y


def divide_trainset_to_client(train_set, cli_num, BATCH_SIZE):
    length_list = []
    train_sets = []
    for _ in range(cli_num - 1):
        length_list.append(len(train_set) // cli_num)
    length_list.append(len(train_set) - (cli_num - 1)* (len(train_set) // cli_num))
    train_sets_pre = random_split(train_set, length_list)
    for i in train_sets_pre:
        train_sets.append(DataLoader(i, batch_size=BATCH_SIZE))
    return train_sets


def define_network(cli_num, lr_=0.05, momentum_=0.9, weight_decay_=0.0001):
    createVar = locals()
    optimizers = []
    models = []
    params = []
    for i in range(cli_num):
        k = str(i + 1)
        model_name = 'model_' + k
        opti_name = 'optimizer_' + k
        createVar[model_name] = Net().to(device)
        createVar[opti_name] = optim.SGD(
            locals()[model_name].parameters(),
            lr=lr_,momentum=momentum_,weight_decay=weight_decay_
            )
        models.append(locals()[model_name])
        params.append(list(locals()[model_name].parameters()))
        optimizers.append(locals()[opti_name])
    return models, optimizers, params


def fl_train(train_sets, fl_models, fl_optimizers, params):
    new_params = list()
    for k in range(len(train_sets)):
        for batch_idx, (data, target) in enumerate(train_sets[k]):
            fl_optimizers[k].zero_grad()
            data, target = data.to(device), target.to(device)
            output = fl_models[k](data)
            loss = criterion(output, target)
            loss.backward()
            fl_optimizers[k].step()
    for param_i in range(len(params[0])):
        fl_params = list()
        for remote_index in range(cli_num):
            clone_param = params[remote_index][param_i].clone().cpu()
            fl_params.append(crypten.cryptensor(torch.tensor(clone_param)))
        sign = 0
        for i in fl_params:
            if sign == 0:
                fl_param = i
                sign = 1
            else:
                fl_param = fl_param + i

        new_param = (fl_param / cli_num).get_plain_text()
        new_params.append(new_param)

    with torch.no_grad():
        for model_para in params:
            for param in model_para:
                param *= 0

        for remote_index in range(cli_num):
            for param_index in range(len(params[remote_index])):
                new_params[param_index] = new_params[param_index].to(device)
                params[remote_index][param_index].set_(new_params[param_index])
    return fl_models


def train(dataloader, model, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    return model


def test(test_x, test_y, model):
    model.eval()
    with torch.no_grad():
        output = model(test_x)
        r_square = calculate_r_square(output.detach(), test_y).item()
        return r_square


parser = argparse.ArgumentParser()
parser.add_argument('-tr','--train_data',required=False,type=str,
    help="train_data",default='./data/METAB_training.csv')
parser.add_argument('-te','--test_data',required=False,type=str,
    help="test_data",default='./data/METAB_test.csv')
parser.add_argument('-n', '--cli_num', required=False, type=int,
                    help="client_number", default=3)
out_args = parser.parse_args()
crypten.init()
args = Setting()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cli_num = out_args.cli_num
epoch = args.epoch
criterion = nn.MSELoss()
train_set, test_x, test_y = preprocess(out_args.train_data, out_args.test_data)
test_x, test_y = test_x.to(device), test_y.to(device)
input_size = test_x.shape[1]
train_sets = divide_trainset_to_client(
    train_set, cli_num, BATCH_SIZE=args.batch_size)
models, optimizers, _ = define_network(
    cli_num, lr_=args.lr, momentum_=args.momentum, weight_decay_=args.weight_decay)
fl_models, fl_optimizers, params = define_network(
    cli_num, lr_=args.lr, momentum_=args.momentum, weight_decay_=args.weight_decay)
with open('./result/result.txt', 'w') as f:
    for n in range(cli_num):
        max_r_square = 0
        for _ in range(epoch):
            model = train(train_sets[n], models[n], optimizers[n])
            r_square = test(test_x, test_y, model)
            if r_square > max_r_square:
                max_r_square = r_square
                torch.save(model.state_dict(),
                           './result/model_' + str(n + 1) + '.pkl')
        f.write('client_' + str(n + 1) +
                ' max_r_square: ' + str(round(max_r_square, 3)) + '\n')
    fl_max_r_square = 0
    for _ in range(epoch):
        fl_models = fl_train(train_sets, fl_models, fl_optimizers, params)
        fl_r_square = test(test_x, test_y, fl_models[0])
        if fl_r_square > fl_max_r_square:
            fl_max_r_square = fl_r_square
            torch.save(fl_models[0].state_dict(), './result/fl_model.pkl')
    f.write('FL max_r_square: ' + str(round(fl_max_r_square, 3)) + '\n')
