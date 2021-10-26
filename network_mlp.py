import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset
from EarlyStopping import EarlyStopping

train_loss = []
validate_loss = []
options = {
    'method': 'test',
    'type': 'cosine',
    'in_feature': 5,
    'learning_rate': 1e-3,
    'weight_decay': 1e-7,
    'epochs': 200,
    'train_batch_size': 64,
    'validate_batch_size': 100,
    'log_interval': 1
}


class Net(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_feature, 100)
        self.fc2 = nn.Linear(3, 100)
        self.fc3 = nn.Linear(200, 500)
        self.fc4 = nn.Linear(500, 250)
        self.fc5 = nn.Linear(250, 125)
        self.fc6 = nn.Linear(125, out_feature)

    def forward(self, data, foil):
        data = F.leaky_relu(self.fc1(data))
        foil = F.leaky_relu(self.fc2(foil))

        x = torch.cat((data, foil), dim=1)
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        x = F.leaky_relu(self.fc5(x))
        x = self.fc6(x)
        return x


class CustomDataset(Dataset):
    def __init__(self, input_data, output_data, foil_paras, seq, type):
        train_count = int(len(input_data) * 0.7)
        validate_count = int(len(input_data) * 0.2)
        if type == 'train':
            seq = seq[:train_count]
        elif type == 'validate':
            seq = seq[train_count:train_count + validate_count]
        elif type == 'test':
            seq = seq[train_count + validate_count:]
        self.source = input_data[seq, :]
        self.foil = foil_paras[seq, :]
        self.target = output_data[seq, :]

    def __getitem__(self, idx):
        return self.source[idx], self.foil[idx], self.target[idx]

    def __len__(self):
        return len(self.source)


def train(epoch, model, data_loader, optimizer, dataset_size):
    model.train()
    loss_val = 0
    criterion = nn.MSELoss()
    for batch_idx, (data, foil, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model.forward(data, foil)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_val += loss.item()
        if batch_idx % options['log_interval'] == 0:
            sum = (batch_idx + 1) * options['train_batch_size']
            sum = sum if sum < dataset_size else dataset_size
            print("\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.8f"
                  % (epoch + 1, sum, dataset_size, loss_val / len(data_loader)), end='')
    loss_val = loss_val / len(data_loader)
    train_loss.append(loss_val)


def validate(model, data_loader, optimizer):
    loss_val = 0
    model.eval()
    criterion = nn.MSELoss()
    optimizer.zero_grad()
    for batch_idx, (data, foil, target) in enumerate(data_loader):
        output = model.forward(data, foil)
        loss = criterion(output, target)
        loss_val += loss.item()
    loss_val = loss_val / len(data_loader)
    print("\nValidate set: Average loss: %.8f" % loss_val)
    validate_loss.append(loss_val)


def predict(model, data_loader):
    model.eval()
    output_list = []
    with torch.no_grad():
        for batch_idx, (data, foil, target) in enumerate(data_loader):
            output = model.forward(data, foil)
            output_list.append(output.data.cpu().numpy())
    return output_list


def save_model(model, epochs):
    path = './model_mlp/%s-%d-%.4f-%.4f' \
           % (options['type'], epochs, train_loss[-1], validate_loss[-1])
    model_path = path + '.pt'
    figure_path = path + '.png'
    loss_path = path + '.txt'
    torch.save(model, model_path)

    loss = np.concatenate([np.array(train_loss).reshape([len(train_loss), 1]),
                           np.array(validate_loss).reshape([len(validate_loss), 1])], axis=1)
    loss.tofile(loss_path, sep=',')

    plt.gcf().set_size_inches(8, 6)
    plt.gcf().set_dpi(150)
    plt.plot(np.linspace(0, epochs, epochs).tolist(), train_loss, label='train')
    plt.plot(np.linspace(0, epochs, epochs).tolist(), validate_loss, label='validate')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(figure_path)
    return model_path


def network_mlp():
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input = np.loadtxt('./data/input_mlp.csv', delimiter=',')
    output = np.loadtxt('./data/output_mlp.csv', delimiter=',')
    foil_paras = np.loadtxt('./data/foil_paras.csv', delimiter=',')

    # 划分输出数据
    out_feature = 0
    if options['type'] == 'cosine':
        out_feature = 31
        output = output[:, :31]
    elif options['type'] == 'sine':
        out_feature = 30
        output = output[:, 31:61]
    elif options['type'] == 'limit':
        out_feature = 2
        output = output[:, 61:63]
    assert out_feature != 0

    # 归一化输入与输出
    input_mean = np.mean(input, axis=0)
    input_std = np.std(input, axis=0)
    output_mean = np.mean(output, axis=0)
    output_std = np.std(output, axis=0)
    foil_paras_mean = np.mean(foil_paras, axis=0)
    foil_paras_std = np.std(foil_paras, axis=0)
    input = (input - input_mean) / input_std
    output = (output - output_mean) / output_std
    foil_paras = (foil_paras - foil_paras_mean) / foil_paras_std

    seq = np.random.permutation(input.shape[0])
    seq.tofile('./data/seq_mlp.txt', sep=',')
    # seq = np.loadtxt('./data/seq_mlp.txt', delimiter=',', dtype=int)

    if options['method'] == 'train':
        # 构建网络
        model = Net(options['in_feature'], out_feature)
        model.double()
        print(model)

        # 训练数据
        train_dataset = CustomDataset(input, output, foil_paras, seq, 'train')
        train_dataset_size = len(train_dataset)
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=options['train_batch_size'],
                                                        num_workers=4,
                                                        pin_memory=True)

        # 验证数据
        validate_dataset = CustomDataset(input, output, foil_paras, seq, 'validate')
        validate_data_loader = torch.utils.data.DataLoader(validate_dataset,
                                                           batch_size=options['validate_batch_size'],
                                                           num_workers=4,
                                                           pin_memory=True)
        # 学习率指数衰减
        # optimizer = torch.optim.SGD(model.parameters(),
        #                             lr=options['learning_rate'],
        #                             momentum=0.9,
        #                             weight_decay=options['weight_decay'])
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=options['learning_rate'],
                                     weight_decay=options['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, verbose=True)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)

        early_stopping = EarlyStopping()
        for epoch in range(options['epochs']):
            print('learning rate: %.8f' % optimizer.param_groups[0]['lr'])
            train(epoch, model, train_data_loader, optimizer, train_dataset_size)
            validate(model, validate_data_loader, optimizer)
            scheduler.step()
            if early_stopping(validate_loss[-1], model):
                model_path = save_model(model, epoch + 1)
                print('End training, model saved in %s' % model_path)
                break

    elif options['method'] == 'test':
        model = torch.load('./model_mlp/cosine-200-0.0354-0.0843.pt')
        print(model)
        test_dataset = CustomDataset(input, output, foil_paras, seq, 'test')
        test_dataset_size = len(test_dataset)
        test_data_loader = torch.utils.data.DataLoader(test_dataset, num_workers=4)
        out = predict(model, test_data_loader)
        output = np.array(out).reshape(test_dataset_size, out_feature)
        output = output * output_std + output_mean
