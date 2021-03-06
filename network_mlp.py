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
    'method': 'train',
    'metrics': ['cosine', 'sine', 'limit'],
    'in_feature': 5,
    'learning_rate': 1e-3,
    'weight_decay': 1e-7,
    'epochs': 200,
    'train_batch_size': 64,
    'validate_batch_size': 100,
    'log_interval': 1,
    'model_name': {'cosine': 'cosine-200-0.0354-0.0843.pt',
                   'sine': 'sine-200-0.0395-0.0884.pt',
                   'limit': 'limit-52-0.0024-0.0043.pt'}
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
    def __init__(self, input_data, output_data, foil_paras, seq, dataset):
        train_count = int(len(input_data) * 0.7)
        validate_count = int(len(input_data) * 0.2)
        if dataset == 'train':
            seq = seq[:train_count]
        elif dataset == 'validate':
            seq = seq[train_count:train_count + validate_count]
        elif dataset == 'test':
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
            samples = (batch_idx + 1) * options['train_batch_size']
            samples = samples if samples < dataset_size else dataset_size
            print("\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.8f"
                  % (epoch + 1, samples, dataset_size, loss_val / len(data_loader)), end='')
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


def save_model(model, metric, epochs):
    path = './model_mlp/%s-%d-%.4f-%.4f' \
           % (metric, epochs, train_loss[-1], validate_loss[-1])
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
    plt.clf()
    return model_path


def network_mlp():
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    result = []

    for metric in options['metrics']:
        input_data = np.loadtxt('./data/input_mlp.csv', delimiter=',')
        output_data = np.loadtxt('./data/output_mlp.csv', delimiter=',')
        foil_paras = np.loadtxt('./data/foil_paras.csv', delimiter=',')

        # ??????????????????
        out_feature = 0
        if metric == 'cosine':
            out_feature = 31
            output_data = output_data[:, :31]
        elif metric == 'sine':
            out_feature = 30
            output_data = output_data[:, 31:61]
        elif metric == 'limit':
            out_feature = 2
            output_data = output_data[:, 61:63]
        assert out_feature != 0

        # ????????????????????????
        input_mean = np.mean(input_data, axis=0)
        input_std = np.std(input_data, axis=0)
        output_mean = np.mean(output_data, axis=0)
        output_std = np.std(output_data, axis=0)
        foil_paras_mean = np.mean(foil_paras, axis=0)
        foil_paras_std = np.std(foil_paras, axis=0)
        input_data = (input_data - input_mean) / input_std
        output_data = (output_data - output_mean) / output_std
        foil_paras = (foil_paras - foil_paras_mean) / foil_paras_std

        # seq = np.random.permutation(input_data.shape[0])
        # seq.tofile('./data/seq_mlp.txt', sep=',')
        seq = np.loadtxt('./data/seq_mlp.txt', delimiter=',', dtype=int)

        if options['method'] == 'train':
            # ????????????
            model = Net(options['in_feature'], out_feature)
            model.double()
            print(model)

            # ????????????
            train_dataset = CustomDataset(input_data, output_data, foil_paras, seq, 'train')
            train_dataset_size = len(train_dataset)
            train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                            batch_size=options['train_batch_size'],
                                                            num_workers=4,
                                                            pin_memory=True)

            # ????????????
            validate_dataset = CustomDataset(input_data, output_data, foil_paras, seq, 'validate')
            validate_data_loader = torch.utils.data.DataLoader(validate_dataset,
                                                               batch_size=options['validate_batch_size'],
                                                               num_workers=4,
                                                               pin_memory=True)
            # ?????????????????????
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
                    model_path = save_model(model, metric, epoch + 1)
                    print('End training, model saved in %s' % model_path)
                    break

        elif options['method'] == 'test':
            model_name = options['model_name'][metric]
            model = torch.load('./model_mlp/' + model_name)
            print(model)
            test_dataset = CustomDataset(input_data, output_data, foil_paras, seq, 'test')
            test_dataset_size = len(test_dataset)
            test_data_loader = torch.utils.data.DataLoader(test_dataset, num_workers=4)
            out = predict(model, test_data_loader)
            predict_data = np.array(out).reshape(test_dataset_size, out_feature)
            predict_data = predict_data * output_std + output_mean
            result.append(predict_data)

    if options['method'] == 'test':
        data = np.column_stack([x for x in result])
        np.savetxt('./output/output.csv', data, delimiter=',')
