import sys

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

train_loss = []
validate_loss = []
options = {
    'method': 'train',
    'type': 'cosine',
    'in_feature': 5,
    'learning_rate': 1e-3,
    'weight_decay': 1e-7,
    'epochs': 500,
    'train_batch_size': 64,
    'validate_batch_size': 100,
    'log_interval': 1,
    'image_size': 80
}


class Net(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_feature, 100)
        self.fc1_bn = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 500)
        self.fc2_bn = nn.BatchNorm1d(500)

        self.conv1 = nn.Conv2d(1, 10, (5, 5), (2, 2))
        self.conv1_bn = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 10, (3, 3), (2, 2))
        self.conv2_bn = nn.BatchNorm2d(10)

        self.fc3 = nn.Linear(660, 125)
        self.fc4 = nn.Linear(125, out_feature)

    def forward(self, data, img):
        # MLP for parameters
        data = self.fc1_bn(F.relu(self.fc1(data)))
        data = self.fc2_bn(F.relu(self.fc2(data)))

        # Convolution network for airfoil image
        img = self.conv1_bn(F.relu(self.conv1(img)))
        img = F.max_pool2d(img, (2, 2), (2, 2))
        img = self.conv2_bn(F.relu(self.conv2(img)))
        img = F.max_pool2d(img, (2, 2), (2, 2))
        img = img.view([img.shape[0], -1])

        # Combine two models to one
        x = torch.cat((data, img), dim=1)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class CustomDataset(Dataset):
    def __init__(self, input_data, foil_name, output_data, seq, type):
        train_count = int(len(input_data) * 0.7)
        validate_count = int(len(input_data) * 0.2)
        if type == 'train':
            seq = seq[:train_count]
        elif type == 'validate':
            seq = seq[train_count:train_count + validate_count]
        elif type == 'test':
            seq = seq[train_count + validate_count:]
        self.source = input_data[seq, :]
        self.target = output_data[seq, :]
        self.img_path = foil_name[seq]
        self.transform = transforms.ToTensor()

    def __getitem__(self, idx):
        img = Image.open('./data/img/' + str(self.img_path[idx])).convert('L')
        img = img.resize((options['image_size'], options['image_size']), Image.ANTIALIAS)
        return torch.Tensor(self.source[idx]), self.transform(img), \
               torch.Tensor(self.target[idx])

    def __len__(self):
        return len(self.source)


def train(epoch, model, device, data_loader, optimizer, dataset_size):
    model.train()
    loss_val = 0
    for batch_idx, (data, img, target) in enumerate(data_loader):
        data, img, target = data.to(device), img.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data, img)
        loss = F.mse_loss(output, target)
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


def validate(model, device, data_loader, optimizer):
    loss_val = 0
    model.eval()
    for batch_idx, (data, img, target) in enumerate(data_loader):
        data, img, target = data.to(device), img.to(device), target.to(device)
        output = model.forward(data, img)
        loss = F.mse_loss(output, target)
        loss_val += loss.item()
    loss_val = loss_val / len(data_loader)
    print("\nValidate set: Average loss: %.8f" % loss_val)
    validate_loss.append(loss_val)


def predict(model, data_loader, dataset_size):
    loss_val = 0
    model.eval()
    criterion = nn.MSELoss()
    with torch.no_grad():
        for batch_idx, (data, img, target) in enumerate(data_loader):
            output = model.forward(data, img)
            loss = criterion(output, target)
            loss_val += loss.item()
        loss_val /= dataset_size
        print("\nTest set: Average loss: %.8f" % loss_val)


def network_conv():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input = np.loadtxt('./data/input.csv', delimiter=',')
    output = np.loadtxt('./data/output.csv', delimiter=',')
    foil_name = np.genfromtxt('./data/img_path.csv', delimiter=',', dtype='str')

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
    input = (input - input_mean) / input_std
    output = (output - output_mean) / output_std

    print(sys.getsizeof(input)  / 1024 / 1024)
    print(sys.getsizeof(output) / 1024 / 1024)

    # seq = np.random.permutation(input.shape[0])
    # seq.tofile('./data/seq.txt', sep=',')
    seq = np.loadtxt('./data/seq.txt', delimiter=',', dtype=int)

    if options['method'] == 'train':
        # 构建网络
        model = Net(options['in_feature'], out_feature)
        model.to(device)
        print(model)

        # 训练数据
        train_dataset = CustomDataset(input, foil_name, output, seq, 'train')
        train_dataset_size = len(train_dataset)
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=options['train_batch_size'],
                                                        num_workers=1,
                                                        pin_memory=True)

        # 验证数据
        validate_dataset = CustomDataset(input, foil_name, output, seq, 'validate')
        validate_data_loader = torch.utils.data.DataLoader(validate_dataset,
                                                           batch_size=options['validate_batch_size'],
                                                           num_workers=1,
                                                           pin_memory=True)
        # 学习率指数衰减
        # optimizer = torch.optim.SGD(model.parameters(), lr=options['learning_rate'],
        #                             momentum=0.9,
        #                             weight_decay=options['weight_decay'])
        optimizer = torch.optim.Adam(model.parameters(), lr=options['learning_rate'])
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, verbose=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)

        for epoch in range(options['epochs']):
            train(epoch, model, device, train_data_loader, optimizer, train_dataset_size)
            validate(model, device, validate_data_loader, optimizer)
            print('learning rate: ', optimizer.param_groups[0]['lr'])
            scheduler.step(validate_loss[-1])

        model_path = 'model/model-%d-%.4f-%.4f.pt' % (options['epochs'], train_loss[-1], validate_loss[-1])
        figuer_path = 'model/model-%d-%.4f-%.4f.png' % (options['epochs'], train_loss[-1], validate_loss[-1])
        torch.save(model, model_path)

        plt.gcf().set_size_inches(8, 6)
        plt.gcf().set_dpi(150)
        plt.plot(np.linspace(0, options['epochs'], options['epochs']).tolist(), train_loss, label='train')
        plt.plot(np.linspace(0, options['epochs'], options['epochs']).tolist(), validate_loss, label='validate')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(figuer_path)
        plt.show()

    elif options['method'] == 'test':
        model = torch.load('./model/model-100-0.2381-0.3610.pt')
        print(model)
        test_dataset = CustomDataset(input, foil_name, output, seq, 'test')
        test_dataset_size = len(test_dataset)
        test_data_loader = torch.utils.data.DataLoader(test_dataset)
        predict(model, test_data_loader, test_dataset_size)
