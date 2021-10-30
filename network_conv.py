import sys

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2

train_loss = []
validate_loss = []
options = {
    'method': 'train',
    'type': 'cosine',
    'in_feature': 5,
    'learning_rate': 1e-3,
    'weight_decay': 1e-8,
    'epochs': 200,
    'train_batch_size': 64,
    'validate_batch_size': 256,
    'log_interval': 10,
}


class Net(nn.Module):
    def __init__(self, out_feature):
        super(Net, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 8, (3, 3), (2, 2)),
            nn.MaxPool2d((3, 3), (2, 2)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, (3, 3), (2, 2)),
            nn.MaxPool2d((3, 3), (2, 2)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, (3, 3), (2, 2)),
            nn.MaxPool2d((3, 3), (2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), (2, 2)),
            nn.MaxPool2d((3, 3), (2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.block2 = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU()
        )

        self.block3 = nn.Sequential(
            nn.Linear(96, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, out_features=out_feature)
        )

    def forward(self, data, img):
        img = self.block1(img)
        data = self.block2(data)
        x = torch.cat([img, data], dim=1)
        x = self.block3(x)
        return x


class CustomDataset(Dataset):
    def __init__(self, input_data, img, output_data, seq, type):
        train_count = int(len(input_data) * 0.7)
        validate_count = int(len(input_data) * 0.2)
        if type == 'train':
            seq = seq[:train_count]
        elif type == 'validate':
            seq = seq[train_count:train_count + validate_count]
        elif type == 'test':
            seq = seq[train_count + validate_count:]
        img = img[seq]
        self.input = input_data[seq, :]
        self.output = output_data[seq, :]
        self.img = img[:, 0]
        self.transform = transforms.ToTensor()

    def __getitem__(self, idx):
        img = cv2.imread('./data/img/foil/' + self.img[idx] + '.bmp', flags=0)
        norm_img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return torch.Tensor(self.input[idx]), self.transform(norm_img), torch.Tensor(self.output[idx])

    def __len__(self):
        return len(self.input)


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
            samples = (batch_idx + 1) * options['train_batch_size']
            samples = samples if samples < dataset_size else dataset_size
            print("\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.8f"
                  % (epoch + 1, samples, dataset_size, loss_val / len(data_loader)), end='')
    loss_val = loss_val / len(data_loader)
    train_loss.append(loss_val)


def validate(model, device, data_loader):
    loss_val = 0
    model.eval()
    for batch_idx, (data, img, target) in enumerate(data_loader):
        data, img, target = data.to(device), img.to(device), target.to(device)
        output = model(data, img)
        loss = F.mse_loss(output, target)
        loss_val += loss.item()
    loss_val = loss_val / len(data_loader)
    print("\nValidate set: Average loss: %.8f" % loss_val)
    validate_loss.append(loss_val)


def predict(model, device, data_loader, dataset_size):
    loss_val = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, input_img, output_img) in enumerate(data_loader):
            data, img, target = data.to(device), img.to(device), target.to(device)
            output = model(data, img)
            loss = F.mse_loss(output, target)
            loss_val += loss.item()
        loss_val /= dataset_size
        print("\nTest set: Average loss: %.8f" % loss_val)


def network_conv():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input = np.loadtxt('./data/input.csv', delimiter=',')
    output = np.loadtxt('./data/output.csv', delimiter=',')
    img_path = np.loadtxt('./data/img_path.csv', delimiter=',', dtype='str')

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
    input = (input - input_mean) / input_std
    output_mean = np.mean(output, axis=0)
    output_std = np.std(output, axis=0)
    output = (output - output_mean) / output_std

    seq = np.random.permutation(input.shape[0])
    seq.tofile('./data/seq.txt', sep=',')
    # seq = np.loadtxt('./data/seq.txt', delimiter=',', dtype=int)

    if options['method'] == 'train':
        # 构建网络
        model = Net(out_feature)
        model.to(device)
        print(model)

        # 训练数据
        train_dataset = CustomDataset(input, img_path, output, seq, 'train')
        train_dataset_size = len(train_dataset)
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=options['train_batch_size'],
                                                        num_workers=4,
                                                        pin_memory=True)

        # 验证数据
        validate_dataset = CustomDataset(input, img_path, output, seq, 'validate')
        validate_data_loader = torch.utils.data.DataLoader(validate_dataset,
                                                           batch_size=options['validate_batch_size'],
                                                           num_workers=4,
                                                           pin_memory=True)
        # 学习率指数衰减
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=options['learning_rate'],
                                     weight_decay=options['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

        for epoch in range(options['epochs']):
            train(epoch, model, device, train_data_loader, optimizer, train_dataset_size)
            validate(model, device, validate_data_loader)
            print('learning rate: %.8f' % optimizer.param_groups[0]['lr'])
            scheduler.step()

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
        test_dataset = CustomDataset(input, img_path, output, seq, 'test')
        test_dataset_size = len(test_dataset)
        test_data_loader = torch.utils.data.DataLoader(test_dataset)
        predict(model, device, test_data_loader, test_dataset_size)
