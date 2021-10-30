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
    'weight_decay': 1e-7,
    'epochs': 500,
    'train_batch_size': 64,
    'validate_batch_size': 1000,
    'log_interval': 10,
}


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.block1 = nn.Sequential(
            nn.Linear(5, 256),
            nn.Linear(256, 120000),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(1, 8, (7, 7), padding='same'),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, (3, 3), padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(32, 16, (2, 2), padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, (1, 1), padding='same'),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )

    def forward(self, data, img):
        # MLP for parameters
        data = self.block1(data)
        data = data.reshape([-1, 1, 200, 600])
        data = self.block2(data)

        # Convolution network for airfoil image
        img = self.block2(img)

        # Combine two models to one
        x = torch.cat((data, img), dim=1)
        x = self.block3(x)
        return x


class CustomDataset(Dataset):
    def __init__(self, input_data, img, seq, type):
        train_count = int(len(input_data) * 0.7)
        validate_count = int(len(input_data) * 0.2)
        if type == 'train':
            seq = seq[:train_count]
        elif type == 'validate':
            seq = seq[train_count:train_count + validate_count]
        elif type == 'test':
            seq = seq[train_count + validate_count:]
        self.source = input_data[seq, :]
        img = img[seq]
        self.output_img = img
        self.input_img = img[:, 0]
        self.transform = transforms.ToTensor()

    def __getitem__(self, idx):
        input_img = cv2.imread('./data/img/foil/' + self.input_img[idx] + '.bmp', flags=0)
        norm_input_img = cv2.normalize(input_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        output_img = cv2.imread('./data/img/' + self.output_img[idx, 0] + '/' + self.output_img[idx, 1] + '.bmp', flags=0)
        norm_output_img = cv2.normalize(output_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return torch.Tensor(self.source[idx]), self.transform(norm_input_img), \
               self.transform(norm_output_img)

    def __len__(self):
        return len(self.source)


def train(epoch, model, device, data_loader, optimizer, dataset_size):
    model.train()
    loss_val = 0
    for batch_idx, (data, input_img, output_img) in enumerate(data_loader):
        data, input_img, output_img = data.to(device), input_img.to(device), output_img.to(device)
        optimizer.zero_grad()
        output = model(data, input_img)
        loss = F.l1_loss(output, output_img)
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


def validate(model, device, data_loader, optimizer):
    loss_val = 0
    model.eval()
    for batch_idx, (data, input_img, output_img) in enumerate(data_loader):
        data, input_img, output_img = data.to(device), input_img.to(device), output_img.to(device)
        output = model.forward(data, input_img)
        loss = F.l1_loss(output, output_img)
        loss_val += loss.item()
    loss_val = loss_val / len(data_loader)
    print("\nValidate set: Average loss: %.8f" % loss_val)
    validate_loss.append(loss_val)


def predict(model, data_loader, dataset_size):
    loss_val = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, input_img, output_img) in enumerate(data_loader):
            output = model.forward(data, input_img)
            loss = F.l1_loss(output, output_img)
            loss_val += loss.item()
        loss_val /= dataset_size
        print("\nTest set: Average loss: %.8f" % loss_val)


def network_conv():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input = np.loadtxt('./data/input.csv', delimiter=',')
    img_path = np.loadtxt('./data/img_path.csv', delimiter=',', dtype='str')

    # 划分输出数据

    # 归一化输入与输出
    input_mean = np.mean(input, axis=0)
    input_std = np.std(input, axis=0)
    input = (input - input_mean) / input_std

    # seq = np.random.permutation(input.shape[0])
    # seq.tofile('./data/seq.txt', sep=',')
    seq = np.loadtxt('./data/seq.txt', delimiter=',', dtype=int)

    if options['method'] == 'train':
        # 构建网络
        model = Net()
        model.to(device)
        print(model)

        # 训练数据
        train_dataset = CustomDataset(input, img_path, seq, 'train')
        train_dataset_size = len(train_dataset)
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=options['train_batch_size'],
                                                        num_workers=1,
                                                        pin_memory=True)

        # 验证数据
        validate_dataset = CustomDataset(input, img_path, seq, 'validate')
        validate_data_loader = torch.utils.data.DataLoader(validate_dataset,
                                                           batch_size=options['validate_batch_size'],
                                                           num_workers=1,
                                                           pin_memory=True)
        # 学习率指数衰减
        # optimizer = torch.optim.SGD(model.parameters(), lr=options['learning_rate'],
        #                             momentum=0.9,
        #                             weight_decay=options['weight_decay'])
        optimizer = torch.optim.Adam(model.parameters(), lr=options['learning_rate'])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, verbose=True)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)

        for epoch in range(options['epochs']):
            train(epoch, model, device, train_data_loader, optimizer, train_dataset_size)
            validate(model, device, validate_data_loader, optimizer)
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
        test_dataset = CustomDataset(input, img_path, seq, 'test')
        test_dataset_size = len(test_dataset)
        test_data_loader = torch.utils.data.DataLoader(test_dataset)
        predict(model, test_data_loader, test_dataset_size)
