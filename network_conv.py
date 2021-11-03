import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2
from EarlyStopping import EarlyStopping

train_loss = []
validate_loss = []
options = {
    'method': 'train',
    'metrics': ['cosine', 'sine', 'limit'],
    'in_feature': 5,
    'learning_rate': 1e-3,
    'weight_decay': 1e-8,
    'epochs': 500,
    'train_batch_size': 64,
    'validate_batch_size': 128,
    'log_interval': 10,
    'model_name': {'cosine': 'cosine-500-0.0341-0.1260.pt',
                   'sine': 'sine-500-0.0387-0.1164.pt',
                   'limit': 'limit-500-0.0003-0.0036.pt'}
}


class Net(nn.Module):
    def __init__(self, out_feature):
        super(Net, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 8, (3, 3), (2, 2)), nn.MaxPool2d((2, 2), (2, 2)), nn.BatchNorm2d(8), nn.ReLU(),
            nn.Conv2d(8, 16, (3, 3), (2, 2)), nn.MaxPool2d((2, 2), (2, 2)), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, (3, 3), (2, 2)), nn.MaxPool2d((2, 2), (2, 2)), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3)), nn.MaxPool2d((2, 2), (2, 2)), nn.BatchNorm2d(32),
            nn.Flatten(),
        )

        self.block2 = nn.Sequential(
            nn.Linear(5, 128), nn.ReLU()
        )

        self.block3 = nn.Sequential(
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
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
        loss = F.huber_loss(output, target,
                            delta=float(options['learning_rate']) / float(optimizer.param_groups[0]['lr']))
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
    for batch_idx, (data, img, target) in enumerate(data_loader):
        data, img, target = data.to(device), img.to(device), target.to(device)
        output = model(data, img)
        loss = F.huber_loss(output, target,
                            delta=float(options['learning_rate']) / float(optimizer.param_groups[0]['lr']))
        loss_val += loss.item()
    loss_val = loss_val / len(data_loader)
    print("\nValidate set: Average loss: %.8f" % loss_val)
    validate_loss.append(loss_val)


def predict(model, device, data_loader):
    model.eval()
    output_list = []
    with torch.no_grad():
        for batch_idx, (data, img, _) in enumerate(data_loader):
            data, img = data.to(device), img.to(device)
            output = model(data, img)
            output_list.append(output.data.cpu().numpy())
    return output_list


def save_model(model, metric, epochs):
    path = './model/%s-%d-%.4f-%.4f' % (metric, epochs, train_loss[-1], validate_loss[-1])
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


def network_conv():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    result = []

    for metric in options['metrics']:
        global train_loss, validate_loss
        train_loss = []
        validate_loss = []
        input = np.loadtxt('./data/input.csv', delimiter=',')
        output = np.loadtxt('./data/output.csv', delimiter=',')
        img_path = np.loadtxt('./data/img_path.csv', delimiter=',', dtype='str')

        # 划分输出数据
        out_feature = 0
        if metric == 'cosine':
            out_feature = 31
            output = output[:, :31]
        elif metric == 'sine':
            out_feature = 30
            output = output[:, 31:61]
        elif metric == 'limit':
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

        # seq = np.random.permutation(input.shape[0])
        # seq.tofile('./data/seq.txt', sep=',')
        seq = np.loadtxt('./data/seq.txt', delimiter=',', dtype=int)

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
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.985)

            early_stopping = EarlyStopping()
            for epoch in range(options['epochs']):
                train(epoch, model, device, train_data_loader, optimizer, train_dataset_size)
                validate(model, device, validate_data_loader, optimizer)
                print('learning rate: %.8f' % optimizer.param_groups[0]['lr'])
                scheduler.step()
                # if early_stopping(validate_loss[-1], model):
                #     model_path = save_model(model, metric, epoch + 1)
                #     print('End training, model saved in %s' % model_path)
                #     break
            model_path = save_model(model, metric, options['epochs'])
            print('End training, model saved in %s' % model_path)

        elif options['method'] == 'test':
            model_name = options['model_name'][metric]
            model = torch.load('./model/' + model_name)
            model.to(device)
            print(model)
            test_dataset = CustomDataset(input, img_path, output, seq, 'test')
            test_dataset_size = len(test_dataset)
            test_data_loader = torch.utils.data.DataLoader(test_dataset, num_workers=4)
            out = predict(model, device, test_data_loader)
            predict_data = np.array(out).reshape(test_dataset_size, out_feature)
            predict_data = predict_data * output_std + output_mean
            result.append(predict_data)

    if options['method'] == 'test':
        data = np.column_stack([x for x in result])
        np.savetxt('./output/output.csv', data, delimiter=',')
