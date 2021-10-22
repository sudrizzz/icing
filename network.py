import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt

train_loss = []
validate_loss = []
options = {
    'method': 'test',
    'type': 'limit',
    'in_feature': 5,
    'learning_rate': 1e-4,
    'weight_decay': 1e-7,
    'epochs': 10,
    'train_batch_size': 50,
    'validate_batch_size': 50
}


class Net(torch.nn.Module):
    def __init__(self, in_feature, out_feature):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(in_feature, 10)
        self.fc2 = torch.nn.Linear(10, 100)
        self.fc3 = torch.nn.Linear(100, 500)
        self.fc4 = torch.nn.Linear(500, 250)
        self.fc5 = torch.nn.Linear(250, 125)
        self.fc6 = torch.nn.Linear(125, 70)
        self.fc7 = torch.nn.Linear(70, out_feature)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        x = F.leaky_relu(self.fc5(x))
        x = F.leaky_relu(self.fc6(x))
        x = self.fc7(x)
        return x


class CustomDataset(Dataset):
    def __init__(self, input_data, output_data, seq, type):
        train_count = int(len(input_data) * 0.7)
        validate_count = int(len(input_data) * 0.2)
        if type == 'train':
            seq = seq[:train_count]
        elif type == 'validate':
            seq = seq[train_count:train_count + validate_count]
        elif type == 'test':
            seq = seq[train_count + validate_count:]
        self.source = torch.Tensor(input_data[seq, :])
        self.target = torch.Tensor(output_data[seq, :])

    def __getitem__(self, idx):
        return self.source[idx], self.target[idx]

    def __len__(self):
        return len(self.source)


def train(epoch, model, data_loader, optimizer, dataset_size):
    model.train()
    loss_val = 0
    criterion = torch.nn.MSELoss()
    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model.forward(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_val += loss.item()
        if batch_idx % 10 == 0:
            print("\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.8f"
                  % (epoch, batch_idx * data.shape[0], dataset_size,
                     loss_val / (dataset_size / options['train_batch_size'])),
                  end='')
    loss_val /= (dataset_size / options['train_batch_size'])
    train_loss.append(loss_val)


def validate(model, data_loader, dataset_size):
    loss_val = 0
    model.eval()
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            output = model.forward(data)
            loss = criterion(output, target)
            loss_val += loss.item()
        loss_val /= (dataset_size / options['validate_batch_size'])
        print("\nValidate set: Average loss: %.8f" % loss_val)
        validate_loss.append(loss_val)


def predict(model, data_loader, dataset_size):
    loss_val = 0
    model.eval()
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            output = model.forward(data)
            loss = criterion(output, target)
            loss_val += loss.item()
        loss_val /= dataset_size
        print("\nTest set: Average loss: %.8f" % loss_val)


if __name__ == '__main__':
    input = np.loadtxt('data/input.csv', delimiter=' ')
    output = np.loadtxt('data/output.csv', delimiter=' ')

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

    # seq = np.random.permutation(input.shape[0])
    # seq.tofile('data/seq.txt', sep=' ')
    seq = np.loadtxt('data/seq.txt', delimiter=' ', dtype=int)

    if options['method'] == 'train':
        # 构建网络
        model = Net(options['in_feature'], out_feature)
        print(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=options['learning_rate'], weight_decay=options['weight_decay'])

        # 训练数据
        train_dataset = CustomDataset(input, output, seq, 'train')
        train_dataset_size = len(train_dataset)
        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=options['train_batch_size'])

        # 验证数据
        validate_dataset = CustomDataset(input, output, seq, 'validate')
        validate_dataset_size = len(validate_dataset)
        validate_data_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=options['validate_batch_size'], )

        for epoch in range(options['epochs']):
            train(epoch, model, train_data_loader, optimizer, train_dataset_size)
            validate(model, validate_data_loader, validate_dataset_size)

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
        model = torch.load('./model/model-10-0.1065-0.1095.pt')
        test_dataset = CustomDataset(input, output, seq, 'test')
        test_dataset_size = len(test_dataset)
        test_data_loader = torch.utils.data.DataLoader(test_dataset)
        predict(model, test_data_loader, test_dataset_size)
