import random as rd
import torch
import torch.nn as nn
import collections

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else "cpu")

class Qnet(nn.Module):
    def __init__(self, Num_packet, Num_file, F_packet, node, output_size):
        super(Qnet, self).__init__()  # n x 5 x 160
        self.node = node
        self.Num_packet = Num_packet
        self.Num_file = Num_file
        self.F_packet = F_packet
        self.output_size = output_size
        conv1 = nn.Conv1d(5, 20, kernel_size=self.Num_packet, stride=self.Num_packet)
        pool1 = nn.MaxPool1d(1)  # n x 5 x 20
        conv2 = nn.Conv1d(20, 80, kernel_size=20, stride=1)
        pool2 = nn.MaxPool1d(1)  # n x 20 x 5
        self.conv_module1 = nn.Sequential(conv1, nn.ReLU(), pool1).to(device)  # n x 20 x 5
        self.conv_module2 = nn.Sequential(conv2, nn.ReLU(), pool2).to(device)  # n x 20 x 5
        self.relu = nn.LeakyReLU(inplace=True).to(device)

        self.Lc1 = nn.Linear(1, 1).to(device)
        self.Lc2 = nn.Linear(1, 1).to(device)
        self.Lc3 = nn.Linear(1, 1).to(device)
        self.Lc4 = nn.Linear(1, 1).to(device)

        self.fc1 = nn.Linear(self.node, self.node).to(device)
        self.fc2 = nn.Linear(self.node, self.output_size).to(device)
        self.fc3 = nn.Linear(self.output_size, self.output_size).to(device)

    def forward(self, x):
        x = x.to(device)
        prop = x[:, 5, :4] * 1
        x_s = x[:, :5] * 1
        x_s[:, 0] += self.Lc1(prop[:, 0].unsqueeze(1))
        x_s[:, 1] += self.Lc2(prop[:, 1].unsqueeze(1))
        x_s[:, 2] += self.Lc3(prop[:, 2].unsqueeze(1))
        x_s[:, 3] += self.Lc4(prop[:, 3].unsqueeze(1))
        ss = self.conv_module1(x_s)
        # ss = torch.cat(list(torch.chunk(ss, 20, dim=2)), dim=1)
        ss = torch.cat(list(torch.split(ss, self.Num_file, dim=2)), dim=1)
        ss = ss.view(-1, 20, 20)
        s_out = self.conv_module2(ss)
        s_out = s_out.view(-1, self.node)

        s_out = self.relu(self.fc1(s_out))
        s_out = self.relu(self.fc2(s_out))
        s_out = self.fc3(s_out)
        return s_out


def Predict_Qnet(model, x):
    prop = x[:, 5, :4] * 1
    x_s = x[:, :5] * 1
    x_s[:, 0] += model.Lc1(prop[:, 0].unsqueeze(1))
    x_s[:, 1] += model.Lc2(prop[:, 1].unsqueeze(1))
    x_s[:, 2] += model.Lc3(prop[:, 2].unsqueeze(1))
    x_s[:, 3] += model.Lc4(prop[:, 3].unsqueeze(1))
    ss = model.conv_module1(x_s)
    ss = torch.cat(list(torch.split(ss, model.Num_file, dim=2)), dim=1)
    ss = ss.view(-1, 20, 20)
    s_out = model.conv_module2(ss)
    s_out = s_out.view(-1, model.node)

    s_out = model.relu(model.fc1(s_out))
    s_out = model.relu(model.fc2(s_out))
    s_out = model.fc3(s_out)

    return s_out
