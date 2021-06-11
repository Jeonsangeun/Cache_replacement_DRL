import random as rd
import torch
import torch.nn as nn
import collections

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else "cpu")

class Qnet2(nn.Module):
    def __init__(self, Num_packet, Num_file, F_packet, node, output_size):
        super(Qnet2, self).__init__()  # n x 5 x 160
        self.node = node
        self.Num_packet = Num_packet
        self.Num_file = Num_file
        self.F_packet = F_packet
        self.output_size = output_size
        conv1 = nn.Conv1d(5, 10, kernel_size=self.Num_packet, stride=self.Num_packet)
        conv1_1 = nn.Conv1d(10, 10, kernel_size=1, stride=1)
        pool1 = nn.MaxPool1d(1)  # n x 20 x 20
        conv2 = nn.Conv1d(self.F_packet, 80, kernel_size=1, stride=1)  # n x 80 x 20
        conv2_1 = nn.Conv1d(80, 40, kernel_size=1, stride=1)  # n x 80 x 20
        pool2 = nn.MaxPool1d(1)  # n x 20 x 20
        self.conv_module1 = nn.Sequential(conv1, nn.ReLU(),
                                          conv1_1, nn.ReLU(), pool1).to(device)
        self.conv_module2 = nn.Sequential(conv2, nn.ReLU(),
                                          conv2_1, nn.ReLU(), pool2).to(device)  # n x 80 x 4

        self.Lc1 = nn.Linear(1, 1).to(device)
        self.Lc2 = nn.Linear(1, 1).to(device)
        self.Lc3 = nn.Linear(1, 1).to(device)
        self.Lc4 = nn.Linear(1, 1).to(device)

        fc1 = nn.Linear(self.node, self.node)
        fc2 = nn.Linear(self.node, self.node)
        fc3 = nn.Linear(self.node, self.output_size)
        self.fc1_module = nn.Sequential(fc1, nn.ReLU(), fc2, nn.ReLU(), fc3).to(device)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         nn.init.kaiming_uniform_(m.weight.data)
        #         m.bias.data.fill_(0)
        #
        #     elif isinstance(m, nn.Linear):
        #         nn.init.kaiming_uniform_(m.weight.data)
        #         m.bias.data.fill_(0)

    def forward(self, x):
        x = x.to(device)
        prop = x[:, 5, :4] * 1
        x_s = x[:, :5] * 1
        x_s[:, 0] += self.Lc1(prop[:, 0].unsqueeze(1))
        x_s[:, 1] += self.Lc2(prop[:, 1].unsqueeze(1))
        x_s[:, 2] += self.Lc3(prop[:, 2].unsqueeze(1))
        x_s[:, 3] += self.Lc4(prop[:, 3].unsqueeze(1))

        ss_1 = self.conv_module1(x_s)
        ss_2 = self.conv_module2(torch.transpose(x_s, 1, 2))

        # ss = torch.cat(list(torch.chunk(ss, 20, dim=2)), dim=1)
        # ss = torch.cat(list(torch.split(ss, env.Num_file, dim=2)), dim=1)
        ss_1 = ss_1.view(-1, 200)
        ss_2 = ss_2.view(-1, 200)

        s_out = torch.cat([ss_1, ss_2], dim=1)
        out = self.fc1_module(s_out)
        return out


def Predict_Qnet2(model, x):
    x = x.to(device)
    prop = x[:, 5, :4] * 1
    x_s = x[:, :5] * 1
    x_s[:, 0] += model.Lc1(prop[:, 0].unsqueeze(1))
    x_s[:, 1] += model.Lc2(prop[:, 1].unsqueeze(1))
    x_s[:, 2] += model.Lc3(prop[:, 2].unsqueeze(1))
    x_s[:, 3] += model.Lc4(prop[:, 3].unsqueeze(1))

    ss_1 = model.conv_module1(x_s)
    ss_2 = model.conv_module2(torch.transpose(x_s, 1, 2))

    ss_1 = ss_1.view(-1, 200)
    ss_2 = ss_2.view(-1, 200)

    s_out = torch.cat([ss_1, ss_2], dim=1)
    out = model.fc1_module(s_out)
    return out