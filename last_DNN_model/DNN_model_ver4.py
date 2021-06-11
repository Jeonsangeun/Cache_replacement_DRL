import random as rd
import torch
import torch.nn as nn
import collections

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else "cpu")

class Qnet4(nn.Module):
    def __init__(self, Num_packet, Num_file, F_packet, node, output_size):
        super(Qnet4, self).__init__()  # n x 5 x 160 -->5
        self.node = node
        self.Num_packet = Num_packet
        self.Num_file = Num_file
        self.F_packet = F_packet
        self.output_size = output_size
        conv1 = nn.Conv2d(2, 20, kernel_size=(1, self.Num_packet),
                          stride=(1, self.Num_packet))  # n x 20 x 5 x 20
        # pool1 = nn.MaxPool1d(1)  # n x 80 x 100

        conv2 = nn.Conv2d(20, 20, kernel_size=(4, 1), stride=(4, 1))  # n x 20 x 1 x 20
        # pool2 = nn.MaxPool1d(1)  # n x # 20 x 1

        self.conv_module = nn.Sequential(conv1, nn.ReLU(), conv2, nn.ReLU()).to(device)  # n x 20 x 20
        self.relu = nn.LeakyReLU(inplace=True).to(device)

        self.Lc1 = nn.Linear(1, 1).to(device)
        self.Lc2 = nn.Linear(1, 1).to(device)
        self.Lc3 = nn.Linear(1, 1).to(device)
        self.Lc4 = nn.Linear(1, 1).to(device)

        fc1 = nn.Linear(400, 400)
        fc2 = nn.Linear(400, self.output_size)
        fc3 = nn.Linear(self.output_size, self.output_size)
        self.fully_module1 = nn.Sequential(fc1, self.relu, fc2, self.relu, fc3).to(device)

        # fc4 = nn.Linear(20, 100)
        # fc5 = nn.Linear(100, 200)
        # fc6 = nn.Linear(200, self.output_size)
        # self.fully_module2 = nn.Sequential(fc4, self.relu, fc5, self.relu, fc6).to(device)

    def forward(self, x):
        x = x.to(device)
        prop = x[:, 5, :4] * 1
        x_s = x[:, :5] * 1
        x_s[:, 0] += self.Lc1(prop[:, 0].unsqueeze(1))
        x_s[:, 1] += self.Lc2(prop[:, 1].unsqueeze(1))
        x_s[:, 2] += self.Lc3(prop[:, 2].unsqueeze(1))
        x_s[:, 3] += self.Lc4(prop[:, 3].unsqueeze(1))
        sbs_s = x_s[:, :4].reshape([-1, 1, 4, self.F_packet])
        request_s = x_s[:, 4].reshape([-1, 1, 1, self.F_packet])
        request_s = torch.cat([request_s, request_s, request_s, request_s], dim=2)
        ss = torch.cat([sbs_s, request_s], dim=1)

        p_s = self.conv_module(ss)  # 4 x 5 * contents

        out = p_s.reshape(-1, 400)
        out = self.fully_module1(out)

        return out


def Predict_Qnet4(model, x):
    prop = x[:, 5, :4] * 1
    x_s = x[:, :5] * 1
    x_s[:, 0] += model.Lc(prop[:, 0].unsqueeze(1))
    x_s[:, 1] += model.Lc(prop[:, 1].unsqueeze(1))
    x_s[:, 2] += model.Lc(prop[:, 2].unsqueeze(1))
    x_s[:, 3] += model.Lc(prop[:, 3].unsqueeze(1))

    sbs_s = x_s[:, :4].reshape([-1, 1, 4, model.F_packet])
    request_s = x_s[:, 4].reshape([-1, 1, 1, model.F_packet])
    request_s = torch.cat([request_s, request_s, request_s, request_s], dim=2)
    ss = torch.cat([sbs_s, request_s], dim=1)

    p_s = model.conv_module(ss)  # 4 x 5 * contents

    out = p_s.reshape(-1, 400)
    out = model.fully_module1(out)

    return out