import random as rd
import torch
import torch.nn as nn
import collections

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else "cpu")

class Qnet1(nn.Module):
    def __init__(self, Num_packet, Num_file, F_packet, node, output_size):
        super(Qnet1, self).__init__()  # n x 5 x 160
        self.node = node
        self.Num_packet = Num_packet
        self.Num_file = Num_file
        self.F_packet = F_packet
        self.output_size = output_size
        conv1 = nn.Conv1d(5, 20, kernel_size=self.Num_packet, stride=self.Num_packet, padding=1, bias=True,
                          padding_mode='zeros')  # N * 10 * 20
        conv1_1 = nn.Conv1d(20, 40, kernel_size=self.Num_file, stride=1, padding=1, bias=True,
                            padding_mode='zeros')  # N * 40 * 3

        conv2 = nn.Conv1d(self.F_packet, 80, kernel_size=5, stride=5)  # n x 40 x 20
        conv2_1 = nn.Conv1d(80, 40, kernel_size=1, stride=1)

        pool = nn.MaxPool1d(1)  # n x 40 x 5
        self.conv_module1 = nn.Sequential(conv1, nn.ReLU(), conv1_1, nn.ReLU()).to(device)
        self.conv_module2 = nn.Sequential(conv2, nn.ReLU(), conv2_1, nn.ReLU()).to(device)

        self.Lc1 = nn.Linear(1, 1).to(device)
        self.Lc2 = nn.Linear(1, 1).to(device)
        self.Lc3 = nn.Linear(1, 1).to(device)
        self.Lc4 = nn.Linear(1, 1).to(device)

        SBS_layer = nn.Linear(40, 4)
        self.s_layer = nn.Sequential(SBS_layer, nn.Softmax(dim=1)).to(device)

        fc = nn.Linear(self.node, self.node)
        fc_out = nn.Linear(self.node, self.output_size)

        self.fc1 = nn.Sequential(fc, nn.ReLU(), fc_out).to(device)
        self.fc2 = nn.Sequential(fc, nn.ReLU(), fc_out).to(device)
        self.fc3 = nn.Sequential(fc, nn.ReLU(), fc_out).to(device)
        self.fc4 = nn.Sequential(fc, nn.ReLU(), fc_out).to(device)

    def forward(self, x):
        x = x.to(device)
        prop = x[:, 5, :4] * 1
        x_s = x[:, :5] * 1
        x_s[:, 0] += self.Lc1(prop[:, 0].unsqueeze(1))
        x_s[:, 1] += self.Lc2(prop[:, 1].unsqueeze(1))
        x_s[:, 2] += self.Lc3(prop[:, 2].unsqueeze(1))
        x_s[:, 3] += self.Lc4(prop[:, 3].unsqueeze(1))

        ss = self.conv_module1(x_s)
        out = ss.view(-1, self.node)

        ss_2 = self.conv_module2(torch.transpose(x_s, 1, 2))
        s_out = ss_2.view(-1, 40)
        sbs_w = self.s_layer(s_out)

        out1 = self.fc1(out + sbs_w[:, 0].unsqueeze(1))
        out2 = self.fc2(out + sbs_w[:, 1].unsqueeze(1))
        out3 = self.fc3(out + sbs_w[:, 2].unsqueeze(1))
        out4 = self.fc4(out + sbs_w[:, 3].unsqueeze(1))

        result = torch.cat([out1, out2, out3, out4], dim=1)
        return result


def Predict_Qnet1(model, x):
    prop = x[:, 5, :4] * 1
    x_s = x[:, :5] * 1
    x_s[:, 0] += model.Lc1(prop[:, 0].unsqueeze(1))
    x_s[:, 1] += model.Lc2(prop[:, 1].unsqueeze(1))
    x_s[:, 2] += model.Lc3(prop[:, 2].unsqueeze(1))
    x_s[:, 3] += model.Lc4(prop[:, 3].unsqueeze(1))

    ss = model.conv_module1(x_s)
    out = ss.view(-1, model.node)

    ss_2 = model.conv_module2(torch.transpose(x_s, 1, 2))
    s_out = ss_2.view(-1, 40)
    sbs_w = model.s_layer(s_out)

    out1 = model.fc1(out + sbs_w[:, 0])
    out2 = model.fc2(out + sbs_w[:, 1])
    out3 = model.fc3(out + sbs_w[:, 2])
    out4 = model.fc4(out + sbs_w[:, 3])

    result = torch.cat([out1, out2, out3, out4], dim=1)
    return result