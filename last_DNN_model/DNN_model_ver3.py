import random as rd
import torch
import torch.nn as nn
import collections

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else "cpu")

class Qnet3(nn.Module):
    def __init__(self, Num_packet, Num_file, F_packet, node, output_size):
        super(Qnet3, self).__init__()  # n x 5 x 160
        self.node = node
        self.Num_packet = Num_packet
        self.Num_file = Num_file
        self.F_packet = F_packet
        self.output_size = output_size
        conv1 = nn.Conv1d(5, 5, kernel_size=self.Num_packet, stride=self.Num_packet) # 5 * packet*file
        batch1 = nn.BatchNorm1d(10)  # n x 10 x file

        conv2 = nn.Conv1d(8, 8, kernel_size=5, stride=5) # packet x 5*file
        batch2 = nn.BatchNorm1d(16)  # n x 16 x file

        self.conv_module1_p = nn.Sequential(conv1, nn.ReLU()).to(device)  # n x 20 x 5
        self.conv_module2_p = nn.Sequential(conv2, nn.ReLU()).to(device)  # n x 20 x 5

        self.conv_module1_l = nn.Sequential(conv1, batch1, nn.ReLU()).to(device)  # n x 20 x 5
        self.conv_module2_l = nn.Sequential(conv2, batch2, nn.ReLU()).to(device)  # n x 20 x 5

        self.relu = nn.LeakyReLU(inplace=True).to(device)

        self.Lc1 = nn.Linear(1, 1).to(device)
        self.Lc2 = nn.Linear(1, 1).to(device)
        self.Lc3 = nn.Linear(1, 1).to(device)
        self.Lc4 = nn.Linear(1, 1).to(device)

        self.fc1 = nn.Linear(self.node, self.node).to(device)
        self.fc2 = nn.Linear(self.node, self.output_size).to(device)

        # self.bn1 = nn.BatchNorm1d(self.node)
        # self.bn2 = nn.BatchNorm1d(self.output_size)

        self.fc3 = nn.Linear(self.output_size, self.output_size).to(device)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         nn.init.kaiming_normal_(m.weight.data) # Kaming He Initialization
        #         m.bias.data.fill_(0)                # 편차를 0으로 초기화

        #     elif isinstance(m, nn.Linear):
        #         nn.init.kaiming_normal_(m.weight.data) # Kaming He Initialization
        #         m.bias.data.fill_(0)                # 편차를 0으로 초기화

    def forward(self, x):
        x = x.to(device)
        prop = x[:, 5, :4] * 1
        x_s = x[:, :5] * 1
        x_s[:, 0] += self.Lc1(prop[:, 0].unsqueeze(1))
        x_s[:, 1] += self.Lc2(prop[:, 1].unsqueeze(1))
        x_s[:, 2] += self.Lc3(prop[:, 2].unsqueeze(1))
        x_s[:, 3] += self.Lc4(prop[:, 3].unsqueeze(1))
        out_sbs = self.conv_module1_p(x_s)
        # ss = torch.cat(list(torch.chunk(ss, 20, dim=2)), dim=1)
        out_p = torch.cat(list(torch.split(x_s, self.Num_packet, dim=2)), dim=1)
        out_p = torch.transpose(out_p, 1, 2)
        out_p = self.conv_module2_p(out_p)


        out_sbs = out_sbs.view(-1, 5*self.Num_file)
        out_p = out_p.view(-1, 8*self.Num_file)

        s_out = torch.cat([out_sbs, out_p], dim=1)

        s_out = self.relu(self.fc1(s_out))
        s_out = self.relu(self.fc2(s_out))
        s_out = self.fc3(s_out)
        return s_out


def Predict_Qnet3(model, x):
    prop = x[:, 5, :4] * 1
    x_s = x[:, :5] * 1
    x_s[:, 0] += model.Lc1(prop[:, 0].unsqueeze(1))
    x_s[:, 1] += model.Lc2(prop[:, 1].unsqueeze(1))
    x_s[:, 2] += model.Lc3(prop[:, 2].unsqueeze(1))
    x_s[:, 3] += model.Lc4(prop[:, 3].unsqueeze(1))
    out_sbs = model.conv_module1_p(x_s)
    # ss = torch.cat(list(torch.chunk(ss, 20, dim=2)), dim=1)
    out_p = torch.cat(list(torch.split(x_s, model.Num_packet, dim=2)), dim=1)
    out_p = torch.transpose(out_p, 1, 2)
    out_p = model.conv_module2_p(out_p)

    out_sbs = out_sbs.view(-1, 5 * model.Num_file)
    out_p = out_p.view(-1, 8 * model.Num_file)

    s_out = torch.cat([out_sbs, out_p], dim=1)

    s_out = model.relu((model.fc1(s_out)))
    s_out = model.relu((model.fc2(s_out)))
    s_out = model.fc3(s_out)
    return s_out